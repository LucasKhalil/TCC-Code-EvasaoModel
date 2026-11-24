import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import SEED, montar_caminho_arquivo

CAT_UNK_IDX = 0  # índice reservado: categoria desconhecida / ausente
PERIOD_COL = "periodo_atual_do_aluno"
SITUACOES_TREINO = ["EGRESSO", "EVADIDO"]
KEY_COLS = ["codigo_do_curso", "codigo_do_curriculo"]  # chave estudante -> instituição
FEATURES_PATH = Path("Model/features_modelo.json")
GROUPS = ("student", "institutional", "discipline")
META_KEY = "dados_preprocessamento"


# ---------------------------------------------------------------------- #
# Funções auxiliares (fit e apply separados)
# ---------------------------------------------------------------------- #
def load_features(path=FEATURES_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_vocab(series: pd.Series) -> dict:
    """Mapeia cada categoria para um índice >= 1 (0 reservado para UNK)."""
    valores = sorted(series.dropna().unique(), reverse=True)
    return {v: i + 1 for i, v in enumerate(valores)}


def fit_vocabs(df: pd.DataFrame, cols: list[str], shared: dict | None = None) -> dict:
    """Ajusta um vocabulário por coluna. `shared` reaproveita vocabulários já ajustados."""
    shared = shared or {}
    return {c: shared[c] if c in shared else build_vocab(df[c]) for c in cols}


def fit_stats(df: pd.DataFrame, cols: list[str]) -> dict:
    """Ajusta (média, desvio) por coluna."""
    stats = {}
    for col in cols:
        mean, std = df[col].mean(), df[col].std()
        std = std if std > 1e-6 else 1.0
        stats[col] = (mean, std)
    return stats


def apply_categorical(df: pd.DataFrame, cols: list[str], vocabs: dict) -> pd.DataFrame:
    """Aplica vocabulário já ajustado (val/test/inferência)."""
    df = df.copy()
    for col in cols:
        df[col] = df[col].map(vocabs[col]).fillna(CAT_UNK_IDX).astype("int64")
    return df


def apply_numeric(df: pd.DataFrame, cols: list[str], stats: dict) -> pd.DataFrame:
    """Aplica média/desvio já ajustados. NaN vira 0 (= média) após a padronização."""
    df = df.copy()
    for col in cols:
        mean, std = stats[col]
        df[col] = ((df[col] - mean) / std).fillna(0.0).astype("float32")
    return df


def to_native(obj):
    """Converte tipos numpy/pandas em tipos nativos para serialização em JSON."""
    if isinstance(obj, dict):
        return {
            (str(k) if isinstance(k, np.generic) else k): to_native(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple, set, np.ndarray, pd.Series)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# ---------------------------------------------------------------------- #
# PreprocessingUnit
# ---------------------------------------------------------------------- #
class PreprocessingUnit:
    """
    Pré-processa os dados de um curso para o treino por período.

    Modos de uso:
      - Treino:   PreprocessingUnit(name, df_e, df_d, df_i)
                  Faz o split estratificado, ajusta vocabulários/estatísticas
                  SOMENTE no treino, monta as disciplinas por período e grava tudo
                  em data/<name>/metadados.json.
      - Cache:    se o metadados.json já contém um pré-processamento válido
                  (mesmas features), o ajuste é restaurado e NÃO refeito; se os
                  DataFrames forem informados, os dados são recodificados com o
                  mesmo split e os mesmos vocabulários.
      - Inferência: PreprocessingUnit(name) com cache válido; use encode_data().
    """

    def __init__(
        self,
        name: str,
        df_e: pd.DataFrame = None,
        df_d: pd.DataFrame = None,
        df_i: pd.DataFrame = None,
        features: dict = None,
        force: bool = False,
        metadata_path=None,
    ):
        self.name = name
        self.features = features if features is not None else load_features()
        self.numeric_dims_student = len(self.features["STUDENT_NUM"])
        self.numeric_dims_institutional = len(self.features["INSTITUTIONAL_NUM"])
        self.numeric_dims_disciplines = len(self.features["DISCIPLINE_NUM"])

        self.caminho_metadados = (
            Path(metadata_path)
            if metadata_path is not None
            else Path(montar_caminho_arquivo("data", name)) / "metadados.json"
        )
        with open(self.caminho_metadados, "r", encoding="utf-8") as json_data:
            self.metadata = json.load(json_data)

        self.P = self.metadata["duracao_minima"]

        # chaves de junção que são categóricas institucionais (compartilham vocabulário)
        self.key_cols = [c for c in KEY_COLS if c in self.features["INSTITUTIONAL_CAT"]]

        self.vocabs = self.stats = None
        self.matriculas_treino = self.matriculas_val = None
        self.df_train_e = self.df_val_e = self.df_i = None
        self.df_train_d_por_periodo = self.df_val_d_por_periodo = None

        cached = (not force) and self._cache_valido()
        if cached:
            self._restore_state()

        sem_dados = df_e is None and df_d is None and df_i is None
        if sem_dados and cached:
            return  # modo inferência: apenas encode_data()

        for nome, df in (
            ("estudantes", df_e),
            ("disciplinas", df_d),
            ("institucional", df_i),
        ):
            if df is None:
                raise ValueError(f"DataFrame de {nome} não foi informado")

        self._build(df_e, df_d, df_i, fit=not cached)

    # ------------------------------------------------------------------ #
    # Construção dos dados de treino/validação
    # ------------------------------------------------------------------ #
    def _build(self, df_e, df_d, df_i, fit: bool):
        print("Processando Dados de Estudante")
        df_e = df_e[df_e["situacao"].isin(SITUACOES_TREINO)]

        if fit:
            treino, val = train_test_split(
                df_e["matricula"],
                test_size=0.2,
                random_state=SEED,
                stratify=df_e["situacao"],
            )
            self.matriculas_treino = treino.tolist()
            self.matriculas_val = val.tolist()

        df_train_e = df_e[df_e["matricula"].isin(self.matriculas_treino)]
        df_val_e = df_e[df_e["matricula"].isin(self.matriculas_val)]
        df_train_e = df_train_e.reset_index(drop=True)
        df_val_e = df_val_e.reset_index(drop=True)

        if fit:
            # Ajuste SOMENTE com dados de treino. Disciplinas: cada linha uma única vez.
            d_validas = self._disciplinas_validas(df_e, df_d)
            d_train = d_validas[
                d_validas["matricula_estudante"].isin(self.matriculas_treino)
            ]
            self._fit(df_train_e, df_i, d_train)

        print("Processando dados da instituição")
        self.df_i = self._encode_institutional(df_i)
        self.df_train_e = self._drop_orphans(
            self._encode_students(df_train_e), self.df_i
        )
        self.df_val_e = self._drop_orphans(self._encode_students(df_val_e), self.df_i)

        print("Processando Dados de Disciplinas")
        por_periodo = self.build_disciplinas_por_periodo(df_e, df_d)

        def split_encode(df, matriculas):
            df = df[df["matricula_estudante"].isin(matriculas)].reset_index(drop=True)
            return self._encode_disciplines(df)

        self.df_train_d_por_periodo = [
            split_encode(d, self.matriculas_treino) for d in por_periodo
        ]
        self.df_val_d_por_periodo = [
            split_encode(d, self.matriculas_val) for d in por_periodo
        ]

        if fit:
            self._save_state()

    def _fit(self, df_train_e, df_i, df_train_d):
        f = self.features
        vocabs, stats = {}, {}

        vocabs["institutional"] = fit_vocabs(df_i, f["INSTITUTIONAL_CAT"])
        # chaves (curso, currículo) usam o MESMO vocabulário no estudante e na
        # instituição, senão o .loc do EvasaoDataset não encontraria a linha
        shared = {c: vocabs["institutional"][c] for c in self.key_cols}
        vocabs["student"] = fit_vocabs(df_train_e, f["STUDENT_CAT"], shared)
        vocabs["discipline"] = fit_vocabs(df_train_d, f["DISCIPLINE_CAT"])

        stats["institutional"] = fit_stats(df_i, f["INSTITUTIONAL_NUM"])
        stats["student"] = fit_stats(df_train_e, f["STUDENT_NUM"])
        stats["discipline"] = fit_stats(df_train_d, f["DISCIPLINE_NUM"])

        self._set_state(vocabs, stats)

    def _set_state(self, vocabs: dict, stats: dict):
        self.vocabs, self.stats = vocabs, stats

        # atributos mantidos por compatibilidade com a versão anterior
        self.vocabs_student = vocabs["student"]
        self.vocabs_institutional = vocabs["institutional"]
        self.vocabs_discipline = vocabs["discipline"]
        self.stats_student = stats["student"]
        self.stats_institutional = stats["institutional"]
        self.stats_discipline = stats["discipline"]

        def sizes(group):  # +1 pelo índice UNK
            return {c: len(v) + 1 for c, v in vocabs[group].items()}

        self.vocab_sizes_student = sizes("student")
        self.vocab_sizes_institutional = sizes("institutional")
        self.vocab_sizes_discipline_full = sizes("discipline")

    # ------------------------------------------------------------------ #
    # Codificação
    # ------------------------------------------------------------------ #
    def _encode_students(self, df):
        f = self.features
        df = apply_categorical(df, f["STUDENT_CAT"], self.vocabs["student"])
        df = apply_numeric(df, f["STUDENT_NUM"], self.stats["student"])
        # chaves institucionais que não são features do estudante também precisam
        # estar codificadas (com o vocabulário institucional) para a junção
        extras = [
            c for c in self.key_cols if c not in f["STUDENT_CAT"] and c in df.columns
        ]
        if extras:
            df = apply_categorical(df, extras, self.vocabs["institutional"])
        return df

    def _encode_institutional(self, df):
        f = self.features
        df = apply_categorical(df, f["INSTITUTIONAL_CAT"], self.vocabs["institutional"])
        return apply_numeric(df, f["INSTITUTIONAL_NUM"], self.stats["institutional"])

    def _encode_disciplines(self, df):
        f = self.features
        df = apply_categorical(df, f["DISCIPLINE_CAT"], self.vocabs["discipline"])
        return apply_numeric(df, f["DISCIPLINE_NUM"], self.stats["discipline"])

    def _drop_orphans(self, df_e, df_i):
        """Remove estudantes cujo (curso, currículo) não existe no institucional."""
        if not all(c in df_e.columns and c in df_i.columns for c in KEY_COLS):
            return df_e
        validas = pd.MultiIndex.from_frame(df_i[KEY_COLS])
        ok = pd.MultiIndex.from_frame(df_e[KEY_COLS]).isin(validas)
        if not ok.all():
            warnings.warn(
                f"{(~ok).sum()} estudante(s) removido(s): "
                "curso/currículo ausente no df_institucional."
            )
            df_e = df_e[ok].reset_index(drop=True)
        return df_e

    def encode_data(self, df_e, df_d, df_i):
        """Mesmas transformações do treino, sem re-ajustar (inferência)."""
        if self.vocabs is None:
            raise RuntimeError("PreprocessingUnit sem ajuste (vocabulários ausentes)")
        df_i = self._encode_institutional(df_i)
        df_e = self._drop_orphans(self._encode_students(df_e), df_i)
        df_d = self._encode_disciplines(df_d)
        return df_e, df_d, df_i

    # ------------------------------------------------------------------ #
    # Disciplinas por período
    # ------------------------------------------------------------------ #
    def _disciplinas_validas(self, df_e, df_d):
        d = df_d[df_d["matricula_estudante"].isin(df_e["matricula"])]
        return d[d[PERIOD_COL] <= self.P]  # NaN <= P é False: descartado

    def build_disciplinas_por_periodo(self, df_e: pd.DataFrame, df_d: pd.DataFrame):
        """
        Para cada período p (1..P), retorna as disciplinas cursadas até o período p,
        apenas dos estudantes que cursaram alguma disciplina exatamente no período p
        (ou seja, que seguiam ativos naquele período). Sempre devolve P DataFrames
        (possivelmente vazios).
        """
        d = self._disciplinas_validas(df_e, df_d)

        frames = []
        for p in range(1, self.P + 1):
            ate_p = d[d[PERIOD_COL] <= p]
            ativos = ate_p.loc[ate_p[PERIOD_COL] == p, "matricula_estudante"].unique()
            frame = ate_p[ate_p["matricula_estudante"].isin(ativos)]
            frames.append(frame)

            print(f"  Período {p}: {len(ativos)} estudantes, {len(frame)} disciplinas")
            if len(ativos) == 0:
                warnings.warn(f"Nenhum estudante ativo no período {p}.")

        return frames

    def period_data(self, p: int) -> dict:
        """Dados do período p (1-indexado), prontos para Trainer.fit(**dados, ...)."""
        if not 1 <= p <= self.P:
            raise ValueError(f"Período inválido: {p} (esperado entre 1 e {self.P})")
        if self.df_train_e is None:
            raise RuntimeError("Dados não carregados: informe df_e, df_d e df_i.")
        return dict(
            df_i=self.df_i,
            df_train_e=self.df_train_e,
            df_train_d=self.df_train_d_por_periodo[p - 1],
            df_val_e=self.df_val_e,
            df_val_d=self.df_val_d_por_periodo[p - 1],
        )

    # ------------------------------------------------------------------ #
    # Persistência / cache
    # ------------------------------------------------------------------ #
    def _cache_valido(self) -> bool:
        d = self.metadata.get(META_KEY)
        return (
            bool(d)
            and all(f"categorias_{g}" in d for g in GROUPS)
            and "matriculas_treino" in d
            and d.get("features_modelo") == to_native(self.features)
        )

    def _restore_state(self):
        d = self.metadata[META_KEY]
        vocabs = {
            g: {
                c: {v: i + 1 for i, v in enumerate(cats)}
                for c, cats in d[f"categorias_{g}"].items()
            }
            for g in GROUPS
        }
        stats = {g: {c: tuple(s) for c, s in d[f"stats_{g}"].items()} for g in GROUPS}
        self.matriculas_treino = d["matriculas_treino"]
        self.matriculas_val = d["matriculas_val"]
        self._set_state(vocabs, stats)

    def _save_state(self):
        f = self.features
        d = {}
        for g in GROUPS:
            # lista ordenada de categorias: índice = posição + 1 (sobrevive ao JSON,
            # ao contrário de dicts com chaves numéricas, que viram string)
            d[f"categorias_{g}"] = {c: list(v) for c, v in self.vocabs[g].items()}
            d[f"stats_{g}"] = {c: list(s) for c, s in self.stats[g].items()}

        d["matriculas_treino"] = self.matriculas_treino
        d["matriculas_val"] = self.matriculas_val
        d["vocab_sizes_student"] = self.vocab_sizes_student
        d["dim_num_student"] = len(f["STUDENT_NUM"])
        d["vocab_sizes_institutional"] = self.vocab_sizes_institutional
        d["dim_num_institutional"] = len(f["INSTITUTIONAL_NUM"])
        d["vocab_sizes_discipline_full"] = self.vocab_sizes_discipline_full
        d["dim_num_discipline"] = len(f["DISCIPLINE_NUM"])
        d["features_modelo"] = f

        self.metadata[META_KEY] = to_native(d)

        self.caminho_metadados.parent.mkdir(parents=True, exist_ok=True)
        with open(self.caminho_metadados, "w", encoding="utf-8") as arquivo:
            json.dump(to_native(self.metadata), arquivo, indent=4, ensure_ascii=False)
