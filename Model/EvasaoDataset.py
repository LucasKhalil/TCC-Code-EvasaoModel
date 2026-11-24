import torch
from torch.utils.data import Dataset

LABEL_MAP = {"ATIVO": -1, "EGRESSO": 0, "EVADIDO": 1}
LABEL_MAP_TRAIN = {"EGRESSO": 0, "EVADIDO": 1}


class EvasaoDataset(Dataset):
    def __init__(
        self,
        df_estudantes,
        df_institucional,
        df_disciplinas,
        student_cat,
        student_num,
        institutional_cat,
        institutional_num,
        discipline_cat,
        discipline_num,
        period_col="periodo_atual_do_aluno",
        train=True,
    ):

        if train:
            self.LABEL_MAP = LABEL_MAP_TRAIN
        else:
            self.LABEL_MAP = LABEL_MAP

        # Em caso de treino, só mantém estudantes com desfecho definido
        df_e = df_estudantes[
            df_estudantes["situacao"].isin(self.LABEL_MAP.keys())
        ].reset_index(drop=True)

        self.student_cat, self.student_num = student_cat, student_num
        self.institutional_cat, self.institutional_num = (
            institutional_cat,
            institutional_num,
        )
        self.discipline_cat, self.discipline_num = discipline_cat, discipline_num

        df_institucional = df_institucional.set_index(
            ["codigo_do_curso", "codigo_do_curriculo"]
        )
        df_institucional = df_institucional[
            ~df_institucional.index.duplicated(keep="first")
        ]
        self.institucional_por_curso = df_institucional

        # disciplinas agrupadas por aluno e, dentro do aluno, por período (ordem cronológica real)
        self.disciplinas_por_aluno = {
            matricula: grupo.sort_values(period_col)
            for matricula, grupo in df_disciplinas.groupby("matricula_estudante")
        }

        self.period_col = period_col

        self.alunos = df_e.reset_index(drop=True)

    def __len__(self):
        return len(self.alunos)

    def __getitem__(self, idx):
        aluno = self.alunos.iloc[idx]
        matricula = aluno["matricula"]

        # Bloco de busca do estudante
        x_cat_student = {
            c: torch.tensor(aluno[c], dtype=torch.long) for c in self.student_cat
        }
        x_num_student = torch.tensor(aluno[self.student_num].values.astype("float32"))

        # Bloco de busca da instituição
        chave = (aluno["codigo_do_curso"], aluno["codigo_do_curriculo"])
        inst = self.institucional_por_curso.loc[chave]
        x_cat_inst = {
            c: torch.tensor(
                (
                    chave[0]
                    if c == "codigo_do_curso"
                    else chave[1] if c == "codigo_do_curriculo" else inst[c]
                ),
                dtype=torch.long,
            )
            for c in self.institutional_cat
        }
        x_num_inst = torch.tensor(inst[self.institutional_num].values.astype("float32"))

        # Bloco de busca de disciplinas
        semestres = []
        disciplinas_aluno = self.disciplinas_por_aluno.get(matricula)
        if disciplinas_aluno is not None:
            for _, grupo_periodo in disciplinas_aluno.groupby(
                self.period_col, sort=False
            ):
                cat_semestre = {
                    c: torch.tensor(grupo_periodo[c].values, dtype=torch.long)
                    for c in self.discipline_cat
                }
                num_semestre = torch.tensor(
                    grupo_periodo[self.discipline_num].values.astype("float32")
                )
                semestres.append((cat_semestre, num_semestre))

        # groupby já preserva a ordem cronológica porque o df foi ordenado por period_col antes
        y = torch.tensor(self.LABEL_MAP[aluno["situacao"]], dtype=torch.float32)

        return {
            "student_cat": x_cat_student,
            "student_num": x_num_student,
            "institutional_cat": x_cat_inst,
            "institutional_num": x_num_inst,
            "semesters": semestres,  # lista de (dict_cat, tensor_num), tamanho variável
            "label": y,
        }
