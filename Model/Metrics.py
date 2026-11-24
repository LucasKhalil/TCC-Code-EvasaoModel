import math
import warnings
from dataclasses import dataclass, asdict

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from Model.Calibrator import brier_score, ece as ece_score

_NON_METRIC_FIELDS = ("label", "epoch")
_defined_steps = set()  # (run_id, prefix) já registrados com wandb.define_metric


def _acerto_na_classe(y_true, y_pred, c: int) -> float:
    """Acurácia dentro da classe c: fração das amostras com label c que foram previstas como c."""
    mask = y_true == c
    if not mask.any():
        return float("nan")
    return float((y_pred[mask] == c).mean())


@dataclass(frozen=True)
class Metrics:
    """
    Métricas de UMA época de treino ou validação.

    A classe 1 é EVADIDO (positiva) e a classe 0 é EGRESSO.

    Campos sem sufixo (acc, precision, recall, f1) são as métricas GERAIS:
    acc é a acurácia global e precision/recall/f1 são a média micro, isto é,
    os verdadeiros e falsos positivos das duas classes são somados antes de
    calcular a métrica. Em classificação binária, a média micro de
    precision, recall e f1 é igual à acurácia.

    Campos com sufixo `_0` / `_1` são por classe:
      - precision_c, recall_c, f1_c: a classe c é tratada como positiva
      - acc_c: acurácia dentro da classe c (acertos / amostras com label c),
        que por definição é igual a recall_c

    Calibração (quanto menor, melhor):
      - brier: média de (probabilidade - label)^2
      - ece: erro de calibração esperado (10 faixas de probabilidade)

    Valores indefinidos (ex.: AUC com uma só classe nos labels) ficam como NaN.
    """

    label: str  # "Train" ou "Val"
    epoch: int
    loss: float
    auc: float
    # calibração
    brier: float
    ece: float
    # gerais
    acc: float
    precision: float
    recall: float
    f1: float
    # classe 1 (EVADIDO)
    acc_1: float
    precision_1: float
    recall_1: float
    f1_1: float
    # classe 0 (EGRESSO)
    acc_0: float
    precision_0: float
    recall_0: float
    f1_0: float

    # ------------------------------------------------------------------ #
    # Construção a partir das predições
    # ------------------------------------------------------------------ #
    @classmethod
    def from_logits(
        cls,
        label: str,
        epoch: int,
        loss: float,
        logits: torch.Tensor,
        labels: torch.Tensor,
        threshold: float = 0.75,
        calibrador=None,
    ) -> "Metrics":
        """
        logits e labels: tensores 1-D na CPU, com todas as amostras da época.
        Se `calibrador` for informado, as probabilidades (e o threshold) passam a
        ser as calibradas; senão, sigmoid(logit).
        """
        if calibrador is not None:
            y_prob = np.asarray(calibrador.predict(logits.numpy()), dtype="float32")
            probs = torch.from_numpy(y_prob)
        else:
            probs = torch.sigmoid(logits)
            y_prob = probs.numpy()
        preds = (probs >= threshold).float()

        y_true = labels.numpy()
        y_pred = preds.numpy()

        # AUC é indefinida com uma só classe nos labels (versões novas do sklearn
        # emitem warning e devolvem NaN; as antigas lançavam ValueError)
        if len(set(y_true.tolist())) < 2:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(y_true, y_prob))

        per_class = dict(average=None, labels=[0, 1], zero_division=0)
        recall_c = recall_score(y_true, y_pred, **per_class)
        precision_c = precision_score(y_true, y_pred, **per_class)
        f1_c = f1_score(y_true, y_pred, **per_class)

        micro = dict(average="micro", zero_division=0)

        return cls(
            label=label,
            epoch=epoch,
            loss=float(loss),
            auc=auc,
            brier=brier_score(y_true, y_prob),
            ece=ece_score(y_true, y_prob),
            acc=float((preds == labels).float().mean().item()),
            precision=float(precision_score(y_true, y_pred, **micro)),
            recall=float(recall_score(y_true, y_pred, **micro)),
            f1=float(f1_score(y_true, y_pred, **micro)),
            acc_1=_acerto_na_classe(y_true, y_pred, 1),
            precision_1=float(precision_c[1]),
            recall_1=float(recall_c[1]),
            f1_1=float(f1_c[1]),
            acc_0=_acerto_na_classe(y_true, y_pred, 0),
            precision_0=float(precision_c[0]),
            recall_0=float(recall_c[0]),
            f1_0=float(f1_c[0]),
        )

    # ------------------------------------------------------------------ #
    # Exportação
    # ------------------------------------------------------------------ #
    def to_dict(self, prefix: str | None = None) -> dict:
        """Só as métricas numéricas (sem label/epoch), com chaves '<prefix>/<label>/<nome>'."""
        base = f"{prefix}/{self.label}" if prefix else self.label
        return {
            f"{base}/{k}": v
            for k, v in asdict(self).items()
            if k not in _NON_METRIC_FIELDS
        }

    def wandb_log(self, prefix: str | None = None):
        """
        Registra as métricas no wandb.

        Cada `prefix` (ex.: "<curso>/<periodo_p>") tem o próprio eixo de época
        (`<prefix>/epoch`), então vários modelos podem ser treinados na mesma
        execução sem conflito de `step`. Valores NaN não são enviados.
        """
        import wandb

        if wandb.run is None:
            raise RuntimeError("wandb não inicializado: chame wandb.init() antes.")

        step_key = f"{prefix}/epoch" if prefix else "epoch"
        registro = (wandb.run.id, prefix)
        if registro not in _defined_steps:
            wandb.define_metric(f"{prefix}/*" if prefix else "*", step_metric=step_key)
            _defined_steps.add(registro)

        data = {k: v for k, v in self.to_dict(prefix).items() if not math.isnan(v)}
        data[step_key] = self.epoch
        wandb.log(data)
