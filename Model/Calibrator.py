"""
Calibração de probabilidades (por período) e métricas de calibração.

O modelo produz logits. Um Calibrator é uma função monótona logit -> probabilidade
ajustada nos dados de validação do período, de modo que, entre os alunos com
probabilidade ~p, uma fração ~p realmente evada. Como é monótona, não altera a
ordenação dos alunos (o AUC se mantém, salvo empates da isotônica).
"""

import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

METODOS = ("platt", "isotonica")
_EPS = 1e-6


class Calibrator:
    """
    platt:     p = sigmoid(a * logit + b)  (2 parâmetros; robusto com poucos dados)
    isotonica: regressão isotônica sobre o logit (flexível; exige bastante dado)
    """

    def __init__(self, metodo: str = "platt", params: dict | None = None):
        if metodo not in METODOS:
            raise ValueError(f"Método de calibração inválido: {metodo} (use {METODOS})")
        self.metodo = metodo
        self.params = params or {}

    def fit(self, logits, y) -> "Calibrator":
        z = np.asarray(logits, dtype="float64").ravel()
        y = np.asarray(y).ravel().astype(int)
        if len(np.unique(y)) < 2:
            raise ValueError("A calibração exige as duas classes nos dados.")

        if self.metodo == "platt":
            # C alto = praticamente sem regularização
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(z.reshape(-1, 1), y)
            self.params = {"a": float(lr.coef_[0][0]), "b": float(lr.intercept_[0])}
        else:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(z, y)
            self.params = {
                "x": iso.X_thresholds_.tolist(),
                "y": iso.y_thresholds_.tolist(),
            }
        return self

    def predict(self, logits) -> np.ndarray:
        z = np.asarray(logits, dtype="float64")
        if self.metodo == "platt":
            p = expit(self.params["a"] * z + self.params["b"])
        else:
            p = np.interp(z, self.params["x"], self.params["y"])
        return np.clip(p, _EPS, 1 - _EPS)

    # persistência -------------------------------------------------------
    def to_dict(self) -> dict:
        return {"metodo": self.metodo, "params": self.params}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrator":
        return cls(d["metodo"], d["params"])

    def save(self, caminho):
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        caminho.write_text(json.dumps(self.to_dict(), indent=4), encoding="utf-8")

    @classmethod
    def load(cls, caminho) -> "Calibrator":
        return cls.from_dict(json.loads(Path(caminho).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------- #
# Métricas de calibração
# ---------------------------------------------------------------------- #
def _arrays(y_true, y_prob):
    return (
        np.asarray(y_true, dtype="float64").ravel(),
        np.asarray(y_prob, dtype="float64").ravel(),
    )


def _faixa(p: np.ndarray, n_bins: int) -> np.ndarray:
    return np.clip((p * n_bins).astype(int), 0, n_bins - 1)


def brier_score(y_true, y_prob) -> float:
    y, p = _arrays(y_true, y_prob)
    return float(np.mean((p - y) ** 2)) if y.size else float("nan")


def log_loss(y_true, y_prob) -> float:
    y, p = _arrays(y_true, y_prob)
    if not y.size:
        return float("nan")
    p = np.clip(p, _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y_true, y_prob, n_bins: int = 10) -> float:
    """Erro de calibração esperado: média ponderada de |freq. real - prob. média| por faixa."""
    y, p = _arrays(y_true, y_prob)
    if not y.size:
        return float("nan")
    faixa = _faixa(p, n_bins)
    total = 0.0
    for b in range(n_bins):
        m = faixa == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def curva_confiabilidade(y_true, y_prob, n_bins: int = 10) -> list[dict]:
    """Por faixa de probabilidade: nº de amostras, probabilidade média e frequência observada."""
    y, p = _arrays(y_true, y_prob)
    faixa = _faixa(p, n_bins)
    curva = []
    for b in range(n_bins):
        m = faixa == b
        if m.any():
            curva.append(
                {
                    "faixa": [round(b / n_bins, 2), round((b + 1) / n_bins, 2)],
                    "n": int(m.sum()),
                    "prob_media": round(float(p[m].mean()), 4),
                    "freq_observada": round(float(y[m].mean()), 4),
                }
            )
    return curva
