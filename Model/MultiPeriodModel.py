import torch
import torch.nn as nn
from Model.EvasaoModel import EvasaoModel
from Model.PreprocessingUnit import PreprocessingUnit
import pandas as pd


class MultiPeriodModel(nn.Module):
    """
    Conjunto de P modelos EvasaoModel, um para cada período (1..P).
    Os períodos são 1-indexados em toda a API pública.

    Esta classe contém apenas os modelos. O treino de cada período é feito
    por um Trainer (Model/Trainer.py), um por período.
    """

    def __init__(
        self,
        prepunit: PreprocessingUnit,
        n_periods: int = 14,
        dim: int = 32,
        dropout: float = 0.3,
        recency_P: int = 9,
    ):
        super().__init__()

        self.P = n_periods

        # ModuleList (e não list): registra os submodelos, de modo que
        # .parameters(), .state_dict(), .to(), .train()/.eval() funcionem.
        self.models = nn.ModuleList(
            [
                EvasaoModel(prepunit, P=recency_P, dim=dim, dropout=dropout)
                for _ in range(n_periods)
            ]
        )

    def _check_period(self, p: int):
        if not 1 <= p <= self.P:
            raise ValueError(f"Período inválido: {p} (esperado entre 1 e {self.P})")

    def get_model(self, p: int) -> EvasaoModel:
        self._check_period(p)
        return self.models[p - 1]

    def init_from_previous(self, p: int):
        """Warm start: copia os pesos do modelo do período p-1 para o do período p."""
        self._check_period(p)
        if p == 1:
            raise ValueError("O período 1 não possui modelo anterior")
        self.get_model(p).load_state_dict(self.get_model(p - 1).state_dict())

    def forward(
        self,
        x_student_categorical: dict,
        x_student_numeric: torch.Tensor,
        x_institutional_categorical: dict,
        x_institutional_numeric: torch.Tensor,
        x_discipline_categorical: dict,
        x_discipline_numeric: torch.Tensor,
        x_discipline_mask: torch.Tensor,
        x_semester_mask: torch.Tensor,
        period: int,
        return_attention: bool = False,
    ):
        return self.get_model(period)(
            x_student_categorical,
            x_student_numeric,
            x_institutional_categorical,
            x_institutional_numeric,
            x_discipline_categorical,
            x_discipline_numeric,
            x_discipline_mask,
            x_semester_mask,
            return_attention=return_attention,
        )

    def parameter_df(self) -> pd.DataFrame:
        dataframes = [model.parameter_df() for model in self.models]

        df_media = pd.concat(dataframes)

        df_media = df_media.groupby("componente", as_index=False).agg(
            {"total_params": "mean", "out_dim": "mean"}
        )

        return df_media
