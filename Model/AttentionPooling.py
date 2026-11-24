import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Este componente implementa um mecanismo de atenção aditiva (Bahdanau) para agregar uma sequência de vetores (as disciplinas cursadas em um semestre, ou os semestres da trajetória de um estudante) em um único vetor de tamanho fixo, ponderando cada elemento pela sua relevância.

Um vetor de consulta (query) aprendível Q é comparado, via produto escalar, com uma projeção não linear (Linear + Tanh) de cada elemento da sequência, gerando os scores de atenção. A máscara garante que posições de padding (semestres ou disciplinas inexistentes) recebam score -inf e, portanto, peso zero após o softmax.

Para tornar o pooling mais robusto, o vetor resultante da atenção é somado a uma média simples (mascarada) dos elementos da sequência antes de passar por uma normalização (LayerNorm), isso evita que o modelo dependa exclusivamente dos pesos aprendidos pela atenção, servindo como uma espécie de atalho residual/skip connection para a informação agregada.
"""


class AttentionPooling(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3, softener: float = 0.2):
        super().__init__()
        self.output_dim = dim
        self.Q = nn.Parameter(torch.randn(dim) * softener)
        self.K = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        assert mask.dtype == torch.bool, (
            f"AttentionPooling espera mask do tipo torch.bool, "
            f"mas recebeu {mask.dtype}. Converta com mask.bool() antes de chamar o módulo."
        )
        # x: [..., seq, dim]  |  mask: [..., seq]
        orig_shape = x.shape
        dim = orig_shape[-1]
        seq = orig_shape[-2]

        x_flat = x.reshape(-1, seq, dim)
        # linhas em que NENHUMA posição é válida (ex: semestre de padding sem disciplinas)
        mask_flat = mask.reshape(-1, seq)
        valid_rows = mask_flat.any(dim=-1, keepdim=True)

        scores = self.K(x_flat) @ self.Q
        scores = scores.masked_fill(~mask_flat, float("-inf"))
        # nas linhas totalmente mascaradas, substitui -inf por 0 em TODAS as posições,
        # apenas para evitar softmax(-inf, ..., -inf) = NaN. O resultado dessas linhas
        # será descartado (zerado) logo abaixo, então o valor aqui é irrelevante.
        scores = scores.masked_fill(~valid_rows, 0.0)
        alpha = F.softmax(scores, dim=-1)

        # Skip Connection
        pooled = (alpha.unsqueeze(-1) * x_flat).sum(dim=1)
        simple_mean = (x_flat * mask_flat.unsqueeze(-1)).sum(dim=1) / mask_flat.sum(
            dim=1, keepdim=True
        ).clamp(min=1)
        pooled = self.norm(pooled + simple_mean)

        # zera explicitamente a saída (e os pesos de atenção) das linhas sem nenhuma posição válida,
        # garantindo que padding nunca produza um vetor "inventado" que vaze para as camadas seguintes
        pooled = pooled * valid_rows.float()
        alpha = alpha * valid_rows.float()

        pooled = pooled.reshape(*orig_shape[:-2], dim)
        alpha = alpha.reshape(*orig_shape[:-1])
        return pooled, alpha
