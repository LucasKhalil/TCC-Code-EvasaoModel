import torch
import torch.nn as nn

"""
Como o modelo recebe uma sequência de semestres, é preciso informar à rede a posição relativa de cada semestre em relação ao presente. Diferentemente de um positional encoding tradicional (senoidal ou aprendido por índice absoluto), aqui a posição é expressa como recência: quão distante um semestre está do semestre mais recente da trajetória do estudante.

A distância é normalizada para o intervalo [0, 1], sendo 0 o semestre mais recente e 1 o mais antigo. Essa distância é projetada (Linear) para o mesmo espaço vetorial das representações de semestre e somada a elas, permitindo que o modelo capture padrões temporais (como uma queda progressiva de desempenho ao longo dos semestres) sem depender de uma arquitetura recorrente. Ao final, a máscara de semestres válidos é reaplicada para zerar qualquer valor introduzido em posições de padding.
"""


class RecencyPositionalEncoding(nn.Module):
    def __init__(self, dim: int, P: int):
        super().__init__()
        self.projection = nn.Linear(1, dim)
        self.P = P

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        assert mask.dtype == torch.bool, (
            f"AttentionPooling espera mask do tipo torch.bool, "
            f"mas recebeu {mask.dtype}. Converta com mask.bool() antes de chamar o módulo."
        )
        batch_size, _, _ = x.size()
        idx = torch.arange(self.P, device=x.device).float()
        dist = (self.P - 1 - idx) / max(1, self.P - 1)  # normaliza para [0,1]
        dist = dist.view(1, self.P, 1).expand(batch_size, self.P, 1)

        positional_encoding = self.projection(dist)
        x_encoded = x + positional_encoding
        x_encoded = x_encoded * mask.unsqueeze(-1)  # aplica a máscara
        return x_encoded
