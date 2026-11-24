import torch.nn as nn

'''
O Bloco regular é o componente responsável por gerar as representações intermediárias do modelo em diversos pontos, e será utilizado repetidamente em várias partes do modelo (após cada FeatureEncoder, na saída do output_block, etc.).

Ele é composto por uma camada linear (que projeta a entrada para a dimensão interna desejada), seguida de uma LayerNorm (para estabilizar o treinamento), uma ativação GELU (não linearidade) e, por fim, Dropout (para regularização e prevenção de overfitting). Essa combinação de camadas forma um bloco simples, porém padronizado, que pode ser empilhado ou reutilizado em qualquer ponto da rede onde seja necessário transformar um vetor de entrada em uma nova representação de tamanho fixo.
'''

class RegularBlock(nn.Module):
    """
    Cria um bloco de camadas linear + GeLU + LayerNorm + Dropout
    """
    def __init__(self, in_features:int, out_features:int, dropout:float=0.3):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.layernorm = nn.LayerNorm(out_features)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output_dim = out_features

    def forward(self, x):
        x = self.linear(x)
        x = self.layernorm(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return x