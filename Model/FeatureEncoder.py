import torch
import torch.nn as nn

"""
Antes de definir o FeatureEncoder, é necessário decidir qual dimensão usar para o embedding de cada variável categórica. Usar sempre uma dimensão fixa desperdiça capacidade do modelo em variáveis com poucas categorias (ex: cor de pele, estado civil, sexo, etc) e pode ser insuficiente para variáveis com muitas categorias (ex: disciplina cursada). Por isso, a dimensão de cada embedding é calculada dinamicamente a partir do tamanho do vocabulário da variável (número de categorias distintas), usando uma heurística amplamente utilizada na prática, a de Guo & Berkhahn, escalando a dimensão com a raiz do número de categorias, e limitando a um teto para evitar embeddings desnecessariamente grandes. A fórmula foi extraída do paper "Entity Embeddings of Categorical Variables", de Cheng Guo e Felix Berkhahn.
"""


def suggested_dim(n_categorias: int, ceiling=512):
    """
    Evita ter embeddings desnecessariamente grandes para poucas categorias
    Utiliza a heurística fast.ai de Guo & Berkhahn
    """
    return int(min(ceiling, round(1.6 * n_categorias**0.56)))


"""
Este componente tem como finalidade codificar as features de entrada, para isso ele deve converter os campos qualitativos (categóricos) em vetores n-dimensionais, através de embeddings, usando a função suggested_dim para definir a dimensão de cada um, e preservar os campos numéricos (quantitativos), passando-os adiante sem transformação.

O FeatureEncoder é o ponto de entrada de cada bloco de dados do modelo (estudante, instituição e disciplina): ele recebe um dicionário com as features categóricas e um tensor com as features numéricas, e devolve um único vetor que une ambos os tipos de informação, servindo de entrada para os RegularBlocks seguintes. Por ser genérico (recebendo vocab_sizes e numeric_dims como parâmetros), o mesmo componente pode ser reaproveitado para codificar os três tipos de dado (estudante, instituição e disciplina), cada um com seu próprio conjunto de variáveis.
"""


class FeatureEncoder(nn.Module):
    """
    Recebe dois parâmetros:
    vocab_sizes: É um dicionário com a seguinte estrutura
        {
            'feature1': tamanho_do_vocabulario_da_feature1,
            'feature2': tamanho_do_vocabulario_da_feature2,
            ...
        }
    numeric_dims: Indica a dimensão das features numéricas (quantas features numéricas existem).

    A saída do encoder será um vetor que une todas as features categóricas (após passarem por embeddings) e as features numéricas (após passarem por uma camada linear)
    """

    def __init__(self, vocab_sizes: dict, numeric_dims: int):
        super().__init__()
        self.features = list(vocab_sizes.keys())
        self.embeddings = nn.ModuleDict(
            {
                feature: nn.Embedding(
                    num_embeddings=vocab_size, embedding_dim=suggested_dim(vocab_size)
                )
                for feature, vocab_size in vocab_sizes.items()
            }
        )
        self.output_dim = (
            sum(suggested_dim(vocab_size) for vocab_size in vocab_sizes.values())
            + numeric_dims
        )

    def forward(self, x_categorical: dict, x_numeric: torch.Tensor):
        """
        x_categorical: dicionário com as features categóricas, onde cada chave é o nome da feature e o valor é um tensor de índices
        x_numeric: tensor com as features numéricas
        """
        embedded_features = [
            self.embeddings[feature](x_categorical[feature])
            for feature in self.features
        ]
        embedded_features = torch.cat(embedded_features, dim=-1)
        return torch.cat([embedded_features, x_numeric], dim=-1)
