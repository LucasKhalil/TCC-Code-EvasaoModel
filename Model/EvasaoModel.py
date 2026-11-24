import torch
import pandas as pd
import torch.nn as nn
from Model.RegularBlock import RegularBlock
from Model.FeatureEncoder import FeatureEncoder
from Model.AttentionPooling import AttentionPooling
from Model.RecencyPositionalEncoding import RecencyPositionalEncoding
from Model.PreprocessingUnit import PreprocessingUnit

"""
O EvasaoModel combina os componentes anteriores para processar a trajetória acadêmica completa de um estudante e estimar sua probabilidade de evasão. A arquitetura segue, hierarquicamente, a estrutura dos dados (estudante → semestres → disciplinas):

Nível de disciplina: cada disciplina cursada em cada semestre é codificada pelo FeatureEncoder e processada por um RegularBlock, produzindo uma representação intermediária por disciplina. Agregação por semestre: as representações das disciplinas de um mesmo semestre são combinadas por atenção (AttentionPooling), gerando um único vetor que resume o desempenho do estudante naquele semestre. Combinação com dados do estudante e institucionais: esse vetor é concatenado com as representações do estudante e da instituição (também processadas por FeatureEncoder + RegularBlock) e passa por um novo RegularBlock, formando a representação completa do semestre. Nível temporal: a sequência de representações de semestre recebe a codificação de recência (RecencyPositionalEncoding) e é agregada por um segundo AttentionPooling, produzindo um único vetor que resume toda a trajetória do estudante. Saída: esse vetor final passa por um RegularBlock seguido de uma camada linear (output_block), produzindo o logit da probabilidade de evasão.
"""


def move_batch_to_device(batch, device):
    def move(x):
        if isinstance(x, dict):
            return {k: move(v) for k, v in x.items()}
        return x.to(device)

    return {k: move(v) for k, v in batch.items()}


class EvasaoModel(nn.Module):
    def __init__(
        self,
        prepunit: PreprocessingUnit,
        P: int,
        dim: int = 32,
        dropout=0.3,
    ):
        super().__init__()

        # Encoders que convertem as entradas em vetores de dimensão fixa, que serão processados pelos blocos subsequentes
        self.student_encoder = FeatureEncoder(
            prepunit.vocab_sizes_student, prepunit.numeric_dims_student
        )
        self.institutional_encoder = FeatureEncoder(
            prepunit.vocab_sizes_institutional, prepunit.numeric_dims_institutional
        )
        self.discipline_encoder = FeatureEncoder(
            prepunit.vocab_sizes_discipline_full, prepunit.numeric_dims_disciplines
        )

        # Blocos que processam os vetores de dimensão fixa e produzem representações intermediárias
        self.student_block = RegularBlock(self.student_encoder.output_dim, dim, dropout)
        self.institutional_block = RegularBlock(
            self.institutional_encoder.output_dim, dim, dropout
        )

        # Bloco que processa as disciplinas, e em seguida aplica atenção para produzir uma representação agregada das disciplinas
        self.discipline_block = RegularBlock(
            self.discipline_encoder.output_dim, dim, dropout
        )
        self.discipline_pooling = AttentionPooling(dim, dropout, 0.2)

        # Bloco que processa a representação agregada das disciplinas, juntamente com as representações do estudante e da instituição, para produzir uma representação final do semestre
        self.semester_block = RegularBlock(
            self.discipline_pooling.output_dim
            + self.institutional_block.output_dim
            + self.student_block.output_dim,
            dim,
            dropout,
        )

        # Bloco que aplica codificação posicional baseada na recência, para capturar a ordem temporal dos semestres
        self.recency_encoding = RecencyPositionalEncoding(dim, P=P)

        # Bloco que aplica atenção para agregar as representações dos semestres, produzindo uma representação final do estudante
        self.semester_pooling = AttentionPooling(dim, dropout, 0.2)

        # Bloco final que produz a saída do modelo, que é a probabilidade de evasão
        self.output_block = nn.Sequential(
            RegularBlock(dim, dim, dropout), nn.Linear(dim, 1)
        )

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
        return_attention: bool = False,
    ):
        # Pré-processamento das entradas
        student_features = self.student_encoder(
            x_student_categorical, x_student_numeric
        )
        institutional_features = self.institutional_encoder(
            x_institutional_categorical, x_institutional_numeric
        )
        discipline_features = self.discipline_encoder(
            x_discipline_categorical, x_discipline_numeric
        )

        # Processamento das entradas pelos blocos correspondentes, produzindo representações intermediárias
        IR_student = self.student_block(student_features)
        IR_disciplines = self.discipline_block(discipline_features)
        IR_institutional = self.institutional_block(institutional_features)

        # Agregação das representações das disciplinas, produzindo uma representação agregada das disciplinas
        IR_disciplines, alpha_disciplines = self.discipline_pooling(
            IR_disciplines, x_discipline_mask
        )

        # Combina as representações do estudante, da instituição e das disciplinas, produzindo uma representação final do semestre
        _, S, _ = IR_disciplines.shape
        IR_student = IR_student.unsqueeze(1).expand(-1, S, -1)
        IR_institutional = IR_institutional.unsqueeze(1).expand(-1, S, -1)

        IR_semester = torch.cat([IR_student, IR_institutional, IR_disciplines], dim=-1)
        IR_semester = self.semester_block(IR_semester)
        IR_semester_encoded = self.recency_encoding(IR_semester, x_semester_mask)

        IR_history, alpha_semesters = self.semester_pooling(
            IR_semester_encoded, x_semester_mask
        )
        logits = self.output_block(IR_history)
        if return_attention:
            return logits, alpha_disciplines, alpha_semesters
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def parameter_df(self) -> pd.DataFrame:
        components = {
            "FeatureEncoder: Estudantes": self.student_encoder,
            "FeatureEncoder: Disciplinas": self.discipline_encoder,
            "FeatureEncoder: Institucional": self.institutional_encoder,
            "RegularBlock: Estudantes": self.student_block,
            "RegularBlock: Disciplinas": self.discipline_block,
            "RegularBlock: Institucional": self.institutional_block,
            "AttentionPooling: Disciplinas": self.discipline_pooling,
            "RegularBlock: Semestre": self.semester_block,
            "RecencyEncoding": self.recency_encoding,
            "AttentionPooling: Semestres": self.semester_pooling,
            "OutputBlock": self.output_block,
        }
        out_dims = {
            "FeatureEncoder: Estudantes": self.student_encoder.output_dim,
            "FeatureEncoder: Disciplinas": self.discipline_encoder.output_dim,
            "FeatureEncoder: Institucional": self.institutional_encoder.output_dim,
            "RegularBlock: Estudantes": self.student_block.output_dim,
            "RegularBlock: Disciplinas": self.discipline_block.output_dim,
            "RegularBlock: Institucional": self.institutional_block.output_dim,
            "AttentionPooling: Disciplinas": self.discipline_pooling.output_dim,
            "RegularBlock: Semestre": self.semester_block.output_dim,
            "AttentionPooling: Semestres": self.semester_pooling.output_dim,
        }

        return pd.DataFrame(
            {
                "componente": list(components.keys()),
                "total_params": [
                    sum(p.numel() for p in m.parameters()) for m in components.values()
                ],
                "out_dim": [out_dims.get(k) for k in components],
            }
        )
