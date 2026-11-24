import math
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

from Model.EvasaoDataset import EvasaoDataset
from Model.Metrics import Metrics
from Model.Calibrator import Calibrator
from utils import SEED, montar_caminho_arquivo, set_seed


def move_batch_to_device(batch, device):
    def move(x):
        if isinstance(x, dict):
            return {k: move(v) for k, v in x.items()}
        return x.to(device)

    return {k: move(v) for k, v in batch.items()}


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        name: str,
        parent_name: str,
        collate: partial,
        columns: dict,
        criterion,
        device,
        threshold: float = 0.75,
        do_wandb: bool = True,
        calibracao: str = "platt",  # "platt" | "isotonica" | "nenhuma"
    ):
        self.model = model.to(device)
        self.name = name
        self.parent_name = parent_name
        self.collate = collate
        self.columns = columns
        self.criterion = criterion
        self.device = device
        self.threshold = threshold
        self.do_wandb = do_wandb
        self.calibracao = calibracao
        self.calibrador = None

    def model_path(self) -> str:
        return f"{montar_caminho_arquivo(Path('course_models') / self.parent_name, self.name)}.pt"

    # ------------------------------------------------------------------ #
    # Dados
    # ------------------------------------------------------------------ #
    def _make_dataset(self, df_e, df_i, df_d, train=True):
        # só estudantes que possuem ao menos uma disciplina
        df_e = df_e[df_e["matricula"].isin(df_d["matricula_estudante"])]
        c = self.columns
        return EvasaoDataset(
            df_e,
            df_i,
            df_d,
            c["STUDENT_CAT"],
            c["STUDENT_NUM"],
            c["INSTITUTIONAL_CAT"],
            c["INSTITUTIONAL_NUM"],
            c["DISCIPLINE_CAT"],
            c["DISCIPLINE_NUM"],
            train=train,
        )

    def _make_loaders(
        self, df_i, df_train_e, df_train_d, df_val_e, df_val_d, batch_size
    ):
        train_ds = self._make_dataset(df_train_e, df_i, df_train_d)
        val_ds = self._make_dataset(df_val_e, df_i, df_val_d)

        generator = torch.Generator()
        generator.manual_seed(SEED)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate,
            generator=generator,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, collate_fn=self.collate
        )
        return train_loader, val_loader

    # ------------------------------------------------------------------ #
    # Uma época
    # ------------------------------------------------------------------ #
    def run_epoch(self, epoch, loader, optimizer=None, grad_clip=None):
        is_train = optimizer is not None
        self.model.train(is_train)

        total_loss, all_logits, all_labels = 0.0, [], []

        with torch.set_grad_enabled(is_train):
            for batch in loader:
                batch = move_batch_to_device(batch, self.device)

                logits = self.model(
                    batch["student_cat"],
                    batch["student_num"],
                    batch["institutional_cat"],
                    batch["institutional_num"],
                    batch["discipline_cat"],
                    batch["discipline_num"],
                    batch["discipline_mask"],
                    batch["semester_mask"],
                ).squeeze(-1)

                loss = self.criterion(logits, batch["label"])

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    if grad_clip:
                        nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    optimizer.step()

                total_loss += loss.item() * batch["label"].size(0)
                all_logits.append(logits.detach().cpu())
                all_labels.append(batch["label"].detach().cpu())

        avg_loss = total_loss / len(loader.dataset)
        metrics = Metrics.from_logits(
            label="Train" if is_train else "Val",
            epoch=epoch,
            loss=avg_loss,
            logits=torch.cat(all_logits),
            labels=torch.cat(all_labels),
            threshold=self.threshold,
        )

        if self.do_wandb:
            metrics.wandb_log(f"{self.parent_name}/{self.name}")
            metrics.wandb_log(f"{self.parent_name}")

        return metrics

    # ------------------------------------------------------------------ #
    # Treino completo
    # ------------------------------------------------------------------ #
    def fit(
        self,
        df_i,
        df_train_e,
        df_train_d,
        df_val_e,
        df_val_d,
        batch_size,
        epochs,
        patience,
        lr,
        weight_decay,
        grad_clip=None,
        callback=None,
        criterio_parada: str = "loss",  # loss | auc
    ):
        set_seed()

        train_loader, val_loader = self._make_loaders(
            df_i, df_train_e, df_train_d, df_val_e, df_val_d, batch_size
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        best_score, best_val_auc = -float("inf"), float("nan")
        best_state, epochs_sem_melhora = None, 0

        for epoch in range(1, epochs + 1):
            train_m = self.run_epoch(epoch, train_loader, optimizer, grad_clip)
            val_m = self.run_epoch(epoch, val_loader)

            print(
                f"[{epoch}/{epochs}] "
                f"train_loss={train_m.loss:.4f} "
                f"val_loss={val_m.loss:.4f} "
                f"val_auc={val_m.auc:.4f} "
                f"val_f1={val_m.f1:.4f} | "
                f"val_f1_1={val_m.f1_1:.4f} "
                f"val_recall_1={val_m.recall_1:.4f}"
            )

            if callback is not None:
                callback(epoch, train_m, val_m)

            # NaN > x é sempre False: métrica indefinida conta como "sem melhora"
            score = val_m.auc if criterio_parada == "auc" else -val_m.loss
            if score > best_score:
                best_score, best_val_auc = score, val_m.auc
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                epochs_sem_melhora = 0
            else:
                epochs_sem_melhora += 1
                if epochs_sem_melhora >= patience:
                    print(
                        f"Early stopping na época {epoch} "
                        f"(sem melhora há {epochs_sem_melhora} épocas)"
                    )
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.save(best_state)
            self._calibrar(val_loader)

        return best_val_auc if math.isfinite(best_val_auc) else float("nan")

    # ------------------------------------------------------------------ #
    # Persistência
    # ------------------------------------------------------------------ #
    def save(self, state=None):
        state = state if state is not None else self.model.state_dict()
        path = self.model_path()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)
        if self.do_wandb:
            wandb.save(path)
        return path

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        return self.model

    @torch.no_grad()
    def predict(self, loader):
        """Logits, labels (CPU) e loss médio sobre um loader, em modo eval."""
        self.model.eval()
        total_loss, all_logits, all_labels = 0.0, [], []
        for batch in loader:
            batch = move_batch_to_device(batch, self.device)
            logits = self.model(
                batch["student_cat"],
                batch["student_num"],
                batch["institutional_cat"],
                batch["institutional_num"],
                batch["discipline_cat"],
                batch["discipline_num"],
                batch["discipline_mask"],
                batch["semester_mask"],
            ).squeeze(-1)
            loss = self.criterion(logits, batch["label"])
            total_loss += loss.item() * batch["label"].size(0)
            all_logits.append(logits.cpu())
            all_labels.append(batch["label"].cpu())
        return (
            torch.cat(all_logits),
            torch.cat(all_labels),
            total_loss / len(loader.dataset),
        )

    def calibrador_path(self) -> str:
        return str(Path(self.model_path()).with_name(f"{self.name}_calibracao.json"))

    def load_calibrador(self):
        caminho = Path(self.calibrador_path())
        return Calibrator.load(caminho) if caminho.exists() else None

    def _remover_calibrador(self):
        Path(self.calibrador_path()).unlink(missing_ok=True)
        self.calibrador = None

    def _calibrar(self, val_loader):
        """Ajusta o calibrador nos logits de validação do melhor checkpoint e o salva."""
        if self.calibracao == "nenhuma":
            return
        logits, labels, _ = self.predict(val_loader)
        try:
            self.calibrador = Calibrator(self.calibracao).fit(
                logits.numpy(), labels.numpy()
            )
        except ValueError as e:  # uma só classe na validação
            print(f"Calibração ignorada: {e}")
            return
        self.calibrador.save(self.calibrador_path())
