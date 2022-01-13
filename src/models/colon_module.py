from typing import Any, List
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy


class ColonLitModule(LightningModule):
    def __init__(
            self,
            model,
            lr: float = 0.001,
            weight_decay: float = 0.0005,
            t_max: int = 20,
            eta_min: int = 1e-6,
            # TODO figure out this part!
    ):
        super(ColonLitModule, self).__init__()
        self.save_hyperparameters(logger=False)
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.val_acc_best = MaxMetric()

        # self.lr = CONFIG["lr"]

    def forward(self, x):
        return self.model(x)

    def step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        acc = self.train_acc(preds=preds, target=targets)
        logs = {
            "train_loss": loss,
            "train_accuracy": acc,
            "preds": preds,
            "targets": targets
        }
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return logs

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch, batch_idx):
        print('\nWe are in validation_step!!!\n')

        loss, preds, targets = self.step(batch)

        acc = self.val_acc(preds, targets)

        # confusion_matrix = torch.zeros(3, 3)
        #
        # for t, p in zip(target.view(-1), output.argmax(1).view(-1)):
        #         confusion_matrix[t.long(), p.long()] += 1
        # print("confusion_matrix:\n",confusion_matrix)

        logs = {
            "valid_loss": loss,
            "valid_accracy": acc,
            # "confusion_matrix": confusion_matrix
        }

        self.log_dict(logs,
                      on_step=False, on_epoch=True, prog_bar=True, logger=True
                      )
        return {"valid_loss": loss, "valid_accracy": acc, "preds": preds, "target": targets}

    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]
        acc = self.val_acc.compute()
        self.val_acc_best.update(acc)
        self.log('val_accuracy_best', self.val_acc_best.compute(), on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        print('\nWe are in test_step!!!\n')
        loss, preds, targets = self.step(batch)
        acc = self.test_acc(preds, targets)

        logs = {
            "test_loss": loss,
            "test_accracy": acc,
            # "confusion_matrix": confusion_matrix
        }

        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"test_loss": loss, "test_accracy": acc, "preds": preds, "target": targets}

    def test_epoch_end(self, outputs):
        pass

    def on_epoch_end(self):
        self.train_acc.reset()
        self.val_acc.reset()
        self.test_acc.reset()

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=CONFIG["t_max"],
            eta_min=CONFIG["min_lr"]
        )

        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}
