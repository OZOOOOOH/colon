from typing import Any, List
import torch
import timm
from pytorch_lightning import plugins
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from src.models.components.vit import ViT
import random


class ColonLitModule(LightningModule):
    def __init__(
            self,
            lr: float = 1e-4,
            weight_decay: float = 0.0005,
            t_max: int = 20,
            min_lr: int = 1e-6,
            T_0=15,
            T_mult=2,
            eta_min=1e-6,
            name='vit_base_patch16_224',
            pretrained=True,
            scheduler='ReduceLROnPlateau',
            factor=0.5,
            patience=5,
            eps=1e-08,
            loss_weight=0.5

    ):
        super(ColonLitModule, self).__init__()
        self.save_hyperparameters(logger=False)

        self.model = timm.create_model(self.hparams.name, pretrained=self.hparams.pretrained, num_classes=4)

        # self.model = ViT(image_size=384, patch_size=16, num_classes=4, dim=768, depth=12, heads=12, mlp_dim=768 * 4,
        #                  pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.)
        self.compare_layer = torch.nn.Linear(self.model.embed_dim * 2, 3)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.train_acc_compare = Accuracy()
        self.val_acc_compare = Accuracy()
        self.test_acc_compare = Accuracy()

        self.val_acc_best = MaxMetric()
        self.val_acc_compare_best = MaxMetric()

    def forward(self, x):
        return self.model(x)

    def shuffle(self, x, y):

        z = [list(z) for z in zip(x, y)]
        z = list(enumerate(z))
        z = random.sample(z, len(z))
        indices, z = zip(*z)
        indices = list(indices)
        z = list(z)

        tmp1 = [i[0] for i in z]
        tmp2 = [i[1] for i in z]

        shuffle_x = torch.stack(tmp1, dim=0)
        shuffle_y = torch.stack(tmp2, dim=0)

        # origin < shuffle : 0
        # origin = shuffle : 1
        # origin > shuffle : 2
        comparison = []
        for i, j in zip(y.tolist(), shuffle_y.tolist()):

            if i > j:
                comparison.append(0)
            elif i == j:
                comparison.append(1)
            else:
                comparison.append(2)
        comparison = torch.tensor(comparison, device=self.device)
        return indices, comparison

    def step(self, batch):
        x, y = batch
        # logits = self.forward(x)
        features = self.model.forward_features(x.float())
        logits = self.model.head(features)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        indices, comparison = self.shuffle(x, y)
        shuffle_features = [features[i] for i in indices]
        shuffle_features = torch.stack(shuffle_features, dim=0)

        concat_features = torch.cat((features, shuffle_features), dim=1)

        logits_compare = self.compare_layer(concat_features)
        loss_compare = self.criterion(logits_compare, comparison)
        preds_compare = torch.argmax(logits_compare, dim=1)

        losses = loss + loss_compare * self.hparams.loss_weight

        return losses, preds, y, preds_compare, comparison

    def training_step(self, batch, batch_idx):
        loss, preds, targets, preds_compare, comparison = self.step(batch)
        acc = self.train_acc(preds=preds, target=targets)
        acc_compare = self.train_acc_compare(preds=preds_compare, target=comparison)
        # sch = self.lr_schedulers()
        # if isinstance(sch, torch.optim.lr_scheduler.CosineAnnealingLR):
        #     sch.step()
        # if isinstance(sch, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
        #     sch.step()

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc_compare", acc_compare, on_step=True, on_epoch=True, prog_bar=True)
        self.log("LearningRate", self.optimizer.param_groups[0]['lr'])

        # self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        logs = {
            "loss": loss,
            "acc": acc,
            "preds": preds,
            "targets": targets,
            "acc_compare": preds_compare,
            "preds_compare": preds_compare,
            "comparison": comparison
        }

        return logs

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val/loss"])

    def validation_step(self, batch, batch_idx):

        loss, preds, targets, preds_compare, comparison = self.step(batch)

        acc = self.val_acc(preds, targets)
        acc_compare = self.val_acc_compare(preds=preds_compare, target=comparison)

        # confusion_matrix = torch.zeros(3, 3)
        #
        # for t, p in zip(target.view(-1), output.argmax(1).view(-1)):
        #         confusion_matrix[t.long(), p.long()] += 1
        # print("confusion_matrix:\n",confusion_matrix)

        logs = {
            "loss": loss,
            "acc": acc,
            "acc_compare": acc_compare,

            # "confusion_matrix": confusion_matrix
        }
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc_compare", acc_compare, on_step=False, on_epoch=True, prog_bar=True)
        # self.log_dict(logs,
        #               on_step=False, on_epoch=True, prog_bar=True, logger=True
        #               )
        return {"loss": loss, "acc": acc, "preds": preds, "targets": targets, "acc_compare": preds_compare,
                "preds_compare": preds_compare, "comparison": comparison}

    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]
        acc = self.val_acc.compute()
        self.val_acc_best.update(acc)

        acc_compare = self.val_acc_compare.compute()
        self.val_acc_compare_best.update(acc_compare)
        # self.log('val_accuracy_best', self.val_acc_best.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)
        self.log("val/acc_compare_best", self.val_acc_compare_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, preds, targets, preds_compare, comparison = self.step(batch)
        acc = self.test_acc(preds, targets)
        acc_compare = self.test_acc_compare(preds_compare, comparison)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        self.log("test/acc_compare", acc_compare, on_step=False, on_epoch=True)
        # self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss, "acc": acc, "preds": preds, "targets": targets, "acc_compare": preds_compare,
                "preds_compare": preds_compare, "comparison": comparison}

    def test_epoch_end(self, outputs):
        pass

    def on_epoch_end(self):
        self.train_acc.reset()
        self.val_acc.reset()
        self.test_acc.reset()
        self.train_acc_compare.reset()
        self.val_acc_compare.reset()
        self.test_acc_compare.reset()

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        self.scheduler = self.get_scheduler()
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler, 'monitor': 'val/loss'}

        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def get_scheduler(self):
        if self.hparams.scheduler == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.hparams.factor,
                patience=self.hparams.patience,
                verbose=True,
                eps=self.hparams.eps)
        elif self.hparams.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.hparams.t_max,
                eta_min=self.hparams.min_lr,
                last_epoch=-1)
        elif self.hparams.scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.hparams.T_0,
                T_mult=1,
                eta_min=self.hparams.min_lr,
                last_epoch=-1)
        elif self.hparams.scheduler == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=200, gamma=0.1,
            )
        elif self.hparams.scheduler == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.95
            )
        # elif self.hparams.scheduler == 'MultiStepLR':
        #     scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #         )
        # elif self.hparams.scheduler == 'ConstantLR':
        #     scheduler = torch.optim.lr_scheduler.ConstantLR(
        #         )
        # elif self.hparams.scheduler == 'LinearLR':
        #     scheduler = torch.optim.lr_scheduler.LinearLR(
        #         )
        # elif self.hparams.scheduler == 'ChainedScheduler':
        #     scheduler = torch.optim.lr_scheduler.ChainedScheduler(
        #         )
        # elif self.hparams.scheduler == 'SequentialLR':
        #     scheduler = torch.optim.lr_scheduler.SequentialLR(
        #         )
        # elif self.hparams.scheduler == 'CyclicLR':
        #     scheduler = torch.optim.lr_scheduler.CyclicLR(
        #         self.optimizer, base_lr=1e-5, max_lr=1e-2, step_size_up=5,mode="exp_range", gamma=0.95
        #     )
        # elif self.hparams.scheduler == 'OneCycleLR':
        #     scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #         self.optimizer, max_lr=1e-2,
        #     )

        return scheduler
