from typing import List, Optional
import wandb
import os
import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
import torch
import torch.backends.cudnn as cudnn
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)
        cudnn.benchmark = False
        cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(config.seed)
        torch.use_deterministic_algorithms(True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))
    # config.logger
    if 'logger' in config.keys():
        config.logger.wandb.name = 'compare_' + config.model.name + '_imgsize_512' + '_lw_' + str(
            config.model.loss_weight) + '_scheduler_' + config.model.scheduler + '_lr_' + str(
            config.model.lr) + '_batchsize_' + str(config.datamodule.batch_size) + '_discriminator1+2'

    # if 'logger' in config.keys():
    #     config.logger.wandb.name = config.model.name + '_imgsize_512' + '_scheduler_' + config.model.scheduler + '_lr_' + \
    #                                str(config.model.lr) + '_batchsize_' + str(config.datamodule.batch_size)
    #     # setting wandb run name

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # log.info("Tune Learning Rate!")
    # datamodule.setup()
    # trainer.tune(model)
    # trainer.tune(model, datamodule.train_dataloader(), datamodule.val_dataloader())

    # lr_finder = trainer.tuner.lr_find(model, datamodule.train_dataloader(), datamodule.val_dataloader())
    # lr_finder = trainer.tuner.lr_find(model, datamodule)
    # new_lr = lr_finder.suggestion()
    # print(f'new_lr: {new_lr}')
    # model.hparams.lr = new_lr
    # new_batch_size = trainer.tuner.scale_batch_size(model, datamodule.train_dataloader(), datamodule.val_dataloader())
    # model.hparams.batch_size = new_batch_size
    # print(f'new_batch_size: {new_batch_size}')

    # Train the model

    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # Get metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric and optimized_metric not in trainer.callback_metrics:
        raise Exception(
            "Metric for hyperparameter optimization not found! "
            "Make sure the `optimized_metric` in `hparams_search` config is correct!"
        )
    score = trainer.callback_metrics.get(optimized_metric)

    # Test the model
    if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run"):
        log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    print(f'score: {score}')
    return score
