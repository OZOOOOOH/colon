#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash bash/schedule.sh

#python run.py trainer.max_epochs=5
#
#python run.py trainer.max_epochs=10 logger=csv

python /home/quiil/PycharmProjects/colon/run_colon.py trainer.max_epochs=20 datamodule.img_size=224
python /home/quiil/PycharmProjects/colon/run_colon.py trainer.max_epochs=20 datamodule.img_size=224 model.name='vit_small_patch16_384'
python /home/quiil/PycharmProjects/colon/run_colon.py trainer.max_epochs=20 datamodule.img_size=384 model.name='vit_base_patch16_384'
python /home/quiil/PycharmProjects/colon/run_colon.py trainer.max_epochs=20 datamodule.img_size=384 model.name='vit_base_patch32_384'
python /home/quiil/PycharmProjects/colon/run_colon.py trainer.max_epochs=20 model.name='efficientnet_b0' datamodule.batch_size=32
python /home/quiil/PycharmProjects/colon/run_colon.py trainer.max_epochs=20 model.name='resnet50' datamodule.batch_size=32
