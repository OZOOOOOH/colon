python run_colon.py trainer.max_epochs=20 datamodule.img_size=224
python run_colon.py trainer.max_epochs=20 datamodule.img_size=224 model.name='vit_small_patch16_384'
python run_colon.py trainer.max_epochs=20 datamodule.img_size=384 model.name='vit_base_patch16_384'
python run_colon.py trainer.max_epochs=20 datamodule.img_size=384 model.name='vit_base_patch32_384'
python run_colon.py trainer.max_epochs=20 model.name='efficientnet_b0' datamodule.batch_size=32
python run_colon.py trainer.max_epochs=20 model.name='resnet50' datamodule.batch_size=32
