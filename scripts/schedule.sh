#python run_colon.py trainer.max_epochs=20 datamodule.img_size=224
#python run_colon.py trainer.max_epochs=20 datamodule.img_size=224 model.name='vit_small_patch16_384'
#python run_colon.py trainer.max_epochs=20 datamodule.img_size=384 model.name='vit_base_patch16_384'
#python run_colon.py trainer.max_epochs=20 datamodule.img_size=384 model.name='vit_base_patch32_384'
#python run_colon.py trainer.max_epochs=20 model.name='efficientnet_b0' datamodule.batch_size=32
#python run_colon.py trainer.max_epochs=20 model.name='resnet50' datamodule.batch_size=32

#python run_colon.py -m model.lr=0.001,0.0001 trainer.max_epochs=5 model.name='efficientnet_b0' datamodule.batch_size=32,64


#python run_colon.py model.lr=0.0001 datamodule.batch_size=16 model.name='vit_base_patch16_384'
#python run_colon.py model.lr=0.0005 datamodule.batch_size=16 model.name='vit_base_patch16_384'
#python run_colon.py model.lr=0.00001 datamodule.batch_size=16 model.name='vit_base_patch16_384'
#
#
#python run_colon.py model.lr=0.0001 datamodule.batch_size=16 model.name='vit_base_patch32_384'
#python run_colon.py model.lr=0.0005 datamodule.batch_size=16 model.name='vit_base_patch32_384'
#python run_colon.py model.lr=0.00001 datamodule.batch_size=16 model.name='vit_base_patch32_384'
#
#
#python run_colon.py model.lr=0.0001 datamodule.batch_size=16 model.name='vit_base_r50_s16_384'
#python run_colon.py model.lr=0.0005 datamodule.batch_size=16 model.name='vit_base_r50_s16_384'
#python run_colon.py model.lr=0.00001 datamodule.batch_size=16 model.name='vit_base_r50_s16_384'
#
#python run_colon.py model.lr=0.0001 datamodule.batch_size=16 model.name='vit_large_patch16_384'
#python run_colon.py model.lr=0.0005 datamodule.batch_size=16 model.name='vit_large_patch16_384'
#python run_colon.py model.lr=0.00001 datamodule.batch_size=16 model.name='vit_large_patch16_384'
#
#python run_colon.py model.lr=0.0001 datamodule.batch_size=16 model.name='vit_large_patch32_384'
#python run_colon.py model.lr=0.0005 datamodule.batch_size=16 model.name='vit_large_patch32_384'
#python run_colon.py model.lr=0.00001 datamodule.batch_size=16 model.name='vit_large_patch32_384'
#
#python run_colon.py model.lr=0.0001 datamodule.batch_size=16 model.name='vit_large_r50_s32_384'
#python run_colon.py model.lr=0.0005 datamodule.batch_size=16 model.name='vit_large_r50_s32_384'
#python run_colon.py model.lr=0.00001 datamodule.batch_size=16 model.name='vit_large_r50_s32_384'
#
#python run_colon.py model.lr=0.0001 datamodule.batch_size=16 model.name='vit_small_patch16_384'
#python run_colon.py model.lr=0.0005 datamodule.batch_size=16 model.name='vit_small_patch16_384'
#python run_colon.py model.lr=0.00001 datamodule.batch_size=16 model.name='vit_small_patch16_384'
#
#python run_colon.py model.lr=0.0001 datamodule.batch_size=16 model.name='vit_small_patch32_384'
#python run_colon.py model.lr=0.0005 datamodule.batch_size=16 model.name='vit_small_patch32_384'
#python run_colon.py model.lr=0.00001 datamodule.batch_size=16 model.name='vit_small_patch32_384'
#
#python run_colon.py model.lr=0.0001 datamodule.batch_size=16 model.name='vit_small_r26_s32_384'
#python run_colon.py model.lr=0.0005 datamodule.batch_size=16 model.name='vit_small_r26_s32_384'
#python run_colon.py model.lr=0.00001 datamodule.batch_size=16 model.name='vit_small_r26_s32_384'
#
#python run_colon.py model.lr=0.0001 datamodule.batch_size=16 model.name='vit_tiny_patch16_384'
#python run_colon.py model.lr=0.0005 datamodule.batch_size=16 model.name='vit_tiny_patch16_384'
#python run_colon.py model.lr=0.00001 datamodule.batch_size=16 model.name='vit_tiny_patch16_384'
#
#python run_colon.py model.lr=0.0001 datamodule.batch_size=16 model.name='vit_tiny_r_s16_p8_384'
#python run_colon.py model.lr=0.0005 datamodule.batch_size=16 model.name='vit_tiny_r_s16_p8_384'
#python run_colon.py model.lr=0.00001 datamodule.batch_size=16 model.name='vit_tiny_r_s16_p8_384'

#['vit_base_patch16_384',
# 'vit_base_patch32_384',
# 'vit_base_r50_s16_384',
# 'vit_large_patch16_384',
# 'vit_large_patch32_384',
# 'vit_large_r50_s32_384',
# 'vit_small_patch16_384',
# 'vit_small_patch32_384',
# 'vit_small_r26_s32_384',
# 'vit_tiny_patch16_384',
# 'vit_tiny_r_s16_p8_384']

#python run_colon.py model.lr=0.0001 datamodule.batch_size=32 model.name='efficientnet_b0'
#python run_colon.py model.lr=0.00001 datamodule.batch_size=32 model.name='efficientnet_b0'
#python run_colon.py model.lr=0.00005 datamodule.batch_size=32 model.name='efficientnet_b0'
#python run_colon.py model.lr=0.000001 datamodule.batch_size=32 model.name='efficientnet_b0'

#python run_colon.py model.lr=1e-3 datamodule.batch_size=16 model.name='vit_base_patch16_384' model.scheduler='CosineAnnealingLR'
#python run_colon.py model.lr=1e-4 datamodule.batch_size=16 model.name='vit_base_patch16_384' model.scheduler='CosineAnnealingLR'
#python run_colon.py model.lr=1e-5 datamodule.batch_size=16 model.name='vit_base_patch16_384' model.scheduler='CosineAnnealingLR'
#python run_colon.py model.lr=1e-3 datamodule.batch_size=16 model.name='vit_base_patch16_384' model.scheduler='CosineAnnealingWarmRestarts'
#python run_colon.py model.lr=1e-4 datamodule.batch_size=16 model.name='vit_base_patch16_384' model.scheduler='CosineAnnealingWarmRestarts'
#python run_colon.py model.lr=1e-5 datamodule.batch_size=16 model.name='vit_base_patch16_384' model.scheduler='CosineAnnealingWarmRestarts'
#python run_colon.py model.lr=1e-3 datamodule.batch_size=16 model.name='vit_base_patch16_384' model.scheduler='ReduceLROnPlateau'
#python run_colon.py model.lr=1e-4 datamodule.batch_size=16 model.name='vit_base_patch16_384' model.scheduler='ReduceLROnPlateau'
#python run_colon.py model.lr=1e-5 datamodule.batch_size=16 model.name='vit_base_patch16_384' model.scheduler='ReduceLROnPlateau'

#python run_colon.py model.lr=1e-3 datamodule.batch_size=16 model.name='vit_base_patch32_384' model.scheduler='CosineAnnealingLR'
#python run_colon.py model.lr=1e-4 datamodule.batch_size=16 model.name='vit_base_patch32_384' model.scheduler='CosineAnnealingLR'
#python run_colon.py model.lr=1e-5 datamodule.batch_size=16 model.name='vit_base_patch32_384' model.scheduler='CosineAnnealingLR'
#python run_colon.py model.lr=1e-3 datamodule.batch_size=16 model.name='vit_base_patch32_384' model.scheduler='CosineAnnealingWarmRestarts'
#python run_colon.py model.lr=1e-4 datamodule.batch_size=16 model.name='vit_base_patch32_384' model.scheduler='CosineAnnealingWarmRestarts'
#python run_colon.py model.lr=1e-5 datamodule.batch_size=16 model.name='vit_base_patch32_384' model.scheduler='CosineAnnealingWarmRestarts'
#python run_colon.py model.lr=1e-3 datamodule.batch_size=16 model.name='vit_base_patch32_384' model.scheduler='ReduceLROnPlateau'
#python run_colon.py model.lr=1e-4 datamodule.batch_size=16 model.name='vit_base_patch32_384' model.scheduler='ReduceLROnPlateau'
#python run_colon.py model.lr=1e-5 datamodule.batch_size=16 model.name='vit_base_patch32_384' model.scheduler='ReduceLROnPlateau'
#여기부터


#python run_colon.py model.lr=1e-3 datamodule.batch_size=16 model.name='vit_base_r50_s16_384' model기.scheduler='CosineAnnealingLR'
#python run_colon.py model.lr=1e-4 datamodule.batch_size=16 model.name='vit_base_r50_s16_384' model.scheduler='CosineAnnealingLR'
#python run_colon.py model.lr=1e-5 datamodule.batch_size=16 model.name='vit_base_r50_s16_384' model.scheduler='CosineAnnealingLR'
#python run_colon.py model.lr=1e-3 datamodule.batch_size=16 model.name='vit_base_r50_s16_384' model.scheduler='CosineAnnealingWarmRestarts'
#python run_colon.py model.lr=1e-4 datamodule.batch_size=16 model.name='vit_base_r50_s16_384' model.scheduler='CosineAnnealingWarmRestarts'
#python run_colon.py model.lr=1e-5 datamodule.batch_size=16 model.name='vit_base_r50_s16_384' model.scheduler='CosineAnnealingWarmRestarts'
#python run_colon.py model.lr=1e-3 datamodule.batch_size=16 model.name='vit_base_r50_s16_384' model.scheduler='ReduceLROnPlateau'
#python run_colon.py model.lr=1e-4 datamodule.batch_size=16 model.name='vit_base_r50_s16_384' model.scheduler='ReduceLROnPlateau'
#python run_colon.py model.lr=1e-5 datamodule.batch_size=16 model.name='vit_base_r50_s16_384' model.scheduler='ReduceLROnPlateau'

#python run_colon.py model.lr=1e-3 datamodule.batch_size=16 model.name='vit_small_patch16_384' model.scheduler='CosineAnnealingLR'
#python run_colon.py model.lr=1e-4 datamodule.batch_size=16 model.name='vit_small_patch16_384' model.scheduler='CosineAnnealingLR'
#python run_colon.py model.lr=1e-5 datamodule.batch_size=16 model.name='vit_small_patch16_384' model.scheduler='CosineAnnealingLR'
#python run_colon.py model.lr=1e-3 datamodule.batch_size=16 model.name='vit_small_patch16_384' model.scheduler='CosineAnnealingWarmRestarts'
#python run_colon.py model.lr=1e-4 datamodule.batch_size=16 model.name='vit_small_patch16_384' model.scheduler='CosineAnnealingWarmRestarts'
#python run_colon.py model.lr=1e-5 datamodule.batch_size=16 model.name='vit_small_patch16_384' model.scheduler='CosineAnnealingWarmRestarts'
#python run_colon.py model.lr=1e-3 datamodule.batch_size=16 model.name='vit_small_patch16_384' model.scheduler='ReduceLROnPlateau'
#python run_colon.py model.lr=1e-4 datamodule.batch_size=16 model.name='vit_small_patch16_384' model.scheduler='ReduceLROnPlateau'
#python run_colon.py model.lr=1e-5 datamodule.batch_size=16 model.name='vit_small_patch16_384' model.scheduler='ReduceLROnPlateau'

#python run_colon.py model.lr=1e-3 datamodule.batch_size=4 model.name='vit_large_patch16_384' model.scheduler='CosineAnnealingLR'
#python run_colon.py model.lr=1e-4 datamodule.batch_size=4 model.name='vit_large_patch16_384' model.scheduler='CosineAnnealingLR'
#python run_colon.py model.lr=1e-5 datamodule.batch_size=4 model.name='vit_large_patch16_384' model.scheduler='CosineAnnealingLR'
#python run_colon.py model.lr=1e-3 datamodule.batch_size=4 model.name='vit_large_patch16_384' model.scheduler='CosineAnnealingWarmRestarts'
#python run_colon.py model.lr=1e-4 datamodule.batch_size=4 model.name='vit_large_patch16_384' model.scheduler='CosineAnnealingWarmRestarts'
#python run_colon.py model.lr=1e-5 datamodule.batch_size=4 model.name='vit_large_patch16_384' model.scheduler='CosineAnnealingWarmRestarts'
#python run_colon.py model.lr=1e-3 datamodule.batch_size=4 model.name='vit_large_patch16_384' model.scheduler='ReduceLROnPlateau'
#python run_colon.py model.lr=1e-4 datamodule.batch_size=4 model.name='vit_large_patch16_384' model.scheduler='ReduceLROnPlateau'
#python run_colon.py model.lr=1e-5 datamodule.batch_size=4 model.name='vit_large_patch16_384' model.scheduler='ReduceLROnPlateau'
#
#python run_colon.py model.lr=1e-3 datamodule.batch_size=4 model.name='vit_large_patch32_384' model.scheduler='CosineAnnealingLR'
#python run_colon.py model.lr=1e-4 datamodule.batch_size=4 model.name='vit_large_patch32_384' model.scheduler='CosineAnnealingLR'
#python run_colon.py model.lr=1e-5 datamodule.batch_size=4 model.name='vit_large_patch32_384' model.scheduler='CosineAnnealingLR'
#python run_colon.py model.lr=1e-3 datamodule.batch_size=4 model.name='vit_large_patch32_384' model.scheduler='CosineAnnealingWarmRestarts'
#python run_colon.py model.lr=1e-4 datamodule.batch_size=4 model.name='vit_large_patch32_384' model.scheduler='CosineAnnealingWarmRestarts'
#python run_colon.py model.lr=1e-5 datamodule.batch_size=4 model.name='vit_large_patch32_384' model.scheduler='CosineAnnealingWarmRestarts'
#python run_colon.py model.lr=1e-3 datamodule.batch_size=4 model.name='vit_large_patch32_384' model.scheduler='ReduceLROnPlateau'
#python run_colon.py model.lr=1e-4 datamodule.batch_size=4 model.name='vit_large_patch32_384' model.scheduler='ReduceLROnPlateau'
#python run_colon.py model.lr=1e-5 datamodule.batch_size=4 model.name='vit_large_patch32_384' model.scheduler='ReduceLROnPlateau'
#
#python run_colon.py model.lr=1e-3 datamodule.batch_size=4 model.name='vit_large_r50_s32_384' model.scheduler='CosineAnnealingLR'
#python run_colon.py model.lr=1e-4 datamodule.batch_size=4 model.name='vit_large_r50_s32_384' model.scheduler='CosineAnnealingLR'
#python run_colon.py model.lr=1e-5 datamodule.batch_size=4 model.name='vit_large_r50_s32_384' model.scheduler='CosineAnnealingLR'
#python run_colon.py model.lr=1e-3 datamodule.batch_size=4 model.name='vit_large_r50_s32_384' model.scheduler='CosineAnnealingWarmRestarts'
#python run_colon.py model.lr=1e-4 datamodule.batch_size=4 model.name='vit_large_r50_s32_384' model.scheduler='CosineAnnealingWarmRestarts'
python run_colon.py model.lr=1e-5 datamodule.batch_size=4 model.name='vit_large_r50_s32_384' model.scheduler='CosineAnnealingWarmRestarts'
python run_colon.py model.lr=1e-3 datamodule.batch_size=4 model.name='vit_large_r50_s32_384' model.scheduler='ReduceLROnPlateau'
python run_colon.py model.lr=1e-4 datamodule.batch_size=4 model.name='vit_large_r50_s32_384' model.scheduler='ReduceLROnPlateau'
python run_colon.py model.lr=1e-5 datamodule.batch_size=4 model.name='vit_large_r50_s32_384' model.scheduler='ReduceLROnPlateau'



python run_colon.py model.lr=1e-3 datamodule.batch_size=16 model.name='efficientnet_b4' model.scheduler='CosineAnnealingLR'
python run_colon.py model.lr=1e-4 datamodule.batch_size=16 model.name='efficientnet_b4' model.scheduler='CosineAnnealingLR'
python run_colon.py model.lr=1e-5 datamodule.batch_size=16 model.name='efficientnet_b4' model.scheduler='CosineAnnealingLR'
python run_colon.py model.lr=1e-3 datamodule.batch_size=16 model.name='efficientnet_b4' model.scheduler='CosineAnnealingWarmRestarts'
python run_colon.py model.lr=1e-4 datamodule.batch_size=16 model.name='efficientnet_b4' model.scheduler='CosineAnnealingWarmRestarts'
python run_colon.py model.lr=1e-5 datamodule.batch_size=16 model.name='efficientnet_b4' model.scheduler='CosineAnnealingWarmRestarts'
python run_colon.py model.lr=1e-3 datamodule.batch_size=16 model.name='efficientnet_b4' model.scheduler='ReduceLROnPlateau'
python run_colon.py model.lr=1e-4 datamodule.batch_size=16 model.name='efficientnet_b4' model.scheduler='ReduceLROnPlateau'
python run_colon.py model.lr=1e-5 datamodule.batch_size=16 model.name='efficientnet_b4' model.scheduler='ReduceLROnPlateau'


python run_colon.py model.lr=1e-5 datamodule.batch_size=32 model.name='resnet50' model.scheduler='ReduceLROnPlateau'
