# 모듈 import

from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# config file 들고오기
cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')

root='../../dataset/'

# dataset config 수정
cfg.data.train.classes = classes
cfg.data.train.img_prefix = root
cfg.data.train.ann_file = root + 'train_split_random.json' # train json 정보
cfg.data.train.pipeline[2]['img_scale'] = (512,512) # Resize

cfg.data.val.classes = classes
cfg.data.val.img_prefix = root
cfg.data.val.ann_file = root + 'val_split_random.json' # test json 정보
cfg.data.val.pipeline[1]['img_scale'] = (512,512) # Resize

cfg.data.test.classes = classes
cfg.data.test.img_prefix = root
cfg.data.test.ann_file = root + 'test.json' # test json 정보
cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize

cfg.data.samples_per_gpu = 4

cfg.seed = 2024
cfg.gpu_ids = [0]
cfg.work_dir = './work_dirs/faster_rcnn_r50_fpn_1x_trash'

cfg.model.roi_head.bbox_head.num_classes = 10

cfg.fp16 = dict(loss_scale=512.0) # Mixed Precision training

cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=6)
cfg.device = get_device()

# wandb init arguments : 필요 없으면 모두 None으로 해도됨 
run_name = 'faster_rcnn_r50_fpn_1x_coco'
tags = ['baseline', 'faster_rcnn'] # 원하는 태그 설정
notes = 'First experiments with wandb' # 해당 run에 대한 설명

# wandb hooks 추가 
cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='MMDetWandbHook',
         init_kwargs={'project': 'MMDetection-OD', 'entity': 'buan99-personal', 'name': run_name, 'tags': tags, 'notes': notes},
         interval=10,
         log_checkpoint=True,
         log_checkpoint_metadata=True,
         num_eval_images=0)]

# build_dataset
datasets = [build_dataset(cfg.data.train)]

# 모델 build 및 pretrained network 불러오기
model = build_detector(cfg.model)
model.init_weights()

meta = dict() # Mixed Precision training

# 모델 학습
train_detector(model, datasets[0], cfg, distributed=False, validate=True, meta=meta)