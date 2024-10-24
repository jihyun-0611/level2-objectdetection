from mmengine.config import Config
from mmengine.runner import Runner

classes = {"classes": 
           ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")}

# config file 들고오기
cfg = Config.fromfile('/data/ephemeral/home/mmdetection/projects/EfficientDet/configs/efficientdet_effb0_bifpn_8xb16-crop512-300e_coco.py')

root = '/data/ephemeral/home/dataset/'

# dataset config 수정
cfg.train_dataloader.dataset.data_prefix = dict(img='')
cfg.train_dataloader.dataset.metainfo = classes # mmengine/dataset/base_dataset에 정의되어 있음.
cfg.train_dataloader.dataset.ann_file = root +'train_split_random.json' 
cfg.train_dataloader.dataset.pipeline[2]['scale'] = (512,512) # Resize

cfg.val_dataloader.dataset.data_prefix = dict(img='')
cfg.val_dataloader.dataset.metainfo = classes
cfg.val_dataloader.dataset.ann_file = root + 'val_split_random.json' 
cfg.val_dataloader.dataset.pipeline[1]['scale'] = (512,512) # Resize

cfg.val_evaluator.ann_file = root + 'val_split_random.json' 

# cfg.data.samples_per_gpu = 4 배치 사이즈는 train_dataloader에 정의됨.
cfg.train_dataloader.batch_size = 16
# cfg.seed = 2022
# cfg.gpu_ids = [0]
# cfg.work_dir = './work_dirs/faster_rcnn_r50_fpn_1x_trash'
cfg.model.bbox_head.num_classes = 10
cfg.work_dir = '/data/ephemeral/home/mmdetection/EfficientDet'


# wandb 설정
# wandb init arguments : 필요 없으면 모두 None으로 해도됨 
run_name = 'efficientdet'
tags = ['50epoch'] # 원하는 태그 설정
notes = 'efficientdet' # 해당 run에 대한 설명

# visualizer 설정 
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={'project': 'MMDetection-OD', 
                            'entity': 'buan99-personal', 
                            'name': run_name, 
                            'tags': tags, 
                            'notes': notes},
         )
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)

# cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
# cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
# cfg.device = get_device()

# mmdetection v2와 다르게 config 설정의 변경을 runner가 받음.
# 훈련 시작
runner = Runner.from_cfg(cfg)
runner.train()