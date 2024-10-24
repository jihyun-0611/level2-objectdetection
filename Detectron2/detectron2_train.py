import os
import copy
import torch
import detectron2
from detectron2.data import detection_utils as utils
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader, build_detection_train_loader

from detectron2.engine import HookBase
import mlflow

from detectron2.engine import HookBase
import wandb

class WandbHook(HookBase):
    """
    Custom hook class to use wandb for logging model artifacts, metrics, and parameters
    """
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg.clone()
    
    def before_train(self):
        wandb.init(
            project=self.cfg.WANDB.PROJECT, 
            entity='buan99-personal', 
            name=self.cfg.WANDB.RUN_NAME, 
            tags=self.cfg.WANDB.TAGS,
            notes=self.cfg.WANDB.NOTES
            )
        
    def after_step(self):
        metrics = {}
        for k, v in self.trainer.storage.latest().items():
            metrics[k] = v[0]  # Latest value
        
        wandb.log(metrics, step=self.trainer.iter)
    
    def after_train(self):
        wandb.save(os.path.join(self.cfg.OUTPUT_DIR, "*"))
        wandb.finish()

class MLflowHook(HookBase):
    """
    custom hook class를 사용하여 mlflow에 model artifacts와 metrics, parameters 추가 
    """
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg.clone()
    
    # tracking server URI, experiment 이름, run 이름을 설정하고 실행을 시작
    def before_train(self):
        with torch.no_grad():
            mlflow.set_tracking_uri(self.cfg.MLFLOW.TRACKING_URI)
            mlflow.set_experiment(self.cfg.MLFLOW.EXPERIMENT_NAME)
            mlflow.start_run(run_name=self.cfg.MLFLOW.RUN_NAME)
        for k, v in self.cfg.items():
                mlflow.log_param(k, v) 
                
    # iteration step마다 Detectron2의 EventStorage에서 latest training metrics을 요청
    def after_step(self):
        with torch.no_grad():
            latest_metrics = self.trainer.storage.latest()
            for k, v in latest_metrics.items():
                mlflow.log_metric(key=k, value=v[0], step=v[1])
    
    # Detectron2 구성을 YAML 파일에 덤프하고 마지막으로 모든 출력 파일(config YAML 포함)을 MLflow에 로깅      
    def after_train(self):
        with torch.no_grad():
            with open(os.path.join(self.cfg.OUTPUT_DIR, "model-config.yaml"), "w") as f:
                f.write(self.cfg.dump())
            mlflow.log_artifacts(self.cfg.OUTPUT_DIR)

# mapper - input data를 어떤 형식으로 return할지 (따라서 augmnentation 등 데이터 전처리 포함 됨)
import detectron2.data.transforms as T

def MyMapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')
    
    transform_list = [
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3)
    ]
    
    image, transforms = T.apply_transform_gens(transform_list, image)
    
    dataset_dict['image'] = torch.as_tensor(image.transpose(2,0,1).astype('float32'))
    
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop('annotations')
        if obj.get('iscrowd', 0) == 0
    ]
    
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict['instances'] = utils.filter_empty_instances(instances)
    
    return dataset_dict

# trainer - DefaultTrainer를 상속
class MyTrainer(DefaultTrainer):
    
    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        return build_detection_train_loader(
        cfg, mapper = MyMapper, sampler = sampler
        )
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs('./output_eval', exist_ok = True)
            output_folder = './output_eval'
            
        return COCOEvaluator(dataset_name, cfg, False, output_folder)
    
# Register Dataset
try:
    register_coco_instances('coco_trash_train', {}, '/data/ephemeral/home/workspace/dataset/train.json', '/data/ephemeral/home/workspace/dataset/')
except AssertionError:
    pass

try:
    register_coco_instances('coco_trash_test', {}, '/data/ephemeral/home/workspace/dataset/test.json', '/data/ephemeral/home/workspace/dataset/')
except AssertionError:
    pass

MetadataCatalog.get('coco_trash_train').thing_classes = ["General trash", "Paper", "Paper pack", "Metal", 
                                                         "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

# config 불러오기
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'))

# config 수정하기
cfg.DATASETS.TRAIN = ('coco_trash_train',)
cfg.DATASETS.TEST = ('coco_trash_test',)

cfg.DATALOADER.NUM_WOREKRS = 2

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml')

cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 15000
cfg.SOLVER.STEPS = (8000,12000)
cfg.SOLVER.GAMMA = 0.005
cfg.SOLVER.CHECKPOINT_PERIOD = 3000

cfg.SOLVER.AMP.ENABLED = True # Mixed precision training

cfg.OUTPUT_DIR = './output'

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10

cfg.TEST.EVAL_PERIOD = 3000

# mlflow config 추가 
cfg.MLFLOW = CfgNode()
cfg.MLFLOW.EXPERIMENT_NAME = "detectron2_faster_rcnn"
cfg.MLFLOW.RUN_NAME = "#0_detectron2_baseline_training"
cfg.MLFLOW.TRACKING_URI = "http://10.28.224.171:30280"

# wandb config 추가 
# run name : 실험할 때마다 모델명으로 바꿔주시면 됩니다. 
# tags : 실험 관련한 키워드를 써주시면 돼요
# notes : 실험에 대한 간략한 설명을 써주시면 됩니다. 
# 그외 project와 entity는 수정하시면 안됩니다. 
cfg.WANDB = CfgNode()
cfg.WANDB.PROJECT = "detectron2-OD"
cfg.WANDB.RUN_NAME = "faster_rcnn_R_101_FPN_3x"
cfg.WANDB.TAGS = ['baseline', 'faster_rcnn']
cfg.WANDB.NOTES = 'First experiments with wandb'

# train
os.makedirs(cfg.OUTPUT_DIR, exist_ok = True)

mlflow_hook = MLflowHook(cfg)
wandb_hook = WandbHook(cfg)

trainer = MyTrainer(cfg)
trainer.register_hooks(hooks=[wandb_hook]) # mlflow hook 대신 wandb hook 등록 
trainer.resume_or_load(resume=False)
trainer.train()