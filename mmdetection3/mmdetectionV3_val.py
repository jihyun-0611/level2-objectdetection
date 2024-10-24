################################################
## Streamit EDA를 위한 validation json 파일 추출 ##
###############################################
import os
import pandas as pd
from pycocotools.coco import COCO
from mmengine.config import Config
from mmengine.runner import Runner
import json
import torch

from mmdet.apis import init_detector, inference_detector
from mmengine.runner import load_checkpoint
import os
import pandas as pd
import json
from pycocotools.coco import COCO

# config file 들고오기
cfg = Config.fromfile('/data/ephemeral/home/mmdetection/projects/EfficientDet/configs/efficientdet_effb0_bifpn_8xb16-crop512-300e_coco.py')

root = '/data/ephemeral/home/dataset/'

epoch = 225 # 검증하고 싶은 모델 명의 epoch 입력
cfg.work_dir = '/data/ephemeral/home/mmdetection/EfficientDet'

cfg.model.bbox_head.num_classes = 10 # bbox_head가 있는 모델들을 클래스 수에 맞게 지정해줘야 함.

checkpoint_path = os.path.join(cfg.work_dir, f'epoch_{epoch}.pth')

model = init_detector(cfg, checkpoint_path, device='cuda:0')

coco = COCO(os.path.join(root, 'test.json'))
img_ids = coco.getImgIds()

prediction_strings = []
file_names = []
results = []
for img_id in img_ids:
    # 이미지 정보 가져오기
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(root, img_info['file_name'])
    
    # 추론 수행
    result = inference_detector(model, img_path)
    
    # 결과 저장을 위한 리스트
    prediction_string = ''
    
    # 바운딩 박스, 점수, 라벨 추출
    bboxes = result.pred_instances.bboxes.cpu()
    scores = result.pred_instances.scores.cpu()
    labels = result.pred_instances.labels.cpu()
    
    # 각 클래스별 결과 처리
    for i in range(len(bboxes)):
        label = labels[i].item()
        score = scores[i].item()
        box = bboxes[i]
        
        # JSON 파일을 위한 결과 딕셔너리
        result_dict = {
            "image_id": img_info['id'],
            "category_id": label,
            "bbox": [box[0].item(), box[1].item(), box[2].item(), box[3].item()],
            "score": score
        }
        results.append(result_dict)

# JSON 파일 생성
json_file = os.path.join(cfg.work_dir, f'val_prediction.json')
with open(json_file, 'w') as f:
    json.dump(results, f)

print("CSV and JSON files have been created!")
