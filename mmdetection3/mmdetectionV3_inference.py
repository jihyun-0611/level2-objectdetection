import os
import pandas as pd
from pycocotools.coco import COCO
from mmengine.config import Config
from mmengine.runner import Runner

from mmdet.apis import init_detector, inference_detector
from mmengine.runner import load_checkpoint
import os
import pandas as pd
import json
from pycocotools.coco import COCO

# config file 들고오기
cfg = Config.fromfile('/data/ephemeral/home/mmdetection/projects/EfficientDet/configs/efficientdet_effb0_bifpn_8xb16-crop512-300e_coco.py')

root = '/data/ephemeral/home/dataset/'

epoch = 225 # test하고 싶은 모델 명의 epoch 입력
cfg.work_dir = '/data/ephemeral/home/mmdetection/EfficientDet'

cfg.model.bbox_head.num_classes = 10 # bbox_head가 있는 모델들을 지정해줘야 함.

checkpoint_path = os.path.join(cfg.work_dir, f'epoch_{epoch}.pth')

model = init_detector(cfg, checkpoint_path, device='cuda:0')

coco = COCO(os.path.join(root, 'test.json'))
img_ids = coco.getImgIds()

prediction_strings = []
file_names = []
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
        
        # CSV 파일을 위한 PredictionString 생성
        prediction_string += f"{label} {score:.6f} {box[0].item():.6f} {box[1].item():.6f} {box[2].item():.6f} {box[3].item():.6f} "
    
    # 각 이미지의 파일명과 예측 결과 저장
    prediction_strings.append(prediction_string)
    file_names.append(img_info['file_name'])

# CSV 파일 생성
submission = pd.DataFrame()
submission['PredictionString'] = prediction_strings
submission['image_id'] = file_names
submission.to_csv(os.path.join(cfg.work_dir, f'submission.csv'), index=None)

print("CSV files have been created!")
