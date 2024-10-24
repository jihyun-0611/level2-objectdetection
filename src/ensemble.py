from ensemble_boxes import nms, weighted_boxes_fusion,non_maximum_weighted, soft_nms
from pycocotools.coco import COCO
import numpy as np
import pandas as pd

# csv 파일 이름 적어주세요
root = ['alpha.csv',]

# csv 저장폴더 경로 지정
submission_dir = '../../../submission/'

submission_files = [submission_dir + r for r in root]
print(submission_files)
submission_df = [pd.read_csv(file) for file in submission_files]

# image_id 추출
image_ids = submission_df[0]['image_id'].tolist()

# ensemble 할 file의 image 정보를 불러오기 위한 json
# 경로가 다르다면 수정해주세요.
annotation = '../../../dataset/test.json'
coco = COCO(annotation)

# 앙상블
### :참고사항:
#### ensemble_type 변수를 통해서 사용할 Ensemble 기법을 선택해주세요
    # ├── NMS 
    # ├── WBF 
    # ├── NMW 
    # └── Soft_NMS 

prediction_strings = []
file_names = []

# 앙상블 기법 선택
ensemble_type = 'soft-nms' #[nms, wbf, nmw, soft-nms]


# 수치 변경가능!!!
# nms, soft-nms의 경우 0.5, wbf와 nmw의 경우 0.55 가 기본값
iou_thr = 0.5 #iou threshold 값 설정


# 아래 두개 기법 같은 경우는 설정 변경에 따른 정확도 상승 기대값이 높아 세부 설정 하도록 했습니다.
# WBF 세부 설정

# 결합된 박스의 최종 점수를 계산
wbf_conf_type='avg' # ['avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg']
wbf_allows_overflow = False # {True: 가중치 합 > 1, False: 가중치 합 1로 고정} # weights 값도 함께 변경 해주세요.
wbf_skip_box_thr = 0.0 # 값보다 낮은 박스는 결합 과정 중 무시 됨


# Soft-NMS 세부 설정 (soft-nms에만 포함 되어있습니다.)

method = 2 # 1 - linear soft-NMS, 2 - gaussian soft-NMS, 3 - standard NMS 기본값: 2
sn_sigma = 0.5 # Gaussian soft-NMS 방식 사용 시 설정 분산 값  기본값: 0.5
sn_thresh = 0.001 # 점수가 값보다 낮으면 박스를 제거  기본값: 0.001



# 각 모델에 동일한 가중치를 적용
# 수치 변경 가능하며 필요 없는 경우 None 으로 바꿔주세요
weights = [1] * len(submission_df)  


# NMS
if ensemble_type == 'nms': 
    
    # 각 image id 별로 submission file에서 box좌표 추출
    for i, image_id in enumerate(image_ids):
        prediction_string = ''
        boxes_list = []
        scores_list = []
        labels_list = []
        image_info = coco.loadImgs(i)[0]
        # 각 submission file 별로 prediction box좌표 불러오기
        for df in submission_df:
            predict_string = df[df['image_id'] == image_id]['PredictionString'].tolist()[0]
            predict_list = str(predict_string).split()

            if len(predict_list)==0 or len(predict_list)==1:
                continue

            predict_list = np.reshape(predict_list, (-1, 6))
            box_list = []

            for box in predict_list[:, 2:6].tolist():
                # box의 각 좌표를 float형으로 변환한 후 image의 넓이와 높이로 각각 정규화
                image_width = image_info['width']
                image_height = image_info['height']
                box[0] = float(box[0]) / image_width
                box[1] = float(box[1]) / image_height
                box[2] = float(box[2]) / image_width
                box[3] = float(box[3]) / image_height
                box_list.append(box)

            boxes_list.append(box_list)
            scores_list.append(list(map(float, predict_list[:, 1].tolist())))
            labels_list.append(list(map(int, predict_list[:, 0].tolist())))

        # 예측 box가 있다면 이를 ensemble 수행
        if len(boxes_list):
            boxes, scores, labels = nms(boxes_list, scores_list, labels_list, iou_thr=iou_thr)
            for box, score, label in zip(boxes, scores, labels):
                prediction_string += str(label) + ' ' + str(score) + ' ' + str(box[0] * image_info['width']) + ' ' + str(box[1] * image_info['height']) + ' ' + str(box[2] * image_info['width']) + ' ' + str(box[3] * image_info['height']) + ' '

        prediction_strings.append(prediction_string)
        file_names.append(image_id)

#WBF
elif ensemble_type == 'wbf':
    # 각 image id 별로 submission file에서 box좌표 추출
    for i, image_id in enumerate(image_ids):
        prediction_string = ''
        boxes_list = []
        scores_list = []
        labels_list = []
        image_info = coco.loadImgs(i)[0]
        # 각 submission file 별로 prediction box좌표 불러오기
        for df in submission_df:
            predict_string = df[df['image_id'] == image_id]['PredictionString'].tolist()[0]
            predict_list = str(predict_string).split()

            if len(predict_list)==0 or len(predict_list)==1:
                continue

            predict_list = np.reshape(predict_list, (-1, 6))
            box_list = []

            for box in predict_list[:, 2:6].tolist():
                # box의 각 좌표를 float형으로 변환한 후 image의 넓이와 높이로 각각 정규화
                image_width = image_info['width']
                image_height = image_info['height']
                box[0] = float(box[0]) / image_width
                box[1] = float(box[1]) / image_height
                box[2] = float(box[2]) / image_width
                box[3] = float(box[3]) / image_height
                box_list.append(box)

            boxes_list.append(box_list)
            scores_list.append(list(map(float, predict_list[:, 1].tolist())))
            labels_list.append(list(map(int, predict_list[:, 0].tolist())))

        # 예측 box가 있다면 이를 ensemble 수행
        if len(boxes_list):
            boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, iou_thr=iou_thr,conf_type=wbf_conf_type,allows_overflow=wbf_allows_overflow, skip_box_thr=wbf_skip_box_thr)
            for box, score, label in zip(boxes, scores, labels):
                prediction_string += str(label) + ' ' + str(score) + ' ' + str(box[0] * image_info['width']) + ' ' + str(box[1] * image_info['height']) + ' ' + str(box[2] * image_info['width']) + ' ' + str(box[3] * image_info['height']) + ' '

        prediction_strings.append(prediction_string)
        file_names.append(image_id)


#NMW
elif ensemble_type == 'nmw':
    # 각 image id 별로 submission file에서 box좌표 추출
    for i, image_id in enumerate(image_ids):
        prediction_string = ''
        boxes_list = []
        scores_list = []
        labels_list = []
        image_info = coco.loadImgs(i)[0]
        # 각 submission file 별로 prediction box좌표 불러오기
        for df in submission_df:
            predict_string = df[df['image_id'] == image_id]['PredictionString'].tolist()[0]
            predict_list = str(predict_string).split()

            if len(predict_list)==0 or len(predict_list)==1:
                continue

            predict_list = np.reshape(predict_list, (-1, 6))
            box_list = []

            for box in predict_list[:, 2:6].tolist():
                # box의 각 좌표를 float형으로 변환한 후 image의 넓이와 높이로 각각 정규화
                image_width = image_info['width']
                image_height = image_info['height']
                box[0] = float(box[0]) / image_width
                box[1] = float(box[1]) / image_height
                box[2] = float(box[2]) / image_width
                box[3] = float(box[3]) / image_height
                box_list.append(box)

            boxes_list.append(box_list)
            scores_list.append(list(map(float, predict_list[:, 1].tolist())))
            labels_list.append(list(map(int, predict_list[:, 0].tolist())))

        # 예측 box가 있다면 이를 ensemble 수행
        if len(boxes_list):
            boxes, scores, labels = non_maximum_weighted(boxes_list, scores_list, labels_list, iou_thr=iou_thr)
            for box, score, label in zip(boxes, scores, labels):
                prediction_string += str(int(label)) + ' ' + str(score) + ' ' + str(box[0] * image_info['width']) + ' ' + str(box[1] * image_info['height']) + ' ' + str(box[2] * image_info['width']) + ' ' + str(box[3] * image_info['height']) + ' '

        prediction_strings.append(prediction_string)
        file_names.append(image_id)
        

# Soft-NMS
elif ensemble_type == 'soft-nms':
    # 각 image id 별로 submission file에서 box좌표 추출
    for i, image_id in enumerate(image_ids):
        prediction_string = ''
        boxes_list = []
        scores_list = []
        labels_list = []
        image_info = coco.loadImgs(i)[0]
        # 각 submission file 별로 prediction box좌표 불러오기
        for df in submission_df:
            predict_string = df[df['image_id'] == image_id]['PredictionString'].tolist()[0]
            predict_list = str(predict_string).split()

            if len(predict_list)==0 or len(predict_list)==1:
                continue

            predict_list = np.reshape(predict_list, (-1, 6))
            box_list = []

            for box in predict_list[:, 2:6].tolist():
                # box의 각 좌표를 float형으로 변환한 후 image의 넓이와 높이로 각각 정규화
                image_width = image_info['width']
                image_height = image_info['height']
                box[0] = float(box[0]) / image_width
                box[1] = float(box[1]) / image_height
                box[2] = float(box[2]) / image_width
                box[3] = float(box[3]) / image_height
                box_list.append(box)

            boxes_list.append(box_list)
            scores_list.append(list(map(float, predict_list[:, 1].tolist())))
            labels_list.append(list(map(int, predict_list[:, 0].tolist())))

        # 예측 box가 있다면 이를 ensemble 수행
        if len(boxes_list):
            boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list, iou_thr=iou_thr, method=method, sigma=sn_sigma, thresh=sn_thresh)
            for box, score, label in zip(boxes, scores, labels):
                prediction_string += str(label) + ' ' + str(score) + ' ' + str(box[0] * image_info['width']) + ' ' + str(box[1] * image_info['height']) + ' ' + str(box[2] * image_info['width']) + ' ' + str(box[3] * image_info['height']) + ' '

        prediction_strings.append(prediction_string)
        file_names.append(image_id)

submission = pd.DataFrame()
submission['PredictionString'] = prediction_strings
submission['image_id'] = file_names

# csv 파일 저장
submission.to_csv(submission_dir+'output_ensembles.csv', index=None)
submission.head() 