import argparse
import streamlit as st
import matplotlib.pyplot as plt
import json
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import albumentations as A
import pandas as pd
import cv2
import seaborn as sns
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='EDA with Streamlit')
    parser.add_argument('--dataset_path', type=str, default='/home/ksy/Documents/naver_ai_tech/LV2/dataset')
    parser.add_argument('--font_path', type=str, default='/home/ksy/Documents/naver_ai_tech/LV2/level2-objectdetection-cv-23/src/arial.ttf')
    parser.add_argument('--inference_path', type=str, default='/home/ksy/Documents/naver_ai_tech/LV2/level2-objectdetection-cv-23/src/inference_json/val_split_rand411_pred_latest.json')
    parser.add_argument('--validation_path', type=str, default='/home/ksy/Documents/naver_ai_tech/LV2/level2-objectdetection-cv-23/src/validation_json/val_split_random411.json')
    args = parser.parse_args()
    return args

# 카테고리별 색상 지정
category_colors = {
    0: ["red", "General trash"],       
    1: ["blue", "Paper"],      
    2: ["green", "Paper pack"],      
    3: ["orange", "Metal"],    
    4: ["yellow", "Glass"],    
    5: ["purple", "Plastic"],      
    6: ["cyan", "Styrofoam"],        
    7: ["magenta", "Plastic bag"],     
    8: ["brown", "Battery"],       
    9: ["pink", "Clothing"]         
}


LABEL_NAME = ["General trash", "Paper", "Paper pack", "Metal",
              "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

def load_train_json(dataset_path):
    with open(os.path.join(dataset_path, 'train.json'), 'r') as f:
        data = json.load(f)

    return data

def load_json(dataset_path):
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    return data

# bbox 좌표 계산
def calculate_bbox(bbox):
    # bbox 좌표 계산
    x_min, y_min, width, height = bbox
    x_max = x_min + width
    y_max = y_min + height

    return x_min, y_min, x_max, y_max

def draw_bbox_comm(opt, draw, bbox, category_id):
    category_name = category_colors[category_id][1]

    # bbox 좌표 계산
    x_min, y_min, x_max, y_max = calculate_bbox(bbox)

    # bbox 그리기
    draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=category_colors[category_id][0], width=3)
    draw_bbox_text(opt, draw, (x_min, y_min), category_name, category_colors[category_id][0])
    return category_name

# bbox 출력
def draw_train_bbox(opt ,image, annotations):
    # 이미지에 대한 draw 객체 생성
    draw = ImageDraw.Draw(image)

    # annotation 별 카테고리 카운트를 위한 딕셔너리
    annotation_table = {category_colors[i][1] : 0 for i in range(10)}

    for ann in annotations:
        bbox = ann['bbox']
        category_id = ann['category_id']

        category_name = draw_bbox_comm(opt, draw, bbox, category_id)

        annotation_table[category_name] += 1
    return image, annotation_table
    
# bbox 텍스트 출력
def draw_bbox_text(opt, draw, position ,category_name, color):
    # 폰트 설정
    font_size = 30
    font = ImageFont.truetype(opt.font_path, font_size) 

    # 텍스트 배경 사각형 좌표 계산
    text_bbox = draw.textbbox(position, category_name, font=font)  # 텍스트 경계 상자 계산
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

    background_bbox = [position[0], position[1] - 35, position[0] + text_width, position[1] - 5]

    # 텍스트 배경 그리기 (객체 색상으로 배경 채우기)
    draw.rectangle(background_bbox, fill=color)

    # 텍스트 그리기 (흰색으로)
    draw.text((position[0], position[1] - 35), category_name, fill="white", font=font)
    return draw

# annotation table 생성
def annotation_table_viz(annotation_table):
    # 0인 카테고리는 제외하고 DataFrame으로 변환
    df = pd.DataFrame({
        'category': [k for k, v in annotation_table.items() if v > 0],
        'count': [v for v in annotation_table.values() if v > 0]
    })

    # 카테고리별 count 내림차순 정렬
    df = df.sort_values(by='count', ascending=False)

    return df

# 파이 차트 생성
def pie_chart(df):
    # 파이 차트 색상을 카테고리 이름에 맞춰 설정
    colors = [category_colors[key][0] for category in df['category'] for key, value in category_colors.items() if value[1] == category]

    # 파이 차트 생성
    fig, ax = plt.subplots()
    ax.pie(df['count'], labels=df['category'], autopct='%1.1f%%', startangle=90, colors=colors)

    return fig

# augmentation 옵션 구성
def augmentation_compose(hflip, vflip, random_crop, rotate, brightness, hue, saturation, value, gauss_noise):
    augmentations = []
    if hflip:
        augmentations.append(A.HorizontalFlip(p=1.0))
    if vflip:
        augmentations.append(A.VerticalFlip(p=1.0))
    if rotate:
        augmentations.append(A.Rotate(limit=(rotate, rotate), p=1.0, border_mode=cv2.BORDER_CONSTANT))
    if brightness:
        augmentations.append(A.RandomBrightnessContrast(brightness_limit=(brightness - 1, brightness - 1), p=1.0))
    if random_crop:
        augmentations.append(A.RandomCrop(width=200, height=200, p=1.0))
    if hue or saturation or value:
        augmentations.append(A.HueSaturationValue(hue_shift_limit=hue, sat_shift_limit=saturation, val_shift_limit=value, p=1.0))
    if gauss_noise:
        augmentations.append(A.GaussNoise(var_limit=(gauss_noise, gauss_noise), p=1.0))

    return augmentations

def apply_augmentation(image, annotations, aug_method):
    image_np = np.array(image)

    # bbox 정보 추출
    bboxes = [ann['bbox'] for ann in annotations]
    # 카테고리 정보 추출
    category_ids = [ann['category_id'] for ann in annotations]
    # bbox, 카테고리, 이미지에 대한 augmentation 수행
    augmentation_image = aug_method(image=image_np,bboxes=bboxes, category_ids=category_ids)
    
    aug_image = Image.fromarray(augmentation_image['image'])

    # augmentation_image의 딕셔너리 key 값 (bboxes, category_ids)이 train.json(bbox, category_id)과 다르므로 일치시킴
    # augmentation을 할 때와 안할 때, bbox를 똑같이 출력해줘야 하니까 같은 key값이 필요함.
    new_annotations = [
        {'bbox': bbox, 'category_id': category_id}
        for bbox, category_id in zip(augmentation_image['bboxes'], augmentation_image['category_ids'])
    ]

    return aug_image, new_annotations


def bbox_heatmap(all_annotations, selected_category, image_size=(1024, 1024)):
    heatmap = np.zeros(image_size, dtype=np.float32)
    fig, ax = plt.subplots()
    
    # 선택된 카테고리에 맞는 annotation 필터링
    if selected_category != "All":
        # 선택된 카테고리 ID가 리스트이므로 [0]을 통해 ID 값만 추출
        category_id = [key for key, value in category_colors.items() if value[1] == selected_category][0]
        # 그때의 카테고리 ID에 해당하는 annotation만 추출
        annotations = [ann for ann in all_annotations if ann['category_id'] == category_id]
    else:
        annotations = all_annotations

    for ann in annotations:
        x_min, y_min, x_max, y_max = map(int, calculate_bbox(ann['bbox']))

        # Increment heatmap values for the region covered by the bbox
        heatmap[y_min:y_max, x_min:x_max] += 1
    
    sns.heatmap(heatmap, cmap="viridis", ax=ax)

    return fig

def count_by_category(annotations, sort=False):
    category_count = {category_colors[i][1]: 0 for i in range(10)}
    for ann in annotations:
        category_id = ann['category_id']
        category_name = category_colors[category_id][1]
        category_count[category_name] += 1
    
    fig, ax = plt.subplots()
    if sort:
        category_count = dict(sorted(category_count.items(), key=lambda x: x[1], reverse=True))
    graphs = ax.bar(category_count.keys(), category_count.values())

    for graph in graphs:
        height = graph.get_height()
        ax.annotate(f'{height}', xy=(graph.get_x() + graph.get_width() / 2, height), ha='center', va='bottom')
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")

    plt.xticks(rotation=90)

    return fig

# bbox 넓이에 따라 선택
def select_bbox_area(all_annotations, min_area, max_area):
    bbox_area = {i : [] for i in range(10)}

    for all_annotation in all_annotations:
        if min_area <= all_annotation["area"] <= max_area:
            bbox_area[all_annotation["category_id"]].append(all_annotation["area"])

    return bbox_area

# bbox area 시각화 (히스토그램)
def bbox_area_viz(bbox_area, selected_category):
    # 선택된 카테고리에 맞는 bbox 영역 필터링
    if selected_category != "All":
        category_id = [key for key, value in category_colors.items() if value[1] == selected_category][0]
        bbox_areas = bbox_area[category_id]
    else:
        # 모든 카테고리 선택 시, 모든 영역을 병합
        bbox_areas = [area for areas in bbox_area.values() for area in areas]

    # 그래프 생성
    fig, ax = plt.subplots()
    sns.histplot(bbox_areas, bins=20, ax=ax)
    ax.set_title(f"BBox Area Distribution for {selected_category}")
    ax.set_xlabel("Area")
    ax.set_ylabel("Count")

    count_bbox_area = len(bbox_areas)
    
    return fig, count_bbox_area

# 하나의 이미지 내에 있는 bbox들의 IoU 계산
def calculate_iou(val_bbox, inference_bbox):
    val_x_min, val_y_min, val_x_max, val_y_max = calculate_bbox(val_bbox)
    inf_x_min, inf_y_min, inf_x_max, inf_y_max = calculate_bbox(inference_bbox)

    # intersection 영역 계산
    x = max(0, min(val_x_max, inf_x_max) - max(val_x_min, inf_x_min))
    y = max(0, min(val_y_max, inf_y_max) - max(val_y_min, inf_y_min))
    intersection = x * y

    # union 영역 계산
    val_area = (val_x_max - val_x_min) * (val_y_max - val_y_min)
    inf_area = (inf_x_max - inf_x_min) * (inf_y_max - inf_y_min)

    # iou 계산
    iou = intersection / (val_area + inf_area - intersection)

    return iou

# 전체 IoU 계산 (벡터화)
def compute_iou(boxes, query_boxes):
    """
    Args:
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float
    Returns:
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """

    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float64)
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0]) *
            (query_boxes[k, 3] - query_boxes[k, 1])
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0])
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1])
                )
                if ih > 0:
                    ua = np.float64(
                        (boxes[n, 2] - boxes[n, 0]) *
                        (boxes[n, 3] - boxes[n, 1]) +
                        box_area - iw * ih
                    )
                        ### YOUR CODE HERE
                        ### ANSWER HERE ###
                        # 구현해야 할 변수 : overlaps[n, k]
                        # overlaps[n, k]
                    overlaps[n, k] = (iw * ih) / ua


    return overlaps

# IoU 기준으로 bbox 그리기
def draw_bbox_by_threshold(opt, image, val_annotations, inference_annotations, threshold, iou_flag):
    
    draw = ImageDraw.Draw(image)
    count = 0

    for inf_ann in inference_annotations:
        inf_bbox = inf_ann['bbox']
        inf_category_id = inf_ann['category_id']
        inf_score = inf_ann['score']

        iou_max = 0

        if iou_flag:
            for val_ann in val_annotations:
                val_category_id = val_ann['category_id']
                val_bbox = val_ann['bbox']

                if val_category_id == inf_category_id:
                    iou = calculate_iou(val_bbox, inf_bbox)
                    if iou >= threshold:
                        iou_max = iou
            # IoU가 설정된 threshold 이상인 bbox만 그리기
            if iou_max >= threshold:
                count += 1
                draw_bbox_comm(opt, draw, inf_bbox, inf_category_id)

        else:
            if inf_score >= threshold:
                count += 1
                draw_bbox_comm(opt, draw, inf_bbox, inf_category_id)

    return image, count

# coco json 파일을 mean_average_precision_for_boxes에 맞게 변환
def convert_coco_to_df(coco_json, is_prediction=False):
    data = json.load(open(coco_json))
    if is_prediction:
        records = [
            {
                'ImageID': item['image_id'],
                'LabelName': item['category_id'],
                'Conf': item['score'],
                'XMin': item['bbox'][0],
                'YMin': item['bbox'][1],
                'XMax': item['bbox'][2],
                'YMax': item['bbox'][3]
            }
            for item in data
        ]
    else:
        records = [
            {
                'ImageID': item['image_id'],
                'LabelName': item['category_id'],
                'XMin': item['bbox'][0],
                'YMin': item['bbox'][1],
                'XMax': item['bbox'][0] + item['bbox'][2],
                'YMax': item['bbox'][1] + item['bbox'][3]
            }
            for item in data['annotations']
        ]

    return pd.DataFrame(records)

def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def get_real_annotations(table):
    res = dict()
    ids = table['ImageID'].values.astype(np.str_)
    labels = table['LabelName'].values.astype(np.str_)
    xmin = table['XMin'].values.astype(np.float32)
    xmax = table['XMax'].values.astype(np.float32)
    ymin = table['YMin'].values.astype(np.float32)
    ymax = table['YMax'].values.astype(np.float32)

    for i in range(len(ids)):
        id = ids[i]
        label = labels[i]
        if id not in res:
            res[id] = dict()
        if label not in res[id]:
            res[id][label] = []
        box = [xmin[i], ymin[i], xmax[i], ymax[i]]
        res[id][label].append(box)

    return res

def get_detections(table):
    res = dict()
    ids = table['ImageID'].values.astype(np.str_)
    labels = table['LabelName'].values.astype(np.str_)
    scores = table['Conf'].values.astype(np.float32)
    xmin = table['XMin'].values.astype(np.float32)
    xmax = table['XMax'].values.astype(np.float32)
    ymin = table['YMin'].values.astype(np.float32)
    ymax = table['YMax'].values.astype(np.float32)

    for i in range(len(ids)):
        id = ids[i]
        label = labels[i]
        if id not in res:
            res[id] = dict()
        if label not in res[id]:
            res[id][label] = []
        box = [xmin[i], ymin[i], xmax[i], ymax[i], scores[i]]
        res[id][label].append(box)

    return res

def mean_average_precision_for_boxes(ann, pred, iou_threshold=0.5):
    """
    :param ann: path to CSV-file with annotations or numpy array of shape (N, 6)
    :param pred: path to CSV-file with predictions (detections) or numpy array of shape (N, 7)
    :param iou_threshold: IoU between boxes which count as 'match'. Default: 0.5
    :return: tuple, where first value is mAP and second values is dict with AP for each class.
    """

    if isinstance(ann, str):
        valid = pd.read_csv(ann)
    else:
        valid = pd.DataFrame(ann, columns=['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax'])

    if isinstance(pred, str):
        preds = pd.read_csv(pred)
    else:
        preds = pd.DataFrame(pred, columns=['ImageID', 'LabelName', 'Conf', 'XMin', 'XMax', 'YMin', 'YMax'])

# ImageID에 파일 경로 추가 및 4자리 zero-padding 적용
    valid['ImageID'] = valid['ImageID'].apply(lambda x: f'train/{str(x).zfill(4)}.jpg')
    preds['ImageID'] = preds['ImageID'].apply(lambda x: f'train/{str(x).zfill(4)}.jpg')

    ann_unique = valid['ImageID'].unique()
    preds_unique = preds['ImageID'].unique()

    print('Number of files in annotations: {}'.format(len(ann_unique)))
    print('Number of files in predictions: {}'.format(len(preds_unique)))

    unique_classes = valid['LabelName'].unique().astype(np.str_)

    print('Unique classes: {}'.format(len(unique_classes)))

    all_detections = get_detections(preds)
    all_annotations = get_real_annotations(valid)

    print('Detections length: {}'.format(len(all_detections)))
    print('Annotations length: {}'.format(len(all_annotations)))

    num_classes = len(unique_classes)
    rows = (num_classes + 4) // 5

    fig, axs = plt.subplots(rows, 5, figsize=(20, 4 * rows))
    axs = axs.ravel() 

    average_precisions = {}
    for idx, label in enumerate(sorted(unique_classes)):

        # Negative class
        if str(label) == 'nan':
            continue

        false_positives = []
        true_positives = []
        scores = []
        num_annotations = 0.0

        for i in range(len(ann_unique)):
            detections = []
            annotations = []
            id = ann_unique[i]
            if id in all_detections:
                if label in all_detections[id]:
                    detections = all_detections[id][label]
            if id in all_annotations:
                if label in all_annotations[id]:
                    annotations = all_annotations[id][label]

            if len(detections) == 0 and len(annotations) == 0:
                continue
            num_annotations += len(annotations)
            detected_annotations = []

            annotations = np.array(annotations, dtype=np.float64)
            for d in detections:
                scores.append(d[4])

                if len(annotations) == 0:
                    false_positives.append(1)
                    true_positives.append(0)
                    continue

                overlaps = compute_iou(np.expand_dims(np.array(d, dtype=np.float64), axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives.append(0)
                    true_positives.append(1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives.append(1)
                    true_positives.append(0)

        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        false_positives = np.array(false_positives)
        true_positives = np.array(true_positives)
        scores = np.array(scores)

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        ### YOUR CODE HERE
        ### ANSWER HERE ###
        recall = true_positives / num_annotations
        precision = true_positives / (true_positives + false_positives)

        # 각 label 별로 PR curve plot
        axs[idx].plot(recall, precision)
        axs[idx].set_title(f'{LABEL_NAME[int(label)]} PR curve', fontsize=15)
        axs[idx].set_xlabel('Recall', fontsize=12)
        axs[idx].set_ylabel('Precision', fontsize=12)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

        s1 = "{:10s} | {:.6f} | {:7d}".format(LABEL_NAME[int(label)], average_precision, int(num_annotations))
        print(s1)

        axs[idx].text(0.5, 0.1, f'AP: {average_precision:.4f}', fontsize=12, transform=axs[idx].transAxes, ha='center', color='blue')
    
    plt.tight_layout()
    present_classes = 0
    precision = 0
    for label, (average_precision, num_annotations) in average_precisions.items():
        if num_annotations > 0:
            present_classes += 1
            precision += average_precision

    # Compute the final mean average precision
    if present_classes > 0:
        mean_ap = precision / present_classes
    else:
        mean_ap = 0.0  # Or handle this case as needed (e.g., log a message or raise an exception)
        
    print('mAP: {:.6f}'.format(mean_ap))

    if num_classes < len(axs):
        for j in range(num_classes, len(axs)):
            fig.delaxes(axs[j])

    return mean_ap, average_precisions, fig

def main(opt):
    # st.set_page_config(layout="wide")

    menu = st.sidebar.radio("Menu", ["inference EDA", "Train 데이터 EDA", "Train 데이터 시각화 확인하기"])

    # json 파일 로드
    dataset_path = opt.dataset_path
    train_data = load_train_json(dataset_path)

    validation_data = load_json(opt.validation_path)

    inference_data = load_json(opt.inference_path)

    all_annotations = [ann for ann in train_data['annotations']]

    # json 파일에서 이미지 파일명, id를 추출
    image_files, image_ids = zip(*[(img['file_name'], img['id']) for img in train_data['images']])

    val_image_files, val_image_ids = zip(*[(img['file_name'], img['id']) for img in validation_data['images']])

    if menu == "Train 데이터 시각화 확인하기":
        if 'image_index' not in st.session_state:
            st.session_state.image_index = 0

        st.title("데이터 시각화 및 증강")

        # 버튼으로 이미지 이동
        prev_button, next_button = st.columns([1, 1])

        # 이미지 파일명을 Select Box로 선택할 수 있도록 구성
        selected_image = st.selectbox("Choose an image to display", image_files, index=st.session_state.image_index)
        if image_files.index(selected_image) != st.session_state.image_index:
            st.session_state.image_index = image_files.index(selected_image)
            st.rerun()

        # 파일 경로 설정
        image_path = os.path.join(dataset_path, selected_image)

        # 선택한 이미지에 대한 annotation 정보 추출
        image_id = image_ids[st.session_state.image_index]
        annotations = [ann for ann in train_data['annotations'] if ann['image_id'] == image_id]

        image = Image.open(image_path)

        # 사이드바에 augmentation 옵션 추가
        st.sidebar.title("Augmentation")

        # augmentation 옵션 설정
        hflip = st.sidebar.checkbox("Horizontal Flip")
        vflip = st.sidebar.checkbox("Vertical Flip")
        random_crop = st.sidebar.checkbox("Random Crop")
        rotate = st.sidebar.slider("Rotate", -180, 180, 0)
        brightness = st.sidebar.slider("Brightness", 0.0, 2.0, 1.0)
        gauss_noise = st.sidebar.slider("Gauss Noise", 0, 50, 0)

        st.sidebar.header("HueSaturationValue")
        hue = st.sidebar.slider("Hue Shift", -20, 20, 0)
        saturation = st.sidebar.slider("Saturation Shift", -30, 30, 0)
        value = st.sidebar.slider("Value Shift", -30, 30, 0)

        augmentations = augmentation_compose(hflip, vflip, random_crop, rotate, brightness, hue, saturation, value, gauss_noise)

        if augmentations:
            # augmentation 메소드 생성. 
            # bbox 정보를 coco format(x_min, y_min, width, height)으로 설정 
            #  -> 제공된 쓰레기 데이터의 bbox가 coco format을 따름
            aug_method = A.Compose(augmentations, bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
            image, annotations = apply_augmentation(image, annotations, aug_method)

        image, annotation_table = draw_train_bbox(opt, image, annotations)

        # 이미지 출력
        st.image(image)

        # annotation table 및 파이 차트 출력 setting
        st.header("Annotation Table")
        category_count, category_pie = st.columns([1, 2])

        # annotation table 출력
        df = annotation_table_viz(annotation_table)
        category_count.dataframe(df)

        # 파이 차트 출력
        category_pie.pyplot(pie_chart(df))

        # 이전 이미지 버튼
        if prev_button.button("Previous Image"):
            if st.session_state.image_index > 0:
                st.session_state.image_index -= 1
                st.rerun()

        # 다음 이미지 버튼
        if next_button.button("Next Image"):
            if st.session_state.image_index < len(image_files) - 1:
                st.session_state.image_index += 1
                st.rerun()

    if menu == "Train 데이터 EDA":
        st.header("EDA")
        
        count_by_category_header, sort_button = st.columns([4, 1])

        count_by_category_header.subheader("카테고리별 annotation 수 count")
        if 'sort_count_by_category' not in st.session_state:
            st.session_state.sort_count_by_category = False

        category_count_viz = count_by_category(train_data['annotations'], sort=st.session_state.sort_count_by_category)
        st.pyplot(category_count_viz)

        if sort_button.button("Sort"):
            st.session_state.sort_count_by_category = not st.session_state.sort_count_by_category
            st.rerun()

        st.subheader("카테고리별 bbox heatmap")
        heatmap_category, heatmap_viz = st.columns([1, 4])

        # 카테고리 별 heatmap 시각화를 위한 radio button
        selected_category = heatmap_category.radio(
            "바운딩 박스를 확인할 카테고리를 선택하세요",
            options=[category_colors[i][1] for i in range(10)] + ["All"],
            key="heatmap_category_selection" 
        )

        # 모든 annotation 데이터를 기반으로 히트맵 생성
        heatmap = bbox_heatmap(all_annotations, selected_category)

        # 히트맵 출력
        heatmap_viz.pyplot(heatmap)

        st.subheader("카테고리 별 bbox 크기 분포")
        col_category_bbox_area, col_bbox_area = st.columns([1, 4])

        selected_category_for_bbox_area = col_category_bbox_area.radio(
            "바운딩 박스를 확인할 카테고리를 선택하세요",
            options=[category_colors[i][1] for i in range(10)] + ["All"],
            key="bbox_area_category_selection"
        )

        min_bbox_area, max_bbox_area = st.columns(2)

        min_bbox_area = min_bbox_area.number_input("최소 bbox 넓이", min_value=0, max_value=1024*1024, value=250000)
        max_bbox_area = max_bbox_area.number_input("최대 bbox 넓이", min_value=0, max_value=1024*1024, value=680000)
        bbox_area_slider = st.slider("확인할 bbox의 넓이를 선택하세요", 0, 1024*1024, (min_bbox_area, max_bbox_area))
        selected_bbox_area = select_bbox_area(all_annotations, bbox_area_slider[0], bbox_area_slider[1])
        bbox_area_histogram, count_bbox_area= bbox_area_viz(selected_bbox_area, selected_category_for_bbox_area)
        col_bbox_area.pyplot(bbox_area_histogram)
        st.write (f"선택된 bbox 영역의 수: {count_bbox_area}")
    
    if menu == "inference EDA":
        if 'inference_image_index' not in st.session_state:
            st.session_state.inference_image_index = 0

        st.title("inference EDA")

        if 'mean_ap' not in st.session_state:
            st.subheader("전체 inference bbox에 대한 카테고리 별 AP")

            # AP 계산: 이미 저장되어 있지 않은 경우에만 수행
            validation_data_for_ap = convert_coco_to_df(opt.validation_path, is_prediction=False)
            inference_data_for_ap = convert_coco_to_df(opt.inference_path, is_prediction=True)

            mean_ap, average_precisions, plot_ap = mean_average_precision_for_boxes(validation_data_for_ap, inference_data_for_ap, 0.5)
            st.session_state.mean_ap = mean_ap
            st.session_state.plot_ap = plot_ap

        # mAP와 플롯 표시
        st.write(f"mAP : {st.session_state.mean_ap}")
        st.pyplot(st.session_state.plot_ap)

        # 버튼으로 이미지 이동
        prev_button, next_button = st.columns([1, 1])

        # 이미지 파일명을 Select Box로 선택할 수 있도록 구성
        selected_image = st.selectbox("이미지를 선택하세요", val_image_files, index=st.session_state.inference_image_index)
        if val_image_files.index(selected_image) != st.session_state.inference_image_index:
            st.session_state.inference_image_index = val_image_files.index(selected_image)
            st.rerun()
        
        # 파일 경로 설정
        image_path = os.path.join(dataset_path, selected_image)

        # 선택한 이미지에 대한 annotation 정보 추출
        image_id = val_image_ids[st.session_state.inference_image_index]
        val_annotations = [ann for ann in validation_data['annotations'] if ann['image_id'] == image_id]
        inference_annotations = [ann for ann in inference_data if ann['image_id'] == image_id]

        image = Image.open(image_path)
        iou_image = image.copy()
        score_image = image.copy()

        train_image, _ = draw_train_bbox(opt, image, val_annotations)

        st.subheader("IoU Threshold로 보는 bbox :rocket:")
        iou_threshold = st.slider("IoU Threshold", 0.0, 1.0, 0.5)

        st.write("Threshold가 0일 땐 모든 예측된 박스가 표시되며 그 외에는 validation 카테고리와 관련된 박스만 표시됩니다.")
        inference_image, iou_count = draw_bbox_by_threshold(opt, iou_image, val_annotations ,inference_annotations, iou_threshold, iou_flag=True)

        validation_image_iou_col, inference_image_iou_col = st.columns([1, 1])
        validation_image_iou_col.image(train_image)
        validation_image_iou_col.write(f"선택한 vaildation bbox의 갯수는 {len(val_annotations)}개 입니다.")
        
        inference_image_iou_col.image(inference_image)
        inference_image_iou_col.write(f"추론한 bbox의 갯수는 {iou_count}개 입니다.")

        ## confidence score
        st.subheader("confidence score로 보는 bbox :fire:")
        score_threshold = st.slider("Confidence Score Threshold", 0.0, 1.0, 0.5)
        st.write("Threshold 이상의 confidence score를 가진 박스만 표시됩니다.")
        inference_image_score, confidence_score_count = draw_bbox_by_threshold(opt, score_image, val_annotations, inference_annotations, score_threshold, iou_flag=False)

        validation_image_score_col, inference_image_score_col = st.columns([1, 1])
        validation_image_score_col.image(train_image)
        validation_image_score_col.write(f"선택한 vaildation bbox의 갯수는 {len(val_annotations)}개 입니다.")
        
        inference_image_score_col.image(inference_image_score)
        inference_image_score_col.write(f"추론한 bbox의 갯수는 {confidence_score_count}개 입니다.")

if __name__ == '__main__':
    opt = parse_args()
    main(opt)