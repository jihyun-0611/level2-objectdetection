import streamlit as st
import matplotlib.pyplot as plt
import json
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import albumentations as A
import pandas as pd

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

# bbox 출력
def draw_bbox(image, annotations):
    # 이미지에 대한 draw 객체 생성
    draw = ImageDraw.Draw(image)

    # annotation 별 카테고리 카운트를 위한 딕셔너리
    annotation_table = {category_colors[i][1] : 0 for i in range(10)}

    for ann in annotations:
        bbox = ann['bbox']
        category_id = ann['category_id']

        # annotation 별 카테고리 카운트 증가
        category_name = category_colors[category_id][1]
        annotation_table[category_name] += 1

        # bbox 좌표 계산
        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height
        
        # bbox 그리기
        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=category_colors[category_id][0], width=3)

        # 폰트 설정
        font_size = 30
        font = ImageFont.truetype("/home/ksy/Documents/naver_ai_tech/LV2/level2-objectdetection-cv-23/src/arial.ttf", font_size) 

        # 텍스트 배경 사각형 좌표 계산
        text = category_colors[category_id][1]
        text_bbox = draw.textbbox((x_min, y_min), text, font=font)  # 텍스트 경계 상자 계산
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        background_bbox = [x_min, y_min - 35, x_min + text_width, y_min]

        # 텍스트 배경 그리기 (객체 색상으로 배경 채우기)
        draw.rectangle(background_bbox, fill=category_colors[category_id][0])

        # 텍스트 그리기 (흰색으로)
        draw.text((x_min, y_min - 35), text, fill="white", font=font)
    
    # 0인 카테고리는 제외하고 DataFrame으로 변환
    df = pd.DataFrame({
        'category': [k for k, v in annotation_table.items() if v > 0],
        'count': [v for v in annotation_table.values() if v > 0]
    })

    # 카테고리별 count 내림차순 정렬
    df = df.sort_values(by='count', ascending=False)

    # 파이 차트 색상을 카테고리 이름에 맞춰 설정
    colors = [category_colors[key][0] for category in df['category'] for key, value in category_colors.items() if value[1] == category]

    fig, ax = plt.subplots()
    ax.pie(df['count'], labels=df['category'], autopct='%1.1f%%', startangle=90, colors=colors)
    st.image(image)

    st.header("Annotation Table")
    category_count, category_pie = st.columns([1, 2])

    category_count.dataframe(df)
    category_pie.pyplot(fig)

def augmentation(image, annotations, aug_method):
    image_np = np.array(image)

    # bbox 정보 추출
    bboxes = [ann['bbox'] for ann in annotations]
    # 카테고리 정보 추출
    category_ids = [ann['category_id'] for ann in annotations]
    # bbox, 카테고리, 이미지에 대한 augmentation 수행
    augmentation_image = aug_method(image=image_np,bboxes=bboxes, category_ids=category_ids)

    aug_image = Image.fromarray(augmentation_image['image'])

    for i, ann in enumerate(annotations):
        ann['bbox'] = augmentation_image['bboxes'][i]

    return aug_image, annotations

st.title("데이터 시각화 및 증강")

with open('/home/ksy/Documents/naver_ai_tech/LV2/dataset/train.json', 'r') as f:
    train_data = json.load(f)

with open('/home/ksy/Documents/naver_ai_tech/LV2/dataset/test.json', 'r') as f:
    test_data = json.load(f)

# json 파일에서 이미지 파일명, id를 추출
image_files, image_ids = zip(*[(img['file_name'], img['id']) for img in train_data['images']])

if 'image_select' not in st.session_state:
    st.session_state.image_select = 0

# 이미지 파일명을 Select Box로 선택할 수 있도록 구성
selected_image = st.selectbox("Choose an image to display", image_files, index=st.session_state.image_select)
if image_files.index(selected_image) != st.session_state.image_select:
    st.session_state.image_select = image_files.index(selected_image)
    st.rerun()

# 파일 경로 설정
image_path = os.path.join('/home/ksy/Documents/naver_ai_tech/LV2/dataset', selected_image)

# 선택한 이미지에 대한 annotation 정보 추출
image_id = image_ids[st.session_state.image_select]
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

# augmentation 옵션에 따라 이미지 변환
augmentations = []
if hflip:
    augmentations.append(A.HorizontalFlip(p=1.0))
if vflip:
    augmentations.append(A.VerticalFlip(p=1.0))
if rotate:
    augmentations.append(A.Rotate(limit=(rotate, rotate), p=1.0))
if brightness:
    augmentations.append(A.RandomBrightnessContrast(brightness_limit=(brightness - 1, brightness - 1), p=1.0))
if random_crop:
    augmentations.append(A.RandomCrop(width=200, height=200, p=1.0))
if hue or saturation or value:
    augmentations.append(A.HueSaturationValue(hue_shift_limit=hue, sat_shift_limit=saturation, val_shift_limit=value, p=1.0))
if gauss_noise:
    augmentations.append(A.GaussNoise(var_limit=(gauss_noise, gauss_noise), p=1.0))


if augmentations:
    # augmentation 메소드 생성. 
    # bbox 정보를 coco format(x_min, y_min, width, height)으로 설정 
    #  -> 제공된 쓰레기 데이터의 bbox가 coco format을 따름
    aug_method = A.Compose(augmentations, bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
    image, annotations = augmentation(image, annotations, aug_method)

draw_bbox(image, annotations)

# 버튼으로 이미지 이동
prev_button, next_button = st.columns([1, 1])

# 이전 이미지 버튼
if prev_button.button("Previous Image"):
    if st.session_state.image_select > 0:
        st.session_state.image_select -= 1
        st.rerun()

# 다음 이미지 버튼
if next_button.button("Next Image"):
    if st.session_state.image_select < len(image_files) - 1:
        st.session_state.image_select += 1
        st.rerun() # 현재 블록을 재실행