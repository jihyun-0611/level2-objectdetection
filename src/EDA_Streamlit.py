import streamlit as st
import matplotlib.pyplot as plt
import json
import os
from PIL import Image, ImageDraw, ImageFont

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
    
    for ann in annotations:
        bbox = ann['bbox']
        category_id = ann['category_id']

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
    
    st.image(image)

    # plt는 느림
    #   plt.figure(figsize=(10, 10))
    #   plt.imshow(image) 

    # for ann in annotations:
    #     bbox = ann['bbox']
    #     category_id = ann['category_id']

    #     # bbox 좌표 계산
    #     x_min, y_min, width, height = bbox
    #     x_max = x_min + width
    #     y_max = y_min + height

    #     # bbox 그리기
    #     plt.gca().add_patch(plt.Rectangle((x_min, y_min), width, height, 
    #                                         linewidth=2, edgecolor=category_colors[category_id][0], facecolor='none'))

    #     # 텍스트 배경 그리기
    #     text = category_colors[category_id][1]
    #     text_size = plt.text(x_min, y_min, text, fontsize=12, color='white', 
    #                           bbox=dict(facecolor=category_colors[category_id][0], alpha=0.5))

    # plt.axis('off')
    # st.pyplot(plt)

st.title("hello ai")

with open('/home/ksy/Documents/naver_ai_tech/LV2/dataset/train.json', 'r') as f:
    train_data = json.load(f)

with open('/home/ksy/Documents/naver_ai_tech/LV2/dataset/train.json', 'r') as f:
    test_data = json.load(f)

# json 파일에서 이미지 파일명, id를 추출
image_files, image_ids = zip(*[(img['file_name'], img['id']) for img in train_data['images']])

# 이미지 파일명을 Select Box로 선택할 수 있도록 구성
selected_image = st.selectbox("Choose an image to display", image_files)

if selected_image:
    image_path = os.path.join('/home/ksy/Documents/naver_ai_tech/LV2/dataset', selected_image)

    select_index = image_files.index(selected_image)
    image_id = image_ids[select_index]
    annotations = [ann for ann in train_data['annotations'] if ann['image_id'] == image_id]

    image = Image.open(image_path)
    draw_bbox(image, annotations)