dataset_path='/home/ksy/Documents/naver_ai_tech/LV2/dataset' # 본인의 dataset 경로를 입력하세요. test, train 디렉토리가 있는 경로입니다.
font_path='/home/ksy/Documents/naver_ai_tech/LV2/level2-objectdetection-cv-23/src/arial.ttf' # 본인의 font 경로를 입력하세요. ttf 파일 경로입니다.
inference_path='/home/ksy/Documents/naver_ai_tech/LV2/level2-objectdetection-cv-23/src/inference_json/val_split_rand411_pred_latest.json' # 본인의 inference json 파일 경로를 입력하세요.
validation_path='/home/ksy/Documents/naver_ai_tech/LV2/level2-objectdetection-cv-23/src/validation_json/val_split_random411.json' # 본인의 validation json 파일 경로를 입력하세요.

streamlit run EDA_Streamlit.py -- --dataset_path "$dataset_path" --font_path "$font_path"
