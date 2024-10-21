# 학습을 마친 모델의 confusion matrix를 출력하는 과정

# 설명 : 
# mmdetection/tools/test.py에서 validation data에 대한 예측 결과를 pickle 파일로 생성하고
# mmdetection/tools/analysis_tools/confusion_matrix.py에서 pickle 파일과 ground truth를 비교하여 confusion matrix를 이미지로 생성
# cf) 학습한 모델에 맞춘 별도의 config 생성이 필요하며, validation data로 test를 진행하므로 data->test->ann_file에 validation json 파일을 지정해줘야함

# config_path : 학습한 모델의 config 파일 경로(.py파일) 
# checkpoint_path : 학습한 모델의 경로(.pth파일)
# save_dir : pkl파일과 confusion matrix 결과를 저장할 폴더
# pkl_path : test.py의 결과 생성될 pickel 파일 경로(.pkl)

config_path="/data/ephemeral/home/baseline/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_trash.py"
checkpoint_path="/data/ephemeral/home/ys/work_dirs/faster_rcnn_r50_fpn_1x_coco_trash/latest.pth"
save_dir="/data/ephemeral/home/ys/work_dirs/faster_rcnn_r50_fpn_1x_coco_trash"
pkl_path="$save_dir/result.pkl"

# test.py path & confusion.py path
testpy_path="/data/ephemeral/home/baseline/mmdetection/tools/test.py"
confusion_path="/data/ephemeral/home/baseline/mmdetection/tools/analysis_tools/confusion_matrix.py"

# test.py
python "$testpy_path" \
        "$config_path" \
        "$checkpoint_path" \
        --work-dir "$save_dir" \
        --out "$pkl_path" --eval bbox

# confusion.py
python "$confusion_path" \
        "$config_path" \
        "$pkl_path" \
        "$save_dir"