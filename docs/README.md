# 📦 Level2 Object Detection (개인 기여 정리)

## 프로젝트 개요

- 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기를 탐지하는 object detection 모델 개발 프로젝트
- **외부 레포지토리(mmdetection2/3, detectron 등)를 참고했으며, 본 레포지토리에는 핵심 코드와 실험 결과만 기록되어 있습니다**.
- 실험 시 작성한 configs는 다음 레포지토리에서 확인할 수 있습니다.
  - [Config](https://github.com/jihyun-0611/trash-object-detection/tree/main/configs/trash)

## 개인 기여

- **Wandb 자동 연동**: 팀원들이 실험할 때 학습 로그가 자동으로 wandb에 기록되도록 코드 수정
- **실험 기록 시스템 구축**: 실험 템플릿 작성 및 주도적으로 기록, 팀원들도 기록하도록 유도
- **Deformable DETR 및 Co-DETR 실험 및 분석** 

---

## 실험 내용 요약
### Challenge

- **객체 간 과도한 중첩 (Overlapping)**  
  - 하나의 이미지 내에 다수의 객체가 서로 많이 겹쳐 있는 경우가 많음.
  
- **잘못된 라벨 (Wrong Labeling)**  
  - 데이터셋에 잘못된 바운딩 박스 라벨이 존재하여 학습 과정에서 noise가 발생.

- **클래스 구분의 모호성 (Class Ambiguity)**  
  - Paper vs. Plastic Bag, Metal vs. Battery처럼 시각적으로 유사한 클래스 간 confusion이 빈번히 발생.

- **데이터 불균형 (Class Imbalance)**  
  - 특정 클래스(예: Paper, Plastic)가 다른 클래스에 비해 데이터 수가 많아 학습이 편향됨.

### 1. Deformable DETR 실험

- Pretrained 모델 사용, 10 epoch 학습
- 결과
  - 작은 bbox 및 유사 클래스 구분 어려움
  - 배경을 객체로 잘못 인식하는 문제 발생
  => 객체들이 많이 겹쳐져 있는 문제와 객체의 Feature Extraction 단계가 충분하지 못하다라고 판단. 

### 2. Co-DETR 실험

- Pretrained 모델 사용, 12 epoch 학습
- Gradient Accumulation 적용(batch 2 → accumulate 4)
- 결과
  - 배경 오탐지 감소
  - 비교적 작은 객체(Battery 등) 검출 성능 향상
  - General trash, Plastic, Paper pack 등 일부 클래스 구분 여전히 어려움
=> Feature Extraction 단계에서 ResNet50을 사용한 경우와 Swin을 사용한 경우 클래스 구분시 성능차이가 큼


## 느낀 점 및 한계

- Feature Extraction 구조가 (Swin vs ResNet50) 성능에 큰 영향을 미친다는 것을 체감
- 비교적 작은 객체와 밀집도가 높은 객체들에 대한 robust한 모델이 필요


---
> 본 문서는 개인 기여 및 실험 중심으로 작성되었습니다.  
> (원본 레포지토리에는 외부 레포지토리 코드는 포함되어 있지 않습니다.)
