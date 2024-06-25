# 프로젝트명

📢 2024년 1학기 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다

## 소개

- 자동차 사고 과실비율 분쟁 심의 청구가 빠르게 증가하는 추세를 보이고 있으며, 자율주행의 상용화에 따라 자동차 사고 발생 시 책임 및 과실 비율 산정에 대한 명확한 기준의 필요성이 증대되고 있다.
- 이에, 현 기술 수준과 미래 자율주행 상용화 시, 모두에 필요한 AI 기반 블랙박스 사고 영상에 대한 주요 장면 추출 및 상황 분석을 통한 과실 비율 산정 알고리즘을 개발하고자 한다.

![](https://github.com/AIKU-Official/aiku-24-1-ai-dashcam-review/assets/63688973/a0ac842d-1ceb-4d13-a9a4-f94c1dc1edf5) <!-- intro image -->

## 방법론
Video recognition deep learning (3DCNN + LSTM)
- 블랙박스 영상에 대해 사고 원인과 발생 부분에 해당하는 주요 영상을 추출한다. 추출한 영상에 대해 사고 상황을 딥러닝을 이용하여 분석하고, 과실 비율을 산정한다.
- 사용 모델
    - Soft Attention
        - 이미지에서 주요한 정보를 지닌 부분들에 집중
    - Dynamic Spatial Attention
        - Faster RCNN 기반 Soft Attention 수행
        - Full frame과 soft attention으로 Dynamic Spatial Attentino 수행
        - LSTM 기반 사고를 최대한 빠르게 예측

### Architecture
![](https://github.com/AIKU-Official/aiku-24-1-ai-dashcam-review/assets/63688973/4c3a21e7-8485-45a8-a0a3-6fd6360fb1f4) <!-- architecture image -->

### Dataset
[**Anticipating Accidents in Dashcam Videos**](https://aliensunmin.github.io/project/dashcam/)
해외 Dashcam video 영상 데이터
![](https://github.com/AIKU-Official/aiku-24-1-ai-dashcam-review/assets/63688973/a1ed8ea6-262f-47a9-993d-19ee42e3c4fb) <!-- dataset example image -->
![](https://github.com/AIKU-Official/aiku-24-1-ai-dashcam-review/assets/63688973/3d8bd99c-7ce8-4957-9fa0-c1d09909ed08) <!-- table -->

## 환경 설정

```
conda create --name dashcam python=3.8
conda activate dashcam
conda install pytorch torchvision torchaudio -c pytorch -c nvidia
git clone https://github.com/AIKU-Official/aiku-24-1-ai-dashcam-review.git
cd aiku-24-1-ai-dashcam-review
pip install -r requirements.txt
```

## 사용 방법

Training
```
python Run.py
```

Visualization
```
python Visualization.py
```

## 예시 결과



## 팀원

- [이진규](https://github.com/jjin-cong): Lead, Modeling, Training pipeline
- [정혜민](https://github.com/hmin27): Modeling, Training pipeline
- [김채현](https://github.com/kchyun): Feature extraction