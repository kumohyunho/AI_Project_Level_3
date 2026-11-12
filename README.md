# 지능형 객체 추적 시스템

## Windows에서 실행하기

Python 3.11 이상을 권장합니다. 기존 프로젝트에 적용하려면 ZIP 안의 `TrackingGUI` 폴더 내용을 기존 `main.py`가 있는 폴더에 복사하세요. 기존 `.venv`, `sample.mp4`, `yolo11n.pt`를 그대로 사용하면 됩니다.

가상 환경이 없는 경우 PowerShell에서 먼저 `python -m venv .venv`를 실행하세요. 이후 `main.py`가 있는 폴더에서 다음 두 줄을 순서대로 실행합니다.

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -r .\requirements.txt
& ".\.venv\Scripts\python.exe" .\main.py
```

## 사람 추적

1. **영상 열기**에서 원본 영상(예: `sample.mp4`)을 선택합니다.
2. **ByteTrack**을 선택하고 **▶ 시작**을 누릅니다.
3. 사람별 박스, ID, 최근 이동 궤적과 오른쪽 객체 목록을 확인합니다. 표시 옵션은 실행 중에도 바꿀 수 있습니다.
4. **일시정지 → 시작**으로 이어서 추적합니다. **중지 → 시작**으로 처음부터 새로 추적합니다.

사전 학습된 YOLO11n으로 사람을 검출하고 ByteTrack으로 연속 프레임을 연결합니다. `models/person.pt` → `models/coco.pt` → `main.py` 옆의 `yolo11n.pt` 순으로 선택합니다. 모두 없으면 Ultralytics가 기본 가중치를 다운로드합니다. MOT17 영상은 추적 입력으로 사용할 수 있습니다.

CPU에서 영상을 프레임 순서대로 처리합니다. 추적 모드의 진행 막대는 처리 위치를 표시합니다. 원하는 위치로 이동하려면 **원본 재생**을 선택하세요. 처리 FPS와 프레임당 시간은 직전 프레임의 모델 추론 및 추적 호출 시간을 기준으로 표시합니다. 거리는 박스 중심의 누적 이동량이며 단위는 픽셀입니다. 가림이나 검출 누락으로 ID가 바뀔 수 있습니다.

`main.py`는 Qt 화면과 재생 제어를 담당하고, `tracking.py`는 영상 읽기, 모델 실행, 추적 통계 계산을 담당합니다. 성능 비교 탭은 정답 데이터와 평가 엔진을 연결하는 후속 단계입니다.

## 데이터셋과 가중치

| 데이터셋 | 사용처 | 결과 가중치 |
|---|---|---|
| [COCO](https://cocodataset.org/#download) | 사람·고양이·개·새 등을 포함한 객체 검출. GUI 추적 대상은 사람입니다. | `models/coco.pt` |
| [CrowdHuman](https://www.crowdhuman.org/download.html) | 가림이 많은 장면에서 사람 검출기를 추가 학습합니다. 공식 `fbox`를 사용하고 `mask`와 `ignore` 박스는 YOLO 라벨에서 제외합니다. | `models/person.pt` |
| [WIDER FACE](https://huggingface.co/datasets/CUHK-CSE/wider_face/tree/main/data) | 얼굴 검출기를 별도 학습합니다. 파일을 지정 위치에 두면 추적 영상에서 검출된 얼굴이 자동으로 모자이크 처리됩니다. | `models/face.pt` |
| [MOT17](https://motchallenge.net/data/MOT17/) · [MOT20](https://motchallenge.net/data/MOT20/) | 정답 ID가 포함된 train 시퀀스로 ByteTrack 추적을 평가합니다. | 평가용 텍스트 |

ID는 영상 안에서 ByteTrack이 부여하는 임시 번호입니다. WIDER FACE 얼굴 검출 결과는 화면 모자이크에만 사용합니다. 원본 재생 모드는 파일 자체를 그대로 재생하며, 추적 모드에서도 검출이 누락된 얼굴에는 모자이크가 적용되지 않습니다.

### CrowdHuman과 WIDER FACE 적용

기존 GUI에서 **ByteTrack → 시작**을 누르면 기본 YOLO11n과 ByteTrack으로 사람 ID·이동 궤적을 표시하고, `models/face.pt`로 검출한 얼굴을 모자이크 처리합니다. 압축을 풀 때 `models` 폴더도 복사하세요. 이전에 별도로 복사한 `models/person.pt`나 `models/coco.pt`가 있다면 그 가중치가 사람 검출에 우선 사용됩니다.

| 포함된 가중치 | 학습 / 검증 이미지 | 설정 | 검증 mAP50–95 |
|---|---:|---|---:|
| `experiments/crowdhuman_person.pt` | 96 / 32 | CrowdHuman train01·val, 4에폭, 크기 256 | 0.154 |
| `models/face.pt` | 96 / 32 | WIDER FACE train·val, 4에폭, 크기 384 | 0.197 |

CrowdHuman은 [공식 데이터](https://www.crowdhuman.org/download.html)의 [미러](https://huggingface.co/datasets/Funtion/CrowdHuman)에서 train01·val 이미지와 라벨을 가져왔습니다. WIDER FACE는 [CUHK 배포처](https://huggingface.co/datasets/CUHK-CSE/wider_face/tree/main/data)의 공식 train·val 이미지와 라벨을 사용했습니다. 두 모델 모두 배치 8, seed 42, AdamW, 초기 학습률 0.0005, 첫 10개 레이어 고정, 모자이크 증강을 끈 설정으로 추가 학습했습니다. CrowdHuman은 기본 YOLO11n에서 COCO 클래스 헤드를 유지해 시작했고, WIDER FACE는 [akanametov/yolo-face의 YOLO11n 얼굴 검출 가중치](https://github.com/akanametov/yolo-face/releases/tag/1.0.0)에서 시작했습니다. 얼굴 사전 학습 자료와 이번 검증 이미지의 독립성은 확인하지 않았습니다.

공식 train과 val에서 따로 추출한 이미지 파일명은 `experiments/training_samples.txt`에서 확인할 수 있습니다. 원본 이미지와 라벨은 ZIP에 포함하지 않았습니다. 전체 데이터로 학습하려면 아래 절차를 사용하세요.

MOT17-02-FRCNN의 처음 60프레임에서 기본 YOLO11n의 HOTA는 **38.40%**, CrowdHuman 추가 학습 모델은 **10.29%**였습니다. 이 구간에서는 CrowdHuman 모델의 오검출이 많아, 기본 사람 검출 모델을 유지했습니다. CrowdHuman과 얼굴 모델을 함께 사용하려면 아래 한 줄을 실행하고 추적을 다시 시작하세요.

```powershell
Copy-Item .\experiments\crowdhuman_person.pt .\models\person.pt
```

기본 사람 검출로 돌아가려면 `models/person.pt`의 이름을 바꾸세요. 모자이크를 끄려면 `models/face.pt`의 이름을 바꾸세요. 원본 재생 모드에서는 얼굴 모자이크가 적용되지 않습니다.

### 포함된 실험 가중치

`experiments/coco128_1epoch.pt`는 사전 학습된 YOLO11n을 COCO128 이미지 96장으로 추가 학습한 결과입니다. 나머지 32장을 검증용으로 분리했으며, 원본 사전 학습 데이터에 포함될 수 있는 이미지이므로 독립 평가 결과로 해석하지 않습니다. CPU에서 1에폭, 이미지 크기 256, 배치 8로 실험했습니다.

| 가중치 | 검증 전체 mAP50–95 | 검증 사람 mAP50–95 |
|---|---:|---:|
| 기본 YOLO11n | 0.385 | 0.369 |
| COCO128 1에폭 추가 학습 | 0.359 | 0.350 |

이 이전 실험에서 기본 YOLO11n의 검출 점수가 COCO128 추가 학습본보다 높았습니다. 실험 파일을 GUI에 적용하려면 `models/coco.pt`로 복사하세요. `models/person.pt`가 있으면 그 파일이 우선합니다.

```powershell
New-Item -ItemType Directory -Force .\models
Copy-Item .\experiments\coco128_1epoch.pt .\models\coco.pt
```

## 전체 데이터로 학습하기

PowerShell에서 프로젝트 폴더를 열어 아래 명령을 실행하세요. `--device cpu`는 기본값입니다. CUDA 지원 환경에서는 `--device 0`을 지정할 수 있습니다. 각 학습이 끝나면 터미널에 출력된 `best.pt`를 위 표의 경로에 복사하세요.

### COCO

```powershell
& ".\.venv\Scripts\python.exe" .\train_models.py coco --data coco.yaml --weights yolo11n.pt --epochs 30
```

`coco.yaml`은 Ultralytics의 COCO 2017 설정이며 필요한 파일이 없으면 다운로드를 시도합니다. 다운로드 후 이미 준비한 데이터가 있다면 해당 데이터의 YAML 경로를 `--data`로 지정하세요.

### CrowdHuman

공식 페이지에서 `CrowdHuman_train01.zip`~`train03.zip`의 이미지를 같은 `crowd/train/Images` 폴더에 모으고, `CrowdHuman_val.zip`은 별도의 `crowd/val/Images`로 풉니다. `annotation_train.odgt`와 `annotation_val.odgt`는 `crowd`에 둡니다.

```powershell
& ".\.venv\Scripts\python.exe" .\prepare_dataset.py crowd --train-images crowd/train/Images --train-labels crowd/annotation_train.odgt --val-images crowd/val/Images --val-labels crowd/annotation_val.odgt --output datasets/crowd
& ".\.venv\Scripts\python.exe" .\train_models.py crowd --data datasets/crowd/dataset.yaml --weights models/coco.pt --epochs 30
```

COCO 추가 학습본을 준비하지 않았다면 CrowdHuman 학습 명령의 `--weights`를 `yolo11n.pt`로 지정하세요.

### WIDER FACE

`WIDER_train.zip`, `WIDER_val.zip`, `wider_face_split.zip`을 받아 아래 명령의 경로가 되도록 `wider`에 풉니다.

```powershell
& ".\.venv\Scripts\python.exe" .\prepare_dataset.py face --train-images wider/WIDER_train/images --train-labels wider/wider_face_split/wider_face_train_bbx_gt.txt --val-images wider/WIDER_val/images --val-labels wider/wider_face_split/wider_face_val_bbx_gt.txt --output datasets/face
& ".\.venv\Scripts\python.exe" .\train_models.py face --data datasets/face/dataset.yaml --weights yolo11n.pt --epochs 30
```

## MOT17 / MOT20 평가

공식 ZIP의 `train` 안에서 `img1`, `seqinfo.ini`, `gt/gt.txt`가 있는 시퀀스를 사용합니다. 동일한 시퀀스와 ByteTrack 설정에서 가중치만 바꿔 비교할 수 있습니다. 다음은 MOT17 예시입니다.

```powershell
& ".\.venv\Scripts\python.exe" .\export_mot.py datasets/MOT17/train/MOT17-02-FRCNN --weights yolo11n.pt --output mot_results/baseline/data/MOT17-02-FRCNN.txt
& ".\.venv\Scripts\python.exe" .\export_mot.py datasets/MOT17/train/MOT17-02-FRCNN --weights models/person.pt --output mot_results/crowd/data/MOT17-02-FRCNN.txt
```

[TrackEval](https://github.com/JonathonLuiten/TrackEval)을 `TrackEval` 폴더에 준비하고 해당 프로젝트 안내대로 평가 환경의 의존성을 설치합니다. 아래 명령은 TrackEval의 HOTA, CLEAR(MOTA, IDSW), Identity(IDF1) 지표를 출력하고 각 추적 결과 폴더에 요약을 저장합니다.

```powershell
& ".\.venv\Scripts\python.exe" .\evaluate_mot.py --benchmark MOT17 --ground-truth datasets/MOT17/train --results mot_results --trackers baseline crowd --sequences MOT17-02-FRCNN
```

MOT20에서는 벤치마크 이름, 데이터 경로, 시퀀스 이름과 결과 폴더를 MOT20에 맞게 바꿉니다. 같은 MOT17 영상의 DPM/FRCNN/SDP 버전을 합산하면 동일 영상이 중복되므로 비교에는 한 버전만 사용하세요. 전체 데이터 학습 및 전체 MOT 벤치마크 점수는 이 ZIP에 포함되지 않습니다.

### 60프레임 추적 실험

각 시퀀스의 공식 raw MP4에서 첫 60프레임을 읽고 정답 해상도로 맞춘 후 공식 train 라벨의 해당 구간과 비교했습니다. 두 검출기에 같은 영상과 ByteTrack 설정을 적용했습니다. HOTA·IDF1·MOTA는 %입니다. 이 값은 짧은 구간의 실행 검증 결과입니다.

| 시퀀스 | 가중치 | HOTA | IDF1 | MOTA | IDSW |
|---|---|---:|---:|---:|---:|
| MOT17-02-FRCNN | 기본 YOLO11n | 38.40 | 38.23 | 15.70 | 5 |
| MOT17-02-FRCNN | COCO128 추가 학습 | 26.80 | 25.45 | 13.85 | 0 |
| MOT20-01 | 기본 YOLO11n | 41.40 | 47.91 | 33.03 | 3 |
| MOT20-01 | COCO128 추가 학습 | 21.65 | 20.86 | 11.22 | 6 |
