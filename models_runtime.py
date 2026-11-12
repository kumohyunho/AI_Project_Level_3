"""GUI와 평가 도구에서 같은 검출 및 추적 설정을 사용합니다."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"


def person_weights():
    for candidate in (MODEL_DIR / "person.pt", MODEL_DIR / "coco.pt", ROOT / "yolo11n.pt"):
        if candidate.is_file():
            return str(candidate)
    return "yolo11n.pt"


def person_class(model):
    for class_id, name in model.names.items():
        if name.casefold() == "person":
            return int(class_id)
    raise ValueError("검출 가중치에 person 클래스가 없습니다.")


def track_people(model, frame, class_id, device="cpu"):
    return model.track(frame, persist=True, tracker="bytetrack.yaml",
                       classes=[class_id], conf=0.1, device=device, verbose=False)[0].boxes


def mosaic_faces(frame, model):
    import cv2

    boxes = model.predict(frame, imgsz=640, conf=0.2, device="cpu", verbose=False)[0].boxes
    height, width = frame.shape[:2]
    for x1, y1, x2, y2 in boxes.xyxy.int().cpu().tolist():
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        region = frame[y1:y2, x1:x2]
        small = cv2.resize(region, (max(1, (x2-x1)//14), max(1, (y2-y1)//14)),
                           interpolation=cv2.INTER_AREA)
        region[:] = cv2.resize(small, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
