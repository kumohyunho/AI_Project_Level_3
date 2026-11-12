"""YOLO11n 검출기 학습. ByteTrack은 별도 학습 없이 검출을 연결합니다."""
import argparse
from pathlib import Path


def train(args):
    from ultralytics import YOLO

    names = {"coco": "coco.pt", "crowd": "person.pt", "face": "face.pt"}
    model = YOLO(args.weights)
    options = {
        "data": args.data, "epochs": args.epochs, "imgsz": args.imgsz,
        "batch": args.batch, "device": args.device, "workers": 0,
        "project": str(Path(args.runs).resolve()), "name": args.kind,
        "seed": 42,
    }
    model.train(**options)
    best = Path(model.trainer.best)
    if not best.is_file():
        raise RuntimeError("학습에서 best.pt가 생성되지 않았습니다.")
    print(f"학습 완료: {best.resolve()}")
    print(f"GUI에서 사용하려면 이 파일을 models/{names[args.kind]}로 복사하세요.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["coco", "crowd", "face"])
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="cpu 또는 CUDA 장치 번호")
    parser.add_argument("--runs", default="training_runs")
    train(parser.parse_args())
