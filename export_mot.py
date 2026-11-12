"""MOT17/MOT20 시퀀스의 ByteTrack 결과를 TrackEval 입력 형식으로 내보냅니다."""
import argparse
import configparser
import csv
from pathlib import Path
from tempfile import NamedTemporaryFile

from models_runtime import person_class, person_weights, track_people


def export(args):
    import cv2
    import numpy as np
    from ultralytics import YOLO

    sequence = args.sequence.resolve()
    metadata = configparser.ConfigParser()
    with (sequence / "seqinfo.ini").open(encoding="utf-8-sig") as source:
        metadata.read_file(source)
    info = metadata["Sequence"]
    frame_count = info.getint("seqLength")
    if frame_count < 1:
        raise ValueError("시퀀스에 프레임이 없습니다.")
    image_dir = sequence / info["imDir"]
    extension = info["imExt"]

    model = YOLO(args.weights or person_weights())
    class_id = person_class(model)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                dir=output.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            writer = csv.writer(temporary, lineterminator="\n")
            for number in range(1, frame_count + 1):
                path = image_dir / f"{number:06d}{extension}"
                frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError(f"프레임을 읽을 수 없습니다: {path}")
                boxes = track_people(model, frame, class_id, args.device)
                if boxes is None or boxes.id is None:
                    continue
                for track_id, box, confidence in zip(boxes.id.int().cpu().tolist(),
                                                      boxes.xyxy.cpu().tolist(),
                                                      boxes.conf.cpu().tolist()):
                    x1, y1, x2, y2 = box
                    writer.writerow((number, track_id, f"{x1+1:.3f}", f"{y1+1:.3f}",
                                     f"{x2-x1:.3f}", f"{y2-y1:.3f}",
                                     f"{confidence:.5f}", -1, -1, -1))
        temporary_path.replace(output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    print(f"{frame_count} frames → {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", type=Path, help="seqinfo.ini와 img1이 있는 폴더")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", help="비교할 검출기 .pt")
    parser.add_argument("--device", default="cpu")
    export(parser.parse_args())
