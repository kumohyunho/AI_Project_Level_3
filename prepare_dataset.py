"""CrowdHuman과 WIDER FACE의 공식 라벨을 YOLO 검출 형식으로 변환합니다."""
import argparse
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory


def crowd_records(annotation):
    with annotation.open(encoding="utf-8-sig") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            boxes = [
                entry["fbox"] for entry in record["gtboxes"]
                if entry["tag"] == "person" and not entry.get("extra", {}).get("ignore")
            ]
            yield Path(record["ID"] + ".jpg"), boxes


def wider_records(annotation):
    lines = annotation.read_text(encoding="utf-8-sig").splitlines()
    index = 0
    while index < len(lines):
        name = lines[index].strip()
        index += 1
        if not name:
            continue
        count = int(lines[index].strip())
        index += 1
        boxes = []
        for _ in range(count):
            fields = [int(value) for value in lines[index].split()]
            index += 1
            if len(fields) < 10:
                raise ValueError(f"잘못된 WIDER FACE 박스: {name}")
            if fields[7] == 0:  # invalid 플래그
                boxes.append(fields[:4])
        if count == 0 and index < len(lines):
            dummy = lines[index].split()
            if len(dummy) == 10 and all(value == "0" for value in dummy):
                index += 1
        yield Path(name), boxes


def normalize(box, width, height):
    x, y, w, h = box
    left, top = max(0, x), max(0, y)
    right, bottom = min(width, x + w), min(height, y + h)
    if right <= left or bottom <= top:
        return None
    return ((left + right) / (2 * width), (top + bottom) / (2 * height),
            (right - left) / width, (bottom - top) / height)


def convert_split(records, source, destination, split):
    from PIL import Image

    images = destination / "images" / split
    labels = destination / "labels" / split
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    image_count = 0
    box_count = 0
    for name, boxes in records:
        if name.is_absolute() or ".." in name.parts or name.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise ValueError(f"잘못된 이미지 경로: {name}")
        image_path, output = source / name, images / name
        if output.exists():
            raise ValueError(f"중복 이미지: {name}")
        with Image.open(image_path) as image:
            width, height = image.size
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(image_path, output)
        except OSError:
            shutil.copyfile(image_path, output)
        label = (labels / name).with_suffix(".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        values = [normalize(box, width, height) for box in boxes]
        rows = ["0 " + " ".join(f"{value:.8f}" for value in box) + "\n"
                for box in values if box is not None]
        label.write_text("".join(rows), encoding="utf-8")
        image_count += 1
        box_count += len(rows)
    if not image_count or not box_count:
        raise ValueError(f"{split}: 이미지 또는 유효한 정답 박스가 없습니다.")
    print(f"{split}: {image_count} images, {box_count} boxes")


def prepare(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"출력 경로가 이미 있습니다: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    records = crowd_records if args.kind == "crowd" else wider_records
    name = "person" if args.kind == "crowd" else "face"
    with TemporaryDirectory(dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        for split in ("train", "val"):
            convert_split(records(getattr(args, split + "_labels")),
                          getattr(args, split + "_images"), staging, split)
        (staging / "dataset.yaml").write_text(
            f"path: {json.dumps(output.as_posix(), ensure_ascii=False)}\n"
            f"train: images/train\nval: images/val\nnames: {{0: {name}}}\n", encoding="utf-8"
        )
        staging.rename(output)
    print(output / "dataset.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["crowd", "face"])
    for split in ("train", "val"):
        parser.add_argument(f"--{split}-images", type=Path, required=True)
        parser.add_argument(f"--{split}-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())
