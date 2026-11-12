"""TrackEval에서 MOT17/MOT20의 HOTA, MOTA, IDF1, IDSW를 계산합니다."""
import argparse
from pathlib import Path
import sys


def evaluate(args):
    import numpy as np

    checkout = args.trackeval.resolve()
    if not (checkout / "trackeval" / "__init__.py").is_file():
        raise FileNotFoundError(f"TrackEval 저장소 경로를 확인하세요: {checkout}")
    sys.path.insert(0, str(checkout))
    # 일부 TrackEval 버전에서 쓰는 NumPy 별칭을 이 평가 프로세스에서 복원합니다.
    for name, scalar in (("float", float), ("int", int)):
        if name not in vars(np):
            setattr(np, name, scalar)
    import trackeval

    evaluator = trackeval.Evaluator({"PLOT_CURVES": False})
    dataset = trackeval.datasets.MotChallenge2DBox({
        "GT_FOLDER": str(args.ground_truth.resolve()),
        "TRACKERS_FOLDER": str(args.results.resolve()),
        "TRACKERS_TO_EVAL": args.trackers,
        "BENCHMARK": args.benchmark,
        "SPLIT_TO_EVAL": "train",
        "SKIP_SPLIT_FOL": True,
        "SEQ_INFO": dict.fromkeys(args.sequences),
    })
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    evaluator.evaluate([dataset], metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trackeval", type=Path, default=Path("TrackEval"))
    parser.add_argument("--benchmark", choices=["MOT17", "MOT20"], required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--trackers", nargs="+", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    evaluate(parser.parse_args())
