"""
List videos that have not been processed yet (no corresponding .pt in results).
Paths are configurable via env DATA_ROOT, EXP_DIR, RUN_NAME.
"""
import os
import pandas as pd
import json

DATA_ROOT = os.environ.get("DATA_ROOT", ".")
EXP_DIR = os.environ.get("EXP_DIR", "exp/cache")
RUN_NAME = os.environ.get("RUN_NAME", "CLIP")

VIDEO_CAP_PATH = os.path.join(DATA_ROOT, "caption_data", "all_video.jsonl")
VIDEO_CAP_PATH_LEFT = os.path.join(DATA_ROOT, "caption_data", f"{RUN_NAME}_left_video.jsonl")
RESULTS_DIR = os.path.join(EXP_DIR, RUN_NAME, "results")


def main():
    full = pd.read_json(VIDEO_CAP_PATH, lines=True)
    all_videos = full["vid"].astype(str).tolist()

    done = {os.path.splitext(f)[0] for f in os.listdir(RESULTS_DIR)}

    left = full[~full["vid"].astype(str).isin(done)]

    with open(VIDEO_CAP_PATH_LEFT, "w") as f:
        for rec in left.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Total: {len(all_videos)}, Done: {len(done)}, Left: {len(left)}")
    print(f"Left list written to {VIDEO_CAP_PATH_LEFT}")


if __name__ == "__main__":
    main()
