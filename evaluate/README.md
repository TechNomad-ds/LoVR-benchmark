# LoVR Evaluation Code

This directory provides the official evaluation pipeline for **LoVR**, a long-video retrieval benchmark.
It supports multiple video–text retrieval backbones (e.g. CLIP, SigLIP, VideoCLIP-XL, LanguageBind) and
computes **text-to-video** and **video-to-text** retrieval metrics, including **pass@k**.

Pre-trained weights for the supported models are available at  
**ModelScope**: `thirstylearning/lovr_models`  
Download and extract them under `evaluate/models/` so that each model (e.g. `CLIP`, `LanguageBind`) resides
in its own subdirectory.

---

## Evaluation Pipeline Overview

All models are evaluated using the same **four-stage pipeline**:

1. **Encode text** – encode all clip-level and full-video captions
2. **Encode video** – sample frames and encode visual features per video
3. **Merge video** – merge per-video results into a single feature file
4. **Calc pass** – compute similarities and report pass@k metrics

Each stage is controlled by a CLI flag and can be run independently, enabling easy resumption and
multi-GPU parallelism.

---

## Evaluating a Custom Model

To evaluate your own model on LoVR, follow the steps below.

### 1. Create a model directory

Create a new subfolder under `evaluate/models/` named after your model:

```bash
mkdir -p evaluate/models/MyModel
````

This directory will contain your evaluation script and (optionally) model weights.

---

### 2. Implement `<RUN_NAME>_test.py`

Inside `evaluate/models/MyModel/`, create a script named:

```text
MyModel_test.py
```

Your script must implement the following **four pipeline stages**.

| Stage            | CLI flag                                                   | Description                                                                                                                                                                                                                                     |
| ---------------- | ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Encode text**  | `--encode_text`                                            | Read `caption_data/all_clip.jsonl` and `all_video.jsonl`, encode all captions, and save results to `$EXP_DIR/$RUN_NAME/text.pt`. The file must contain:<br>• `feat`: L2-normalized text features<br>• `source`: caption IDs aligned with `feat` |
| **Encode video** | `--encode_video --num_chunks N --chunk_idx i --interval K` | Sample frames for each video and encode visual features. Save per-video outputs to `$EXP_DIR/$RUN_NAME/results/<video_id>.pt`. Each file must contain:<br>• `image_feats` (L2-normalized)<br>• `clip_src`<br>• `full_src`                       |
| **Merge video**  | `--video_merge`                                            | Merge all per-video `.pt` files into a single `$EXP_DIR/$RUN_NAME/video.pt`.                                                                                                                                                                    |
| **Calc pass**    | `--calc_pass --topk 1,5,10`                                | Load `text.pt` and `video.pt`, compute similarities, and report pass@k for both clip-level and full-video retrieval.                                                                                                                            |

---

### Text–Video ID Alignment (Important)

The evaluation relies on a strict alignment convention:

* In `text.pt`, entries in `source` are ordered as:

  1. **Clip-level captions**, corresponding to `all_clip.jsonl`
  2. **Full-video captions**, corresponding to `all_video.jsonl`

* Clip caption IDs are derived from the `path` field in `all_clip.jsonl`
  (directory name and `.mp4` suffix removed).

* Full-video caption IDs correspond to the `vid` field in `all_video.jsonl`.

During `calc_pass`, the constant:

```python
CLIP_NUM = number of clip captions
```

is used to split clip-level and full-video retrieval.
This value **must equal** the number of lines in `all_clip.jsonl`.

---

### Path Conventions

All paths must be derived from environment variables—do **not** hardcode absolute paths.

* `DATA_ROOT`: LoVR data root
  (must contain `video_data/long_video_clip/` and `caption_data/`)
* `EXP_DIR`: experiment output root (default: `exp/cache`)
* `RUN_NAME`: model name used in output paths

Example:

```python
TEXT_RESULT_PATH = os.path.join(EXP_DIR, RUN_NAME, "text.pt")
VIDEO_RESULT_DIR = os.path.join(EXP_DIR, RUN_NAME, "results")
```

---

### Recommended Templates

We strongly recommend copying an existing implementation and modifying only model-specific parts:

* `models/CLIP/CLIP_test.py`
* `models/siglip-base-patch16-224/siglip-base-patch16-224_test.py`

Typically, you only need to replace:

* model loading (`load_model`)
* text encoding
* image preprocessing and encoding

Keep the shared logic for:

* video chunking (`make_video_dict`, `get_video_list`, `get_chunk`)
* `video_merge`
* `calc_pass`
* data formats

---

### 3. (Optional) Add a launch script

To enable multi-GPU video encoding, you may add a launch script under `evaluate/`,
for example `run_MyModel.sh`.

(脚本内容与你原版一致，这里不重复贴，保持不变即可)

---

### 4. Run the evaluation

From the `evaluate` directory:

```bash
export DATA_ROOT=/path/to/LoVR/data
export EXP_DIR=exp/cache

cd evaluate
cd models/MyModel

# 1) Encode text
python MyModel_test.py --encode_text

# 2) Encode video
python MyModel_test.py --encode_video --num_chunks 1 --chunk_idx 0 --interval 10

# 3) Merge video features
python MyModel_test.py --video_merge

# 4) Compute pass@k
python MyModel_test.py --calc_pass --topk 1,5,10
```

---

## Utilities and Scripts

* `utils.py` – shared helpers for video I/O and chunking
* `check_left.py` – lists videos without results for resumable runs
* `read_pt.py` – inspect `.pt` files for debugging
* `run_*.sh` – launch scripts for multi-GPU encoding

---

## Notes

* **Model weights**: place under `evaluate/models/<RUN_NAME>/` or specify via `MODEL_DIR`.
* **Resuming**: use `check_left.py` to identify missing videos.
* **Optional features**: if `text.pt` contains `theme_feat` / `theme_src`, additional metrics may be reported.

---

## Requirements

* Python 3
* PyTorch
* decord, pandas, tqdm
* model-specific dependencies (e.g. `transformers`, `clip`)
* CUDA-capable GPU