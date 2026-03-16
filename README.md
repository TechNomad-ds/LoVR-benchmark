# LoVR Benchmark

Official repository for **LoVR: A Benchmark for Long Video Retrieval in Multimodal Contexts**.

LoVR is a large-scale benchmark designed to evaluate **long video–text retrieval** in realistic multimodal scenarios.
The benchmark contains **467 long videos and 40,804 fine-grained clips**, each paired with detailed captions for both **clip-level** and **video-level retrieval tasks**.

The repository provides:

* Dataset generation pipeline
* Caption generation framework
* Benchmark evaluation scripts
* Model evaluation implementations

---

# Pipeline Overview

The LoVR dataset is constructed using a **three-stage pipeline**:

1. **Clip Segmentation**
   Segment long videos into fine-grained clips based on visual scene changes.

2. **Caption Generation**
   Generate detailed captions for each clip using Vision-Language Models (VLMs).

3. **Caption Aggregation**
   Merge clip-level captions into coherent long-video descriptions.

The implementation of these steps is provided in the `data_generation/` directory.

---

# Project Structure

```
LoVR/
│
├── data_generation/
│   ├── clip_segmentation.py
│   ├── caption_generator.py
│   ├── caption_merger.py
│
├── evaluate/
│   ├── models/
│   ├── run_*.sh
│   └── README.md
│
└── README.md
```

---

# Dataset Generation

The dataset generation pipeline consists of **three steps**.
Please execute them **in the following order**.

```
Clip Segmentation → Caption Generation → Caption Merging
```

All scripts are located in:

```
data_generation/
```

---

# 1. Clip Segmentation

Script:

```
clip_segmentation.py
```

This script segments long videos into clips based on visual changes.

### Parameters

| Parameter        | Description                          |
| ---------------- | ------------------------------------ |
| `--input_folder` | Directory containing original videos |
| `--output_dir`   | Output directory for generated clips |
| `--max_workers`  | Number of parallel workers           |

### Example

```bash
cd data_generation

python clip_segmentation.py \
    --input_folder /path/to/videos \
    --output_dir /path/to/output/clips \
    --max_workers 50
```

---

# 2. Caption Generation

Script:

```
caption_generator.py
```

This script generates captions for the segmented clips using a **Vision-Language Model (VLM)**.

### Parameters

| Parameter        | Description                         |
| ---------------- | ----------------------------------- |
| `--model-path`   | Path to model checkpoint            |
| `--video-folder` | Directory containing video clips    |
| `--jsonl-file`   | JSONL file containing clip metadata |
| `--result-file`  | Output caption file                 |
| `--batch-size`   | Inference batch size                |
| `--num-chunks`   | Number of task chunks               |
| `--chunk-idx`    | Current chunk index                 |
| `--rerun`        | Rerun existing results              |
| `--debug`        | Debug mode                          |

### Example

```bash
cd data_generation

export CKPT=/path/to/model_weights
CHUNKS=8
IDX=0
LOG_FILE=output_log_${IDX}.log

python caption_generator.py \
    --model-path ${CKPT} \
    --video-folder your/video/folder \
    --jsonl-file your/jsonl/file \
    --result-file your/result/file \
    --batch-size 16 \
    --num-chunks ${CHUNKS} \
    --chunk-idx ${IDX} \
    > "$LOG_FILE" 2>&1 &
```

This script supports **chunked processing**, enabling distributed inference across multiple GPUs.

---

# 3. Caption Merging

Script:

```
caption_merger.py
```

This script merges all generated captions into a final JSONL file.

It supports **resume functionality** — previously processed videos will be skipped automatically.

### Parameters

| Parameter       | Description              |
| --------------- | ------------------------ |
| `--cap-file`    | Input caption JSONL file |
| `--result-file` | Output merged file       |
| `--num-workers` | Number of workers        |

### Example

```bash
cd data_generation

python caption_merger.py \
    --cap-file /path/to/caption_data.jsonl \
    --result-file /path/to/final_output.jsonl \
    --num-workers 50
```

---

# Evaluation

The evaluation pipeline is located in:

```
evaluate/
```

It supports evaluation for:

* **Text-to-Video Retrieval**
* **Video-to-Text Retrieval**
* **Text-to-Clip Retrieval**
* **Clip-to-Text Retrieval**

### Directory Description

| Directory            | Description                      |
| -------------------- | -------------------------------- |
| `evaluate/models/`   | Model-specific implementations   |
| `evaluate/run_*.sh`  | Evaluation scripts               |
| `evaluate/README.md` | Detailed evaluation instructions |

Supported baseline models include:

* CLIP
* SigLIP
* VideoCLIP-XL
* LanguageBind
* MM-Embed

---

# Model Weights

Pretrained model weights are available via **ModelScope**:

```
thirstylearning/lovr_models
```

Download the weights and place them under:

```
evaluate/models/
```

Each model should reside in its own subdirectory.

---

# Responsible Use

The LoVR dataset is intended for **academic research and educational purposes only**.

Users must not use this dataset to develop systems that are:

* harmful
* discriminatory
* privacy-invasive

Please ensure compliance with **ethical AI research guidelines**.

---

# Citation

If you find **LoVR** useful in your research, please cite:

```bibtex
@article{cai2025lovr,
  title={LoVR: A Benchmark for Long Video Retrieval in Multimodal Contexts},
  author={Cai, Qifeng and Liang, Hao and Han, Zhaoyang and Dong, Hejun and Qiang, Meiyi and An, Ruichuan and Xu, Quanqing and Cui, Bin and Zhang, Wentao},
  journal={arXiv preprint arXiv:2505.13928},
  year={2025}
}
```
