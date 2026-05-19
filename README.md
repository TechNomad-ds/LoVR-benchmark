# LoVR: A Benchmark for Long Video Retrieval in Multimodal Contexts

<p align="center">
  <a href="https://arxiv.org/abs/2505.13928"><img src="https://img.shields.io/badge/arXiv-2505.13928-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/debugger123/LoVR-benchmark"><img src="https://img.shields.io/badge/Dataset-HuggingFace-yellow.svg" alt="Dataset"></a>
</p>

<p align="center">
  <b>Official repository for the paper<br>
  <i>LoVR: A Benchmark for Long Video Retrieval in Multimodal Contexts</i></b>
</p>

LoVR is a large-scale benchmark for evaluating long video--text retrieval in realistic multimodal scenarios. It is designed to assess whether retrieval models can understand fine-grained temporal content, aggregate clip-level semantics, and retrieve long videos or relevant clips from complex natural-language queries.

The benchmark contains **467 long videos** and **40,804 fine-grained clips**, with detailed annotations supporting both **video-level retrieval** and **clip-level retrieval**.

---

## Overview

<p align="center">
  <img src="assets/lovr_overview.png" width="90%" alt="Overview of the LoVR benchmark">
</p>

<p align="center">
  <i>Figure 1. Overview of LoVR. The benchmark targets long video retrieval under multimodal contexts and supports both video-level and clip-level retrieval evaluation.</i>
</p>

LoVR is motivated by the gap between existing short-video retrieval benchmarks and real-world long-video search scenarios. In practical applications, users often search for events, actions, scenes, or semantic moments that are distributed across long videos. This requires models to perform both local fine-grained understanding and global semantic aggregation.

LoVR provides:

* A long-video retrieval benchmark with fine-grained clip annotations.
* A reproducible data construction pipeline from raw videos to structured captions.
* Evaluation protocols for video-level and clip-level retrieval.
* Baseline implementations for representative vision-language and video-language models.

---

## Benchmark Statistics

| Item                            |                                                   Number |
| ------------------------------- | -------------------------------------------------------: |
| Long videos                     |                                                      467 |
| Fine-grained clips              |                                                   40,804 |
| Supported retrieval granularity |                                 Video-level / Clip-level |
| Main evaluation settings        | Text-to-Video, Video-to-Text, Text-to-Clip, Clip-to-Text |

---

## Dataset

The LoVR dataset is available at:

```text
https://huggingface.co/datasets/debugger123/LoVR-benchmark
```

The dataset includes long videos, segmented clips, clip-level captions, and aggregated video-level descriptions. These resources can be used to evaluate retrieval models under different granularity settings.

A typical data record contains:

```json
{
  "video_id": "example_video_id",
  "clip_id": "example_clip_id",
  "video_path": "path/to/video.mp4",
  "clip_path": "path/to/clip.mp4",
  "start_time": 12.4,
  "end_time": 25.7,
  "clip_caption": "A detailed caption describing the visual content of the clip.",
  "video_caption": "An aggregated description summarizing the long video."
}
```

Please refer to the dataset page for the complete file structure and metadata format.

---

## Data Construction Pipeline

<p align="center">
  <img src="assets/data_construction_pipeline.png" width="90%" alt="Data construction pipeline of LoVR">
</p>

<p align="center">
  <i>Figure 2. Data construction pipeline. LoVR is constructed through clip segmentation, clip-level caption generation, and video-level caption aggregation.</i>
</p>

The LoVR dataset is constructed using a three-stage pipeline:

1. **Clip Segmentation**
   Long videos are segmented into fine-grained clips according to visual scene changes.

2. **Clip-level Caption Generation**
   A Vision-Language Model is used to generate detailed natural-language captions for each segmented clip.

3. **Video-level Caption Aggregation**
   Clip-level captions are merged into coherent long-video descriptions, enabling video-level retrieval evaluation.

All data construction scripts are provided in the `data_generation/` directory.

---

## Repository Structure

```text
LoVR/
├── assets/
│   ├── lovr_overview.png
│   └── data_construction_pipeline.png
│
├── data_generation/
│   ├── clip_segmentation.py
│   ├── caption_generator.py
│   └── caption_merger.py
│
├── evaluate/
│   ├── models/
│   ├── run_*.sh
│   └── README.md
│
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-org/LoVR.git
cd LoVR
```

Create the environment:

```bash
conda create -n lovr python=3.10 -y
conda activate lovr
pip install -r requirements.txt
```

> Note: Please install model-specific dependencies according to the baseline model you intend to evaluate. Some models may require additional packages or customized inference environments.

---

## Generating the Dataset

The dataset generation pipeline should be executed in the following order:

```text
Clip Segmentation → Caption Generation → Caption Merging
```

All scripts are located in:

```text
data_generation/
```

### 1. Clip Segmentation

Script:

```text
data_generation/clip_segmentation.py
```

This script segments long videos into fine-grained clips based on visual scene changes.

| Parameter        | Description                               |
| ---------------- | ----------------------------------------- |
| `--input_folder` | Directory containing original long videos |
| `--output_dir`   | Output directory for generated clips      |
| `--max_workers`  | Number of parallel workers                |

Example:

```bash
cd data_generation

python clip_segmentation.py \
  --input_folder /path/to/videos \
  --output_dir /path/to/output/clips \
  --max_workers 50
```

### 2. Clip-level Caption Generation

Script:

```text
data_generation/caption_generator.py
```

This script generates detailed captions for segmented clips using a Vision-Language Model.

| Parameter        | Description                         |
| ---------------- | ----------------------------------- |
| `--model-path`   | Path to the model checkpoint        |
| `--video-folder` | Directory containing video clips    |
| `--jsonl-file`   | JSONL file containing clip metadata |
| `--result-file`  | Output caption file                 |
| `--batch-size`   | Inference batch size                |
| `--num-chunks`   | Number of task chunks               |
| `--chunk-idx`    | Current chunk index                 |
| `--rerun`        | Whether to rerun existing results   |
| `--debug`        | Whether to enable debug mode        |

Example:

```bash
cd data_generation

export CKPT=/path/to/model_weights
CHUNKS=8
IDX=0
LOG_FILE=output_log_${IDX}.log

python caption_generator.py \
  --model-path ${CKPT} \
  --video-folder /path/to/clips \
  --jsonl-file /path/to/clip_metadata.jsonl \
  --result-file /path/to/caption_results_${IDX}.jsonl \
  --batch-size 16 \
  --num-chunks ${CHUNKS} \
  --chunk-idx ${IDX} \
  > "$LOG_FILE" 2>&1 &
```

This script supports chunked processing, enabling distributed inference across multiple GPUs or machines.

### 3. Video-level Caption Aggregation

Script:

```text
data_generation/caption_merger.py
```

This script merges clip-level captions into final video-level descriptions. It supports resume functionality, so previously processed videos are skipped automatically.

| Parameter       | Description              |
| --------------- | ------------------------ |
| `--cap-file`    | Input caption JSONL file |
| `--result-file` | Output merged JSONL file |
| `--num-workers` | Number of workers        |

Example:

```bash
cd data_generation

python caption_merger.py \
  --cap-file /path/to/caption_data.jsonl \
  --result-file /path/to/final_output.jsonl \
  --num-workers 50
```

---

## Evaluation

The evaluation code is provided in:

```text
evaluate/
```

LoVR supports four retrieval settings:

| Setting                 | Query             | Target            |
| ----------------------- | ----------------- | ----------------- |
| Text-to-Video Retrieval | Text              | Long video        |
| Video-to-Text Retrieval | Long video        | Text              |
| Text-to-Clip Retrieval  | Text              | Fine-grained clip |
| Clip-to-Text Retrieval  | Fine-grained clip | Text              |

The evaluation scripts are provided as shell scripts:

```text
evaluate/run_*.sh
```

Please see `evaluate/README.md` for detailed instructions on running each baseline.

---

## Baseline Models

The repository includes evaluation implementations for representative multimodal retrieval models:

* CLIP
* SigLIP
* VideoCLIP-XL
* LanguageBind
* MM-Embed

Pretrained model weights are available via ModelScope:

```text
thirstylearning/lovr_models
```

Download the model weights and place them under:

```text
evaluate/models/
```

Each model should be placed in its own subdirectory.

---

## Citation

If you find LoVR useful for your research, please cite our paper:

```bibtex
@article{cai2025lovr,
  title={LoVR: A Benchmark for Long Video Retrieval in Multimodal Contexts},
  author={Cai, Qifeng and Liang, Hao and Han, Zhaoyang and Dong, Hejun and Qiang, Meiyi and An, Ruichuan and Xu, Quanqing and Cui, Bin and Zhang, Wentao},
  journal={arXiv preprint arXiv:2505.13928},
  year={2025}
}
```
