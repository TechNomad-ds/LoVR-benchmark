import torch, argparse, os, pickle
import time
import pandas as pd
import numpy as np
from PIL import Image
from utils import *
from contextlib import contextmanager
from models.VideoCLIP_XL.modeling import VideoCLIP_XL
from models.VideoCLIP_XL.utils.text_encoder import text_encoder
from models.VideoCLIP_XL.demo import video_preprocessing_frame

RUN_NAME = "VideoCLIP-XL"
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))
WEIGHT_PATH = os.path.join(MODEL_DIR, "VideoCLIP-XL", "VideoCLIP-XL.bin")
device = "cuda" if torch.cuda.is_available() else "cpu"

TEXT_RESULT_PATH = os.path.join(EXP_DIR, RUN_NAME, "text.pt")
TEXT_ALL_RESULT_PATH = os.path.join(EXP_DIR, RUN_NAME, "text_all.pt")
VIDEO_RESULT_FOLDER = os.path.join(EXP_DIR, RUN_NAME, "results")
VIDEO_RESULT_PATH = os.path.join(EXP_DIR, RUN_NAME, "video.pt")
ERROR_LOG = os.path.join(EXP_DIR, RUN_NAME, "logs", "error.txt")
os.makedirs(VIDEO_RESULT_FOLDER, exist_ok=True)


def load_model():
    print("Loading VideoCLIP-XL model")
    model = VideoCLIP_XL()
    state_dict = torch.load(WEIGHT_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model

def encode_text(args):
    model = load_model()
    clip_path = os.path.join(DATA_ROOT, "caption_data", "all_clip.jsonl")
    full_path = os.path.join(DATA_ROOT, "caption_data", "all_video.jsonl")
    clip_df = pd.read_json(clip_path, lines=True)
    full_df = pd.read_json(full_path, lines=True)
    all_text = clip_df["cap"].to_list() + full_df["cap"].to_list()
    source = clip_df["path"].apply(lambda x: x.split("/")[-1][:-4]).to_list() + full_df["vid"].to_list()

    text = text_encoder.tokenize(all_text, truncate=True).to(device)
    BATCH_SIZE = 1024
    with torch.no_grad():
        text_features = [model.text_model.encode_text(text[i : i + BATCH_SIZE]) for i in range(0, len(text), BATCH_SIZE)]
    text_features = torch.cat(text_features)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    text_dict = {"feat": text_features, "source": source}
    torch.save(text_dict, TEXT_RESULT_PATH)

def encode_video(args):
    model = load_model()
    video_list = get_video_list()
    processed = [i[:-3] for i in os.listdir(VIDEO_RESULT_FOLDER)]
    video_list = [i + '/' for i in video_list if i not in processed]
    video_list = get_chunk(video_list, args.num_chunks, args.chunk_idx)
    
    print(f'{len(video_list)} items to be processed')
    
    pbar = tqdm(video_list)
    for video in pbar:
        pbar.set_description(f'processing {video[:-1]}')
        
        elapsed_time = {}
        @contextmanager
        def _timer(name):
            start = time.time()
            yield
            end = time.time()
            elapsed_time[name] = end - start
        
        save_path = os.path.join(VIDEO_RESULT_FOLDER, f'{video[:-1]}.pt')
        if os.path.exists(save_path):
            continue
        
        with _timer("video_dict_cost"):
            try:
                video_dict = make_video_dict(video, args.interval)
            except RuntimeError as e:
                with open(ERROR_LOG, 'a') as f:
                    f.write(f'when processing {video[:-1]} following error occurred:\n')
                    f.write(str(e))
                    f.write('\n\n')
                continue
        
        image_dict = {}
        images = video_dict['images']
        clip_src = video_dict['clip_src']
        full_src = video_dict['full_src']
        for i in range(len(images)):
            if clip_src[i] is not None:
                image_dict[clip_src[i]] = image_dict.get(clip_src[i], []) + [images[i]]
            if full_src[i] is not None:
                image_dict[full_src[i]] = image_dict.get(full_src[i], []) + [images[i]]
        
        images, clip_src, full_src = [], [], []
        for k, v in image_dict.items():
            images.append(v)
            if 'Scene' in k:
                clip_src.append(k)
                full_src.append(None)
            else:
                clip_src.append(None)
                full_src.append(k)
        
        with _timer('preprocess_cost'):
            video_inputs = torch.cat([video_preprocessing_frame(video) for video in images], 0).float().cuda()
        
        with torch.no_grad():
            with _timer("encode_image_cost"):
                video_features = model.vision_model.get_vid_features(video_inputs).float()
                video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        
        result = {
            "image_feats": video_features,
            "clip_src": clip_src,
            "full_src": full_src,
            **elapsed_time,
        }
        
        torch.save(result, save_path)

def video_merge(args):
    video_list = get_video_list()
    processed = [i[:-3] for i in os.listdir(VIDEO_RESULT_FOLDER)]
    assert all([video in processed for video in video_list]) == True
    
    result = None
    for video in tqdm(video_list, desc="merging..."):
        path = os.path.join(VIDEO_RESULT_FOLDER, f"{video}.pt")
        video_dict = torch.load(path, map_location="cpu")
        if result is None:
            result = video_dict
        else:
            result['image_feats'] = torch.cat((result['image_feats'], video_dict['image_feats']))
            result['clip_src'] += video_dict['clip_src']
            result['full_src'] += video_dict['full_src']
            result['video_dict_cost'] += video_dict['video_dict_cost']
            result['preprocess_cost'] += video_dict['preprocess_cost']
            result['encode_image_cost'] += video_dict['encode_image_cost']
    
    torch.save(result, VIDEO_RESULT_PATH)

def calc_pass(args):
    assert os.path.exists(TEXT_RESULT_PATH)
    assert os.path.exists(VIDEO_RESULT_PATH)
    text_dict = torch.load(TEXT_RESULT_PATH, map_location="cpu")
    video_dict = torch.load(VIDEO_RESULT_PATH, map_location="cpu")

    text_feat = text_dict["feat"]
    image_feat = video_dict["image_feats"]
    text_src = text_dict["source"]
    has_theme = "theme_feat" in text_dict and "theme_src" in text_dict
    if has_theme:
        text_theme_feat = text_dict["theme_feat"]
        text_theme_src = text_dict["theme_src"]
        if hasattr(text_theme_src, "to_list"):
            text_theme_src = text_theme_src.to_list()

    clip_src = video_dict["clip_src"]
    clip_feat = torch.stack([row for idx, row in enumerate(image_feat) if clip_src[idx] is not None])
    clip_src = [i for i in clip_src if i is not None]
    full_src = video_dict["full_src"]
    full_feat = torch.stack([row for idx, row in enumerate(image_feat) if full_src[idx] is not None])
    full_src = [i for i in full_src if i is not None]

    clip_1000_id = [i for i in range(len(clip_src)) if clip_src[i] in text_src]
    clip_1000_feat = torch.stack([clip_feat[i] for i in clip_1000_id])
    clip_1000_src = [clip_src[i] for i in clip_1000_id]
    CLIP_NUM = 1000

    clip_sim = text_feat[:CLIP_NUM] @ clip_feat.T
    full_sim = text_feat[CLIP_NUM:] @ full_feat.T
    v2t_clip_sim = clip_1000_feat @ text_feat[:CLIP_NUM].T
    v2t_full_sim = full_feat @ text_feat[CLIP_NUM:].T

    if has_theme:
        clip_theme_id = [i for i in range(len(clip_src)) if clip_src[i] in text_theme_src]
        clip_theme_feat = torch.stack([clip_feat[i] for i in clip_theme_id])
        clip_theme_src = [clip_src[i] for i in clip_theme_id]
        theme_sim = text_theme_feat @ clip_feat.T
        v2t_theme_sim = clip_theme_feat @ text_theme_feat.T

    for k in args.topk:
        clip_pass = full_pass = v2t_clip_pass = v2t_full_pass = 0
        theme_pass = v2t_theme_pass = 0
        _, clip_topk_ids = torch.topk(clip_sim, k, dim=1)
        for i, t_src in enumerate(text_src[:CLIP_NUM]):
            if t_src in [clip_src[j] for j in clip_topk_ids[i]]:
                clip_pass += 1
        _, full_topk_ids = torch.topk(full_sim, k, dim=1)
        for i, t_src in enumerate(text_src[CLIP_NUM:]):
            if t_src in [full_src[j] for j in full_topk_ids[i]]:
                full_pass += 1
        _, v2t_clip_topk_ids = torch.topk(v2t_clip_sim, k, dim=1)
        for i, v_src in enumerate(clip_1000_src):
            if v_src in [text_src[j] for j in v2t_clip_topk_ids[i]]:
                v2t_clip_pass += 1
        _, v2t_full_topk_ids = torch.topk(v2t_full_sim, k, dim=1)
        for i, v_src in enumerate(full_src):
            if v_src in [text_src[j + CLIP_NUM] for j in v2t_full_topk_ids[i]]:
                v2t_full_pass += 1

        print(f"clip pass@{k} = {clip_pass / CLIP_NUM}")
        print(f"full pass@{k} = {full_pass / (len(text_src) - CLIP_NUM)}")
        print(f"v2t clip pass@{k} = {v2t_clip_pass / CLIP_NUM}")
        print(f"v2t full pass@{k} = {v2t_full_pass / (len(text_src) - CLIP_NUM)}")
        if has_theme:
            _, theme_topk_ids = torch.topk(theme_sim, k, dim=1)
            for i, t_src in enumerate(text_theme_src):
                if t_src in [clip_src[j] for j in theme_topk_ids[i]]:
                    theme_pass += 1
            _, v2t_theme_topk_ids = torch.topk(v2t_theme_sim, k, dim=1)
            for i, v_src in enumerate(clip_theme_src):
                if v_src in [text_theme_src[j] for j in v2t_theme_topk_ids[i]]:
                    v2t_theme_pass += 1
            print(f"theme pass@{k} = {theme_pass / len(text_theme_src)}")
            print(f"v2t theme pass@{k} = {v2t_theme_pass / len(text_theme_src)}")
        print()
    
    io_cost = video_dict['video_dict_cost']
    encode_cost = video_dict['preprocess_cost'] + video_dict['encode_image_cost']
    
    w_io = (io_cost + encode_cost) * 1000
    wo_io = encode_cost * 1000
    
    clip_w_io = w_io / image_feat.shape[0] * clip_feat.shape[0]
    clip_wo_io = wo_io / image_feat.shape[0] * clip_feat.shape[0]
    full_w_io = w_io / image_feat.shape[0] * full_feat.shape[0]
    full_wo_io = wo_io / image_feat.shape[0] * full_feat.shape[0]
    
    print(f'all w/ io cost {w_io} ms')
    print(f'all w/o io cost {wo_io} ms')
    print(f'clip w/ io cost {clip_w_io} ms')
    print(f'clip w/o io cost {clip_wo_io} ms')
    print(f'full w/ io cost {full_w_io} ms')
    print(f'full w/o io cost {full_wo_io} ms')

def main(args):
    if args.encode_text:
        encode_text(args)
    elif args.encode_video:
        assert args.num_chunks is not None
        assert args.chunk_idx is not None
        assert args.interval is not None
        encode_video(args)
    elif args.video_merge:
        video_merge(args)
    elif args.calc_pass:
        assert args.topk is not None
        calc_pass(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encode_text', action='store_true')
    
    parser.add_argument('--encode_video', action='store_true')
    parser.add_argument('--num_chunks', type=int)
    parser.add_argument('--chunk_idx', type=int)
    parser.add_argument('--interval', type=int)
    
    parser.add_argument('--video_merge', action='store_true')
    
    parser.add_argument('--calc_pass', action='store_true')
    parser.add_argument('--topk', type=lambda s: map(int, s.split(',')), default=','.join(list(map(str, range(1, 21)))))
    
    args = parser.parse_args()
    
    main(args)
    