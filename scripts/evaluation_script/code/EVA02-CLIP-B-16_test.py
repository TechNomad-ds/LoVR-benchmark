import torch, argparse, os, pickle
import time
import pandas as pd
import numpy as np
from PIL import Image
from contextlib import contextmanager
# from modeling import VideoCLIP_XL
# from utils.text_encoder import text_encoder
# from demo import video_preprocessing_frame
from decord import VideoReader
from tqdm import tqdm
import tempfile, os, pickle
from eva_clip import create_model_and_transforms, get_tokenizer

os.environ['DECORD_EOF_RETRY_MAX'] = '20480'

RUN_NAME = 'EVA02-CLIP-B-16' #

VIDEO_URL = '/mnt/public/***/***/LOVR/LoVR-benchmark/video_data/long_video_clip/'
MERGED_VIDEO_URL = '/mnt/public/***/***/LOVR/LoVR-benchmark/video_data/merged/'
VIDEO_CAP_PATH = '/mnt/public/***/***/LOVR/LoVR-benchmark/caption_data/all_video.jsonl'
VIDEO_CAP_PATH_LEFT = f'/mnt/public/***/***/LOVR/LoVR-benchmark/caption_data/{RUN_NAME}_left_video.jsonl'
CLIP_CAP_PATH = '/mnt/public/***/***/LOVR/LoVR-benchmark/caption_data/all_clip.jsonl'
WEIGHT_PATH = '/mnt/public/***/***/LOVR/models/MetaCLIP-ViT-H-14' #

TEXT_RESULT_PATH = f'/mnt/public/***/***/LOVR/exp/cache/{RUN_NAME}/text.pt'
TEXT_ALL_RESULT_PATH = f'/mnt/public/***/***/LOVR/exp/cache/{RUN_NAME}/text_all.pt'
VIDEO_RESULT_FOLDER = f'/mnt/public/***/***/LOVR/exp/cache/{RUN_NAME}/results'
VIDEO_RESULT_PATH = f'/mnt/public/***/***/LOVR/exp/cache/{RUN_NAME}/video.pt'
ERROR_LOG = f'/mnt/public/***/***/LOVR/exp/cache/{RUN_NAME}/logs/error.txt'
os.makedirs(VIDEO_RESULT_FOLDER, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    print(f"Begin loading {RUN_NAME} model")
    model_name = "EVA02-CLIP-B-16" 
    pretrained = "/mnt/public/***/***/LOVR/models/EVA/EVA02_CLIP_B_psz16_s8B.pt"
    model, _, preprocess = create_model_and_transforms(model_name, pretrained, force_custom_clip=True)
    tokenizer = get_tokenizer(model_name)
    model = model.to(device)
    return model, preprocess, tokenizer

def encode_text(args):
    model, _, processor = load_model()
    clip_path = CLIP_CAP_PATH
    clip_df = pd.read_json(clip_path, lines=True)
    full_path = VIDEO_CAP_PATH
    full_df = pd.read_json(full_path, lines=True)
    
    all_text = clip_df['cap'].to_list() + full_df['cap'].to_list()
    source = clip_df['path'].apply(lambda x: x.split('/')[-1][:-4]).to_list() + full_df['vid'].to_list()

    # >>>>>>>>>>>>>>>>>>>> encode api <<<<<<<<<<<<<<<<<<<<
    BATCH_SIZE = 256
    text_features = []

    with torch.no_grad(), torch.cuda.amp.autocast():
        for i in range(0, len(all_text), BATCH_SIZE):
            batch_texts = all_text[i:i+BATCH_SIZE]

            # 严格按照例程用 processor 编码
            text = processor(batch_texts).to(device)
            feat = model.encode_text(text)
            # feat = outputs.hidden_states
            feat = feat / feat.norm(dim=-1, keepdim=True)
            text_features.append(feat)

    text_features = torch.cat(text_features, dim=0)
    # >>>>>>>>>>>>>>>>>>>> encode api <<<<<<<<<<<<<<<<<<<<

    # theme_path = '/mnt/public/***/***/LOVR/exp/video_clip_caption/clip_cap_theme.jsonl'
    # theme_df = pd.read_json(theme_path, lines=True)
    # theme_text = theme_df['cap'].to_list()
    # theme_src = theme_df['vid']
    
    # text = text_encoder.tokenize(theme_text, truncate=True).to(device)
    # with torch.no_grad():
        # theme_feat = [model.text_model.encode_text(text[i : i+BATCH_SIZE]) for i in range(0, len(text), BATCH_SIZE)]
    # theme_feat = torch.cat(theme_feat)
    # theme_feat = theme_feat / theme_feat.norm(dim=-1, keepdim=True)
    
    # print(f'{text_features.shape=}')
    text_dict = {
        "feat": text_features,
        "source": source,
        # "theme_feat": theme_feat,
        # "theme_src": theme_src,
    }
    
    torch.save(text_dict, TEXT_RESULT_PATH)

def encode_video(args):
    model, processor, _ = load_model()
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
            except Exception as e:   # 注意这里用 Exception
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
        
        # >>>>>>>>>>>>>>>>>>>> encode api <<<<<<<<<<<<<<<<<<<<
        with _timer("preprocess_cost"):
            video_inputs_list = []
            for frame_list in images:
                pil_imgs = [Image.fromarray(fr[:, :, ::-1]) for fr in frame_list]
                # 对每一帧做 preprocess，结果是 [3, 224, 224]
                video_inputs = torch.stack([processor(pil) for pil in pil_imgs], dim=0).cuda()
                # 变成 [num_frames, 3, 224, 224]
                video_inputs_list.append(video_inputs)

        with torch.no_grad(), torch.autocast("cuda"):
            with _timer("encode_image_cost"):
                video_features = []
                BATCH_SIZE = 32

                # 遍历每个视频
                for video_inputs in video_inputs_list:
                    frame_feats = []
                    num_frames = video_inputs.shape[0]  # 帧数

                    # 生成占位文本（长度要和 batch 对齐）
                    # fake_text = torch.zeros(1, 32).cuda()

                    for i in range(0, num_frames, BATCH_SIZE):
                        batch_imgs = video_inputs[i:i+BATCH_SIZE]  # [B, 3, 224, 224]
                        image_features = model.encode_image(batch_imgs)  
                            # 注意：fake_text 只需要有 batch=1 就行，不用和图像等长
                        frame_feats.append(image_features)

                    # 拼接该视频的所有帧
                    frame_feats = torch.cat(frame_feats, dim=0)

                    # 单帧归一化
                    frame_feats = frame_feats / frame_feats.norm(dim=-1, keepdim=True)

                    # 对该视频的所有帧特征取平均
                    video_feat = frame_feats.mean(dim=0, keepdim=True)

                    # 再次归一化
                    video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)

                    video_features.append(video_feat)

                # 拼接所有视频特征
                video_features = torch.cat(video_features, dim=0)
        # >>>>>>>>>>>>>>>>>>>> encode api <<<<<<<<<<<<<<<<<<<<

        result = {
            "image_feats": video_features,
            "clip_src": clip_src,
            "full_src": full_src,
            **elapsed_time,
        }
        
        try:
            torch.save(result, save_path)
        except:
            print(f"error in saving {save_path}")
            
def video_merge(args):
    video_list = get_video_list()
    processed = [i[:-3] for i in os.listdir(VIDEO_RESULT_FOLDER)]
    assert all([video in processed for video in video_list]) == True

    result = None
    for video in tqdm(video_list, desc='merging...'):
        path = os.path.join(VIDEO_RESULT_FOLDER, f'{video}.pt')
        try:
            video_dict = torch.load(
                path,
                map_location=("cuda:0" if torch.cuda.is_available() else "cpu")
            )
        except Exception as e:
            print(f"[ERROR] Failed to load {path}: {e}")
            continue  # 跳过损坏文件

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
    print("Merging complete.")


def calc_pass(args):
    assert os.path.exists(TEXT_RESULT_PATH)
    assert os.path.exists(VIDEO_RESULT_PATH)
    text_dict = torch.load(TEXT_RESULT_PATH,map_location=("cuda:0" if torch.cuda.is_available() else "cpu"))
    video_dict = torch.load(VIDEO_RESULT_PATH,map_location=("cuda:0" if torch.cuda.is_available() else "cpu"))
    
    text_feat = text_dict['feat']
    image_feat = video_dict['image_feats']
    text_src = text_dict['source']
    # text_theme_feat = text_dict['theme_feat']
    # text_theme_src = text_dict['theme_src'].to_list()
    
    clip_src = video_dict['clip_src']
    clip_feat = torch.stack([row for idx, row in enumerate(image_feat) if clip_src[idx] is not None])
    clip_src = [i for i in clip_src if i is not None]
    
    full_src = video_dict['full_src']
    full_feat = torch.stack([row for idx, row in enumerate(image_feat) if full_src[idx] is not None])
    full_src = [i for i in full_src if i is not None]
    
    # print(clip_feat.shape)
    # print(clip_src[0])
    # print(full_feat.shape)
    # print(full_src[0])
    # print(len(text_src), text_src[0])
    
    clip_1000_id = [i for i in range(len(clip_src)) if clip_src[i] in text_src]
    clip_1000_feat = torch.stack([clip_feat[i] for i in clip_1000_id])
    clip_1000_src = [clip_src[i] for i in clip_1000_id]
    
    # clip_theme_id = [i for i in range(len(clip_src)) if clip_src[i] in text_theme_src]
    # clip_theme_feat = torch.stack([clip_feat[i] for i in clip_theme_id])
    # clip_theme_src = [clip_src[i] for i in clip_theme_id]
        
    CLIP_NUM = len(text_feat) - 467  # 剩下的都是 clip 文本


    # 将所有参与计算的张量转为 float32，避免类型冲突
    clip_sim = text_feat[:CLIP_NUM].float() @ clip_feat.T.float()
    full_sim = text_feat[CLIP_NUM:].float() @ full_feat.T.float()
    # theme_sim = text_theme_feat.float() @ clip_feat.T.float()  # 如果用到

    v2t_clip_sim = clip_1000_feat.float() @ text_feat[:CLIP_NUM].T.float()
    v2t_full_sim = full_feat.float() @ text_feat[CLIP_NUM:].T.float()
    # v2t_theme_sim = clip_theme_feat.float() @ text_theme_feat.T.float()

    
    for k in args.topk:
        clip_pass, full_pass, theme_pass, v2t_clip_pass, v2t_full_pass, v2t_theme_pass = 0, 0, 0, 0, 0, 0
        _, clip_topk_ids = torch.topk(clip_sim, k, dim=1)
        for i, t_src in enumerate(text_src[:CLIP_NUM]):
            if t_src in [clip_src[j] for j in clip_topk_ids[i]]:
                clip_pass += 1
        
        _, full_topk_ids = torch.topk(full_sim, k, dim=1)
        for i, t_src in enumerate(text_src[CLIP_NUM:]):
            if t_src in [full_src[j] for j in full_topk_ids[i]]:
                full_pass += 1
        
        # _, theme_topk_ids = torch.topk(theme_sim, k, dim=1)
        # for i, t_src in enumerate(text_theme_src):
        #     if t_src in [clip_src[j] for j in theme_topk_ids[i]]:
        #         theme_pass += 1
        
        _, v2t_clip_topk_ids = torch.topk(v2t_clip_sim, k, dim=1)
        for i, v_src in enumerate(clip_1000_src):
            if v_src in [text_src[j] for j in v2t_clip_topk_ids[i]]:
                v2t_clip_pass += 1
        
        _, v2t_full_topk_ids = torch.topk(v2t_full_sim, k, dim=1)
        for i, v_src in enumerate(full_src):
            if v_src in [text_src[j + CLIP_NUM] for j in v2t_full_topk_ids[i]]:
                v2t_full_pass += 1
        
        # _, v2t_theme_topk_ids = torch.topk(v2t_theme_sim, k, dim=1)
        # for i, v_src in enumerate(clip_theme_src):
        #     if v_src in [text_theme_src[j] for j in v2t_theme_topk_ids[i]]:
        #         v2t_theme_pass += 1
        
        print(f'clip pass@{k} = {clip_pass / CLIP_NUM}')
        print(f'full pass@{k} = {full_pass / (len(text_src) - CLIP_NUM)}')
        # print(f'theme pass@{k} = {theme_pass / len(text_theme_src)}')
        print(f'v2t clip pass@{k} = {v2t_clip_pass / CLIP_NUM}')
        print(f'v2t full pass@{k} = {v2t_full_pass / (len(text_src) - CLIP_NUM)}')
        # print(f'v2t theme pass@{k} = {v2t_theme_pass / len(text_theme_src)}')
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


def readVideo(local_path, f):
    assert os.path.exists(local_path), f"File not found: {local_path}"
    with open(local_path, 'rb') as vf:
        f.write(vf.read())

def make_video_dict(video_name: str, interval: int):
    """
    功能：
        给定一个视频文件夹（video_name），按指定间隔抽帧，生成图像帧和对应的来源信息。
        返回一个字典，包含：
            - images: 所有抽取到的帧（numpy 数组）
            - clip_src: 每个帧来自的具体 clip 名称
            - full_src: 每个帧所属的大视频名（只在关键帧标注，其他为 None）
    
    参数：
        video_name (str): 视频文件夹名，位于全局 VIDEO_URL 下
        interval (int): 抽帧间隔，例如 interval=30 表示每 30 帧取 1 帧
    """
    images, clip_src, full_src = [], [], []   # 存放抽取结果
    start_offset = 0                          # 控制帧采样的偏移量
    
    vpath = os.path.join(VIDEO_URL, video_name)  # 拼接出视频所在路径

    # 方法 A：尝试读取文件夹内容
    try:
        iter_list = os.listdir(vpath)  # 遍历该文件夹下所有 clip
    except FileNotFoundError:
        raise RuntimeError(f"Directory not found: {vpath}")

    # 遍历每一个视频 clip（比如一个长视频被切分成多个小段）
    for clip in tqdm(iter_list, desc=f'make dict of {video_name[:-1]}'):
        clip_path = os.path.join(vpath, clip)
        with tempfile.NamedTemporaryFile() as f:
            # 将视频 clip 读取到临时文件
            readVideo(clip_path, f)
            try:
                vr = VideoReader(f.name)  # 初始化视频读取器
            except RuntimeError as e:
                # 如果读取失败，就跳过（可能是损坏视频）
                continue

            total_frames = len(vr)  # 视频总帧数

            # 从 0 开始，每隔 interval 帧采样一帧
            clip_frames = vr.get_batch(range(0, total_frames, interval)).asnumpy()
            
            # 为每帧记录它来自哪个 clip（clip[:-4] 去掉扩展名）
            clip_src.extend([clip[:-4]] * len(clip_frames))
            
            # 将这些帧拼接到 images 中
            images = np.concatenate((images, clip_frames), axis=0) if len(images) else clip_frames

            # full_src 用来标记“大视频名”
            if start_offset == 0:
                # 如果当前 clip 是对齐好的，就直接记录视频名
                full_src.extend([video_name[:-1]] * len(clip_frames))
            else:
                # 否则在本批帧里只放 None，额外再补一批对齐帧
                full_src.extend([None] * len(clip_frames))
                
                # 从 (interval - start_offset) 开始，重新采一批帧
                full_frames = vr.get_batch(range(interval - start_offset, total_frames, interval)).asnumpy()
                images = np.concatenate((images, full_frames), axis=0)
                
                # 给这批帧标注完整来源（大视频名）
                full_src.extend([video_name[:-1]] * len(full_frames))
                clip_src.extend([None] * len(full_frames))

            # 更新 offset，确保跨 clip 时采样间隔不被破坏
            start_offset = (start_offset + total_frames) % interval

    # 打包结果
    result = {
        "images": images,       # numpy 数组，存放所有抽取的帧
        "clip_src": clip_src,   # 每帧对应的 clip 名
        "full_src": full_src,   # 每帧对应的大视频名（或 None）
    }
    
    return result

def get_video_list():
    full = pd.read_json(VIDEO_CAP_PATH, lines=True)
    # print(full['vid'])
    return full['vid'].to_list()

def get_chunk(lst, n, k):
    total_len = len(lst)
    base_size = total_len // n
    remainder = total_len % n
    
    start = k * base_size + min(k, remainder)
    end = start + base_size + (1 if k < remainder else 0)
    
    return lst[start:end]

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
    