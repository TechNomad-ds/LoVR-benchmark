# from petrel_client.client import Client
from decord import VideoReader
from tqdm import tqdm
import tempfile, os, pickle
import numpy as np
import pandas as pd

# conf_path = '~/petreloss.conf'
# client = Client(conf_path)

# def readVideo(url, f):
#     exists = client.contains(url)
#     assert exists
#     video_bytes = client.get(url)
#     f.write(video_bytes)

def readVideo(local_path, f):
    assert os.path.exists(local_path), f"File not found: {local_path}"
    with open(local_path, 'rb') as vf:
        f.write(vf.read())


# readVideo('s3://dhj_data/acmmm_video_clips/-7wwfGJXEZg/-7wwfGJXEZg-Scene-001.mp4')

VIDEO_URL = '/mnt/public/lianghao/hzy/LOVR/LoVR-benchmark/video_data/long_video_clip/'
MERGED_VIDEO_URL = '/mnt/public/lianghao/hzy/LOVR/LoVR-benchmark/video_data/merged/'

def make_video_dict(video_name: str, interval: int):
    images, clip_src, full_src = [], [], []
    start_offset = 0
    
    vpath = os.path.join(VIDEO_URL, video_name)
    try:
        iter_list = os.listdir(vpath)
    except FileNotFoundError:
        raise RuntimeError(f"Directory not found: {vpath}")

    for clip in tqdm(iter_list, desc=f'make dict of {video_name[:-1]}'):
        clip_path = os.path.join(vpath, clip)
        with tempfile.NamedTemporaryFile() as f:
            readVideo(clip_path, f)
            try:
                vr = VideoReader(f.name)
            except RuntimeError:
                continue
            total_frames = len(vr)
            
            clip_frames = vr.get_batch(range(0, total_frames, interval)).asnumpy()
            clip_src.extend([clip[:-4]] * len(clip_frames))
            images = np.concatenate((images, clip_frames), axis=0) if len(images) else clip_frames

            if start_offset == 0:
                full_src.extend([video_name[:-1]] * len(clip_frames))
            else:
                full_src.extend([None] * len(clip_frames))
                full_frames = vr.get_batch(range(interval - start_offset, total_frames, interval)).asnumpy()
                images = np.concatenate((images, full_frames), axis=0)
                full_src.extend([video_name[:-1]] * len(full_frames))
                clip_src.extend([None] * len(full_frames))

            start_offset = (start_offset + total_frames) % interval

    result = {
        "images": images,
        "clip_src": clip_src,
        "full_src": full_src,
    }
    
    return result

def get_video_list():
    path = os.path.join(DATA_ROOT, "caption_data", "all_video.jsonl")
    full = pd.read_json(path, lines=True)
    return full["vid"].to_list()

def get_chunk(lst, n, k):
    total_len = len(lst)
    base_size = total_len // n
    remainder = total_len % n
    
    start = k * base_size + min(k, remainder)
    end = start + base_size + (1 if k < remainder else 0)
    
    return lst[start:end]

def main():
    video_list = get_video_list()
    if not video_list:
        print("No videos. Set DATA_ROOT to your LoVR data root (see evaluate/README.md).")
        return
    video_dict = make_video_dict(video_list[0] + "/", 30)
    print(video_dict["images"].shape, video_dict["images"].dtype, len(video_dict["clip_src"]))


if __name__ == "__main__":
    main()


