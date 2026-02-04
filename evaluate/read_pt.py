import torch
import os

def summarize_value(key, value):
    print(f"🔑 {key}:")
    try:
        if isinstance(value, torch.Tensor):
            print(f"  - Type: torch.Tensor, shape: {value.shape}, dtype: {value.dtype}")
            # 打印前两个样本的形状
            if value.dim() > 1:  # 确保是二维及以上张量
                print(f"  - Sample 0 shape: {value[0].shape}")
                if value.size(0) > 1:
                    print(f"  - Sample 1 shape: {value[1].shape}")
            print(f"  - Full tensor shape: {value.shape}")
            
        elif isinstance(value, list):
            print(f"  - Type: list, length: {len(value)}")
            if len(value) > 0:
                print(f"    - First item type: {type(value[0])}")
                # 打印前两个列表元素的信息
                if len(value) >= 1:
                    if isinstance(value[0], torch.Tensor):
                        print(f"    - Sample 0: tensor with shape {value[0].shape}")
                    else:
                        print(f"    - Sample 0: {type(value[0]).__name__}, value: {repr(value[0])[:100]}{'...' if len(repr(value[0])) > 100 else ''}")
                
                if len(value) >= 2:
                    if isinstance(value[1], torch.Tensor):
                        print(f"    - Sample 1: tensor with shape {value[1].shape}")
                    else:
                        print(f"    - Sample 1: {type(value[1]).__name__}, value: {repr(value[1])[:100]}{'...' if len(repr(value[1])) > 100 else ''}")
        
        elif isinstance(value, dict):
            print(f"  - Type: dict, keys: {list(value.keys())[:5]}{' ...' if len(value) > 5 else ''}")
            # 打印字典前两个键值对的样本信息
            items = list(value.items())[:2]
            for i, (k, v) in enumerate(items):
                if isinstance(v, torch.Tensor):
                    print(f"    - Sample {i} key: {k}, value: tensor with shape {v.shape}")
                else:
                    print(f"    - Sample {i} key: {k}, value: {type(v).__name__} {repr(v)[:50]}{'...' if len(repr(v)) > 50 else ''}")
        
        elif isinstance(value, (int, float, str, bool)):
            print(f"  - Type: {type(value).__name__}, value: {value}")
        else:
            print(f"  - Type: {type(value).__name__}")
    except Exception as e:
        print(f"  - ⚠️ Error while inspecting value: {e}")

def load_pt_file(pt_path):
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"{pt_path} does not exist.")

    data = torch.load(pt_path, map_location='cpu')

    print(f"\n✅ Loaded file: {pt_path}")
    print(f"📦 Keys in file: {list(data.keys())}")
    print("=" * 60)

    for key, value in data.items():
        summarize_value(key, value)
        print("-" * 40)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect a .pt result file")
    parser.add_argument("pt_path", nargs="?", help="Path to .pt file (e.g. exp/cache/RUN_NAME/results/video_id.pt)")
    args = parser.parse_args()
    if not args.pt_path:
        parser.print_help()
        print("\nExample: python read_pt.py exp/cache/gme-Qwen2-VL-7B-Instruct/results/xxx.pt")
        exit(1)
    load_pt_file(args.pt_path)