import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, required=True, help='实验名称，如 5b_lora_ot_dim_align_exp1_videophy2')
parser.add_argument('--video_dir', type=str, required=True, help='生成视频的目录路径')
args = parser.parse_args()

target_ver = args.exp_name
prefix = args.video_dir.rstrip('/') + '/'

files = os.listdir(prefix)
mp4_files = [f for f in files if f.endswith('.mp4')]
print(f"Found {len(mp4_files)} mp4 files")

data = []

for fl in mp4_files:
    caption = ' '.join(fl.removesuffix('.mp4').split('_'))
    video_path = prefix + fl
    data.append([video_path, caption])

df = pd.DataFrame(data, columns=['videopath', 'caption'])

os.makedirs('./csv_file', exist_ok=True)
output_file = f'./csv_file/{target_ver}.csv'
df.to_csv(output_file, index=False, encoding='utf-8')

print(f"Data has been written to {output_file}")
print(f"Total videos: {len(df)}")
