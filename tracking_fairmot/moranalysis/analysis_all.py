import os
import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import math
import re
from scipy.fft import rfft, rfftfreq

root_folder = '/home/ubuntu/projects/FairMOT/MOT_dataset2/moranalysis_yellow'
tail_length_threshold = 20
angle_threshold_deg = 95
distance_threshold = 20

features = [
    'head_area', 'head_aspect_ratio', 'head_circularity',
    'tail_length', 'tail_straightness', 'head_tail_angle'
]

def extract_frame_no(basename):
    m = re.search(r'frame(\d+)', basename)
    if m:
        return int(m.group(1))
    return -1

def extract_global_id(basename):
    m = re.search(r'id(\d+)', basename)
    if m:
        return int(m.group(1))
    return -1

def find_skeleton_endpoints(skeleton):
    endpoints = []
    h, w = skeleton.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skeleton[y, x] == 0:
                continue
            neighborhood = skeleton[y-1:y+2, x-1:x+2]
            count = np.sum(neighborhood > 0) - 1
            if count == 1:
                endpoints.append((x, y))
    return endpoints

def choose_tail_tip_by_head(endpoints, head_center):
    endpoints = np.array(endpoints)
    distances = np.linalg.norm(endpoints - head_center, axis=1)
    min_idx = np.argmin(distances)
    tip = endpoints[min_idx]
    other = endpoints[1 - min_idx] if len(endpoints) == 2 else tip
    return np.array(tip), np.array(other)

def extract_tail_skeleton_and_endpoints(tail_mask, head_center):
    binary = (tail_mask > 0).astype(np.uint8)
    skeleton = skeletonize(binary).astype(np.uint8) * 255
    endpoints = find_skeleton_endpoints(skeleton)
    if len(endpoints) < 2:
        return skeleton, None, None
    tip, other = choose_tail_tip_by_head(endpoints, head_center)
    return skeleton, tip, other

def extract_multiple_sperm_features(mask_path,
                                    tail_length_threshold=10,
                                    distance_threshold=20,
                                    angle_threshold_deg=25):
    original_img = cv2.imread(mask_path)
    if original_img is None:
        print(f"[Warning] Failed to load image: {mask_path}")
        return []

    mask = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    value_counts = dict(zip(*np.unique(mask, return_counts=True)))
    for bg in (0, 255):
        value_counts.pop(bg, None)
    if not value_counts:
        return []

    tail_pixel_value = max(value_counts, key=value_counts.get)
    head_mask = (mask == 2).astype(np.uint8)
    tail_mask = (mask == 1).astype(np.uint8)

    head_contours, _ = cv2.findContours(head_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tail_contours, _ = cv2.findContours(tail_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heads = []
    tails = []

    for hi, head in enumerate(head_contours):
        if len(head) < 5:
            continue
        (cx, cy), (MA, ma), angle = cv2.fitEllipse(head)
        center = np.array([cx, cy])
        head_area = cv2.contourArea(head)
        head_perimeter = cv2.arcLength(head, True)
        aspect_ratio = ma / MA if MA != 0 else 0
        circularity = 4 * np.pi * head_area / (head_perimeter ** 2) if head_perimeter != 0 else 0
        heads.append({
            'type': 'head',
            'id': hi,
            'head_area': head_area,
            'head_aspect_ratio': aspect_ratio,
            'head_circularity': circularity,
            'head_center_x': center[0],
            'head_center_y': center[1],
            'angle': angle,
            'head_contour': head
        })

    for ti, t_contour in enumerate(tail_contours):
        sub_mask = np.zeros_like(tail_mask)
        cv2.drawContours(sub_mask, [t_contour], -1, 255, -1)
        skeleton = skeletonize(sub_mask > 0).astype(np.uint8) * 255
        coords = np.column_stack(np.where(skeleton > 0))[:, ::-1]
        if coords.shape[0] < 2: continue
        length = np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1))
        straight_line = np.linalg.norm(coords[0] - coords[-1])
        tail_straightness = straight_line / length if length > 0 else 0
        tails.append({
            'type': 'tail',
            'id': ti,
            'tail_length': length,
            'tail_straightness': tail_straightness,
            'tail_root_x': coords[0][0],
            'tail_root_y': coords[0][1],
            'tail_tip_x': coords[-1][0],
            'tail_tip_y': coords[-1][1],
            'tail_coords': coords,
            'tail_contour': t_contour
        })

    pair_candidates = []
    for hi, head in enumerate(heads):
        for ti, tail in enumerate(tails):
            head_center = np.array([head['head_center_x'], head['head_center_y']])
            coords = tail['tail_coords']
            dists = np.linalg.norm(coords - head_center, axis=1)
            root_idx = np.argmin(dists)
            tip_idx = 1 - root_idx if len(coords) == 2 else (-1 if root_idx == 0 else 0)
            tail_vec = coords[tip_idx] - coords[root_idx]
            tail_vec_unit = tail_vec / (np.linalg.norm(tail_vec) + 1e-8)
            angle_rad = np.deg2rad(head['angle'] + 90)
            direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
            dotp = np.clip(np.dot(direction, tail_vec_unit), -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(dotp))
            angle_deg = min(angle_deg, 180 - angle_deg)
            distance = np.linalg.norm(coords[root_idx] - head_center)
            score = 5.0 * distance + angle_deg
            if distance < distance_threshold and angle_deg < angle_threshold_deg:
                pair_candidates.append((score, hi, ti, angle_deg, distance))

    pair_candidates.sort()
    used_heads = set()
    used_tails = set()
    pairs = []
    for score, hi, ti, angle_deg, distance in pair_candidates:
        if hi in used_heads or ti in used_tails:
            continue
        head = heads[hi]
        tail = tails[ti]
        if tail['tail_length'] < tail_length_threshold:
            continue
        used_heads.add(hi)
        used_tails.add(ti)
        pairs.append({
            'type': 'pair',
            'head_id': hi,
            'tail_id': ti,
            'head_area': head['head_area'],
            'head_aspect_ratio': head['head_aspect_ratio'],
            'head_circularity': head['head_circularity'],
            'tail_length': tail['tail_length'],
            'tail_straightness': tail['tail_straightness'],
            'head_tail_angle': angle_deg,
            'head_center_x': head['head_center_x'],
            'head_center_y': head['head_center_y'],
            'tail_root_x': tail['tail_root_x'],
            'tail_root_y': tail['tail_root_y'],
            'tail_tip_x': tail['tail_tip_x'],
            'tail_tip_y': tail['tail_tip_y']
        })

    heads_out = [{k: v for k, v in h.items() if k not in ['head_contour', 'angle']} for h in heads]
    tails_out = [{k: v for k, v in t.items() if k not in ['tail_contour', 'tail_coords']} for t in tails]

    return heads_out + tails_out + pairs

if __name__ == "__main__":
    mask_paths = sorted(glob.glob(os.path.join(root_folder, '*.png')))
    records = []
    for mask_path in tqdm(mask_paths, desc='Processing Frames'):
        basename = os.path.basename(mask_path)
        frame_no = extract_frame_no(basename)
        global_id = extract_global_id(basename)
        feats = extract_multiple_sperm_features(
            mask_path,
            tail_length_threshold=tail_length_threshold,
            distance_threshold=distance_threshold,
            angle_threshold_deg=angle_threshold_deg
        )
        for f in feats:
            f['frame'] = frame_no
            f['filename'] = basename
            f['global_id'] = global_id
            records.append(f)

    df = pd.DataFrame(records)
    df = df.sort_values(['global_id', 'frame', 'type', 'id'], ignore_index=True)
    df.to_csv('all_sperm_features_full_yellow_17.csv', index=False)
    print('✅ 全部头部、尾部、配对精子特征已保存到 all_sperm_features_full.csv')
    print(df.head())

    # ===== 按 global_id 聚合统计每个精子的均值/方差/CV（只对pair，也可换成tail/head） =====
    df_pair = df[df['type'] == 'pair']
    grouped = df_pair.groupby('global_id')

    summary_list = []
    for gid, group in grouped:
        summary = {'global_id': gid}
        for col in features:
            mean = group[col].mean()
            std = group[col].std()
            cv = std / mean if mean != 0 else 0
            summary[f'{col}_mean'] = mean
            summary[f'{col}_std'] = std
            summary[f'{col}_cv'] = cv
        summary_list.append(summary)

    summary_df = pd.DataFrame(summary_list)
    summary_df.to_csv('sperm_features_by_id_yellow_17.csv', index=False)
    print('\n已保存每个精子的统计（按id聚合）：sperm_features_by_id.csv')

    # ==== 可选：画每个精子的特征随帧数变化趋势图 ====
    out_dir = 'sperm_id_trend_plots'
    os.makedirs(out_dir, exist_ok=True)
    for gid, group in grouped:
        for feat in features:
            plt.figure(figsize=(10, 6))
            plt.plot(group['frame'], group[feat], marker='o')
            plt.title(f'Global ID {gid}: {feat} over frames')
            plt.xlabel('Frame')
            plt.ylabel(feat)
            plt.grid(True)
            plt.tight_layout()
            save_path = os.path.join(out_dir, f'id{gid:03d}_{feat}_trend.png')
            plt.savefig(save_path)
            plt.close()
            print(f'Saved: {save_path}')

plot_dir = "sperm_id_stat_plots_yellow_17"
os.makedirs(plot_dir, exist_ok=True)

main_feats = ['head_area', 'tail_length', 'head_tail_angle']   # 大量级
side_feats = ['head_aspect_ratio', 'head_circularity', 'tail_straightness']  # 0~1小量级

# ====== 图1：大量级特征 ======
plt.figure(figsize=(16, 8))
for feat in main_feats:
    plt.plot(summary_df['global_id'], summary_df[f'{feat}_mean'], marker='o', label=feat)
plt.xlabel('Global ID')
plt.ylabel('Area / Length / Angle')
plt.title('各精子ID特征均值（大量级, pair聚合）')
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path_main = os.path.join(plot_dir, 'stat_by_id_mean_main_features_yellow_17.png')
plt.savefig(save_path_main)
plt.show()
print(f'Saved: {save_path_main}')

# ====== 图2：小量级特征（0~1）======
plt.figure(figsize=(16, 8))
for feat in side_feats:
    plt.plot(summary_df['global_id'], summary_df[f'{feat}_mean'], marker='o', label=feat)
plt.xlabel('Global ID')
plt.ylabel('Aspect Ratio / Circularity / Straightness')
plt.ylim(0, 3)
plt.title('各精子ID特征均值（小量级0~1, pair聚合）')
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path_side = os.path.join(plot_dir, 'stat_by_id_mean_small_features_yellow_17.png')
plt.savefig(save_path_side)
plt.show()
print(f'Saved: {save_path_side}')

# ===== 每个 ID 做 BoxPlot，分开画主量纲和0~1量纲特征 =====
boxplot_dir = "sperm_id_boxplots"
os.makedirs(boxplot_dir, exist_ok=True)

main_feats = ['head_area', 'tail_length', 'head_tail_angle']  # 大量级
side_feats = ['head_aspect_ratio', 'head_circularity', 'tail_straightness']  # 0~1小量级

for gid, group in grouped:
    # 1️⃣ 大量纲特征
    plt.figure(figsize=(12, 6))
    data_main = [group[feat].dropna().values for feat in main_feats]
    plt.boxplot(data_main, labels=main_feats, showmeans=True)
    plt.title(f'Global ID {gid} - Main Features Distribution (Boxplot)yellow_17')
    plt.ylabel('Area / Length / Angle')
    plt.xticks(rotation=20)
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(boxplot_dir, f'id{gid:03d}_boxplot_main_yellow_17.png')
    plt.savefig(save_path)
    plt.close()
    print(f'Saved: {save_path}')

    # 2️⃣ 小量纲特征（0~1范围）
    plt.figure(figsize=(12, 6))
    data_side = [group[feat].dropna().values for feat in side_feats]
    plt.boxplot(data_side, labels=side_feats, showmeans=True)
    plt.title(f'Global ID {gid} - Small Features (0~1) Distribution (Boxplot)yellow_17')
    plt.ylabel('Aspect Ratio / Circularity / Straightness (0~1)')
    plt.ylim(0, 3)  # 同前，预留空间
    plt.xticks(rotation=20)
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(boxplot_dir, f'id{gid:03d}_boxplot_small_yellow_17.png')
    plt.savefig(save_path)
    plt.close()
    print(f'Saved: {save_path}')
