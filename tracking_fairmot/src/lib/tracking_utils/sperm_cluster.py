# sperm_cluster.py
import os
import json
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import collections

# def load_and_align_tracks(root_dir, max_len=60, return_ids=False):
#     """
#     加载所有track_normalized.json轨迹，并做对齐归一化，返回轨迹序列（可选返回track_ids）。
#     每个track是点的序列：[(x0, y0), (x1, y1), ...]
#     """
#     tracks = []
#     track_ids = []
#     for subdir, dirs, files in os.walk(root_dir):
#         for file in files:
#             if file == "track_normalized.json":
#                 file_path = os.path.join(subdir, file)
#                 # 获取track_id（假定文件夹名如 id_1, id_2 ...）
#                 try:
#                     tid = int(os.path.basename(subdir).replace('id_', ''))
#                 except:
#                     tid = None
#                 with open(file_path, 'r') as f:
#                     track_data = json.load(f)
#                 # 点排序（默认按key为帧id数字升序）
#                 points = [tuple(v) for k, v in sorted(track_data.items(), key=lambda x: int(x[0]))]
#                 # 坐标归一化
#                 origin_x, origin_y = points[0]
#                 normalized = [(x - origin_x, y - origin_y) for x, y in points]
#                 # 中心点
#                 mid_index = len(normalized) // 2
#                 if len(normalized) % 2 == 0:
#                     x1, y1 = normalized[mid_index - 1]
#                     x2, y2 = normalized[mid_index]
#                     xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
#                 else:
#                     xm, ym = normalized[mid_index]
#                 # 主轴对齐
#                 angle = np.arctan2(ym, xm)
#                 theta = -angle
#                 cos_a, sin_a = np.cos(theta), np.sin(theta)
#                 rotated = [(x * cos_a - y * sin_a, x * sin_a + y * cos_a) for x, y in normalized]
#                 # 朝向统一
#                 if len(rotated) % 2 == 0:
#                     x1, y1 = rotated[mid_index - 1]
#                     x2, y2 = rotated[mid_index]
#                     rx, ry = (x1 + x2) / 2, (y1 + y2) / 2
#                 else:
#                     rx, ry = rotated[mid_index]
#                 if rx < 0:
#                     rotated = [(-x, -y) for x, y in rotated]
#                 tracks.append(rotated)
#                 track_ids.append(tid)
#     if return_ids:
#         return tracks, track_ids
#     else:
#         return tracks


def load_tracks_from_gt(root_dir, num_frames=60, return_ids=True):
    """
    从每个组的gt.txt还原每个精子的轨迹，要求轨迹要完整有num_frames帧。
    返回: tracks, track_ids
    """
    tracks = []
    track_ids = []
    groups = sorted([g for g in os.listdir(root_dir) if g.isdigit()])
    for group in groups:
        gt_path = os.path.join(root_dir, group, 'gt.txt')
        if not os.path.exists(gt_path):
            continue
        # 解析每行
        id2track = collections.defaultdict(lambda: [None] * num_frames)
        with open(gt_path, 'r') as f:
            for line in f:
                fields = line.strip().split(',')
                if len(fields) < 4:
                    continue
                frame_id = int(fields[0])
                sperm_id = int(fields[1])
                x, y = float(fields[2]), float(fields[3])
                if 1 <= frame_id <= num_frames:
                    id2track[sperm_id][frame_id - 1] = (x, y)
        # 只收集完整轨迹
        for sperm_id, points in id2track.items():
            if all(pt is not None for pt in points):
                tracks.append(points)
                track_ids.append(f"{group}_{sperm_id}")
    if return_ids:
        return tracks, track_ids
    else:
        return tracks


def pad_or_crop(track, length=60):
    """
    补齐/截断轨迹为固定长度
    """
    if len(track) >= length:
        return track[:length]
    else:
        #return track + [(0, 0)] * (length - len(track))
        track + [track[-1]] * (length - len(track))


# def compute_average_speed(track):
#     """
#     计算轨迹平均总速度
#     """
#     displacements = [np.linalg.norm(np.subtract(p2, p1)) for p1, p2 in zip(track[:-1], track[1:])]
#     return np.mean(displacements) if len(displacements) > 0 else 0
microns_per_pixel = 0.323
fps=30

def compute_normalized_max_distance(track):
    points = np.array(track)
    n_frames = len(points)
    if n_frames < 2:
        return 0
    dists = np.sqrt(np.sum((points[None, :, :] - points[:, None, :]) ** 2, axis=2))
    max_dist = np.max(dists)
    # 推荐用 (帧数 - 1) 归一化，这样和平均速度类似
    return max_dist / (n_frames - 1)*microns_per_pixel*fps

def compute_max_distance(track):
    points = np.array(track)
    if len(points) < 2:
        return 0
    dists = np.sqrt(np.sum((points[None, :, :] - points[:, None, :]) ** 2, axis=2))
    return np.max(dists)

def compute_average_curvature(track):
    """
    计算轨迹平均曲率
    """
    curvatures = []
    for i in range(1, len(track) - 1):
        p0 = np.array(track[i - 1])
        p1 = np.array(track[i])
        p2 = np.array(track[i + 1])
        a = np.linalg.norm(p1 - p0)
        b = np.linalg.norm(p2 - p1)
        c = np.linalg.norm(p2 - p0)
        if a == 0 or b == 0:
            continue
        angle = np.arccos(np.clip(np.dot(p1 - p0, p2 - p1) / (a * b), -1.0, 1.0))
        if c != 0:
            curvature = abs(2 * np.sin(angle) / c)
            curvatures.append(curvature)
    return np.mean(curvatures) if curvatures else 0


def fit_and_save_global_cluster(train_dirs, model_path):
    features = []
    for d in train_dirs:
        # 读取、提取所有track特征
        features += extract_features_from_dir(d)
    features = np.vstack(features)
    gmm = GaussianMixture(n_components=4)
    gmm.fit(features)
    joblib.dump(gmm, model_path)
    return gmm

def predict_with_global_cluster(test_dirs, model_path):
    gmm = joblib.load(model_path)
    for d in test_dirs:
        features = extract_features_from_dir(d)
        labels = gmm.predict(features)

def extract_features_for_clustering(tracks, speed_weight=10.0, curvature_weight=0.01, pad_length=60):
    """
    针对聚类特征抽取：平均速度 + 平均曲率
    """
    features = []
    for track in tracks:
        track = pad_or_crop(track, pad_length)
        #avg_speed = compute_average_speed(track)
        #max_distance = compute_max_distance(track)
        avg_curvature = compute_average_curvature(track)
        avg_speed = compute_normalized_max_distance(track)
        features.append([avg_speed * speed_weight, avg_curvature * curvature_weight])
    return np.array(features)

def cluster_tracks_gmm(tracks, n_components=4, speed_weight=10.0, curvature_weight=0.01, random_state=42, pad_length=60):
    """
    GMM 聚类，返回labels和features
    """
    features = extract_features_for_clustering(tracks, speed_weight, curvature_weight, pad_length)
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=random_state)
    labels = gmm.fit_predict(features)
    return labels, features

def map_gmm_labels_to_grades(labels, features):
    """
    把聚类标签按最大运动距离从大到小映射为A/B/C/D医学分级
    """
    cluster_avg_dist = {}
    for label in np.unique(labels):
        cluster_dists = features[labels == label, 0]
        cluster_avg_dist[label] = np.mean(cluster_dists)
    sorted_labels = sorted(cluster_avg_dist, key=cluster_avg_dist.get, reverse=True)

    # 打印每个cluster的最大运动距离
    print("每个聚类的平均最大运动距离：")
    for label, avg_dist in cluster_avg_dist.items():
        print(f"  label {label}: 平均最大运动距离 {avg_dist:.4f}")

    print("排序后标签 -> 分级：")
    for label, grade in zip(sorted_labels, ['A', 'B', 'C', 'D']):
        print(f"  label {label} -> {grade} (平均最大运动距离 {cluster_avg_dist[label]:.4f})")
    
    label_to_grade = {label: grade for label, grade in zip(sorted_labels, ['A', 'B', 'C', 'D'])}
    return np.array([label_to_grade[label] for label in labels], dtype=str)

# ==== 方便的all-in-one主函数，返回所有聚类和映射 ====
def cluster_tracks_and_map_grades(tracks, track_ids, n_components=4, speed_weight=10.0, curvature_weight=0.01, pad_length=60):
    """
    聚类+医学分级，返回track_id到grade的映射字典
    """
    labels, features = cluster_tracks_gmm(tracks, n_components, speed_weight, curvature_weight, pad_length=pad_length)
    mapped_grades = map_gmm_labels_to_grades(labels, features)
    track_id_to_grade = {tid: grade for tid, grade in zip(track_ids, mapped_grades)}
    return track_id_to_grade, labels, mapped_grades, features
