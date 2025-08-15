import joblib
import json
from lib.tracking_utils.sperm_cluster import load_tracks_from_gt, extract_features_for_clustering, map_gmm_labels_to_grades
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
from matplotlib.patches import Patch

def get_train_labels_dir(json_path):
    with open(json_path, 'r') as f:
        cfg = json.load(f)
    return cfg['root'] + '/labels_with_ids'

if __name__ == '__main__':
    json_path = 'src/lib/cfg/sperm.json'
    gt_root_dir = get_train_labels_dir(json_path)
    print(f'Label root dir: {gt_root_dir}')

    speed_weight = 40.0
    curvature_weight = 0.01

    tracks, ids = load_tracks_from_gt(gt_root_dir, num_frames=60, return_ids=True)
    print(f"加载到 {len(tracks)} 条完整轨迹。")

    features = extract_features_for_clustering(tracks, speed_weight=speed_weight, curvature_weight=curvature_weight)
    gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
    labels = gmm.fit_predict(features)
    grades = map_gmm_labels_to_grades(labels, features)

    # 保存 label -> grade 对应关系
    label_to_grade = {}
    for label in np.unique(labels):
        idx = np.where(labels == label)[0][0]
        label_to_grade[label] = grades[idx]
    joblib.dump(label_to_grade, 'label_to_grade.joblib')

    # 保存 GMM 模型
    joblib.dump(gmm, 'gmm_model.joblib')
    print('训练聚类完成，模型与映射已保存！')

    # === 可视化按 grade 显示颜色 ===
    max_distances = features[:, 0] / speed_weight
    avg_curvatures = features[:, 1] / curvature_weight

    # 定义 grade 对应颜色
    grade_colors = {'A': 'blue', 'B': 'green', 'C': 'orange', 'D': 'red'}

    # 每个点根据其 grade 映射颜色
    colors = [grade_colors[g] for g in grades]

    plt.figure(figsize=(8, 6))
    plt.scatter(max_distances, avg_curvatures, c=colors, s=20)
    plt.axvspan(0, 5, color='#FFFACD', alpha=0.4, zorder=0)      # 淡黄色
    plt.axvspan(5, 25, color='#E0FFE0', alpha=0.4, zorder=0)     # 淡绿色
    plt.axvspan(25, plt.xlim()[1] if plt.xlim()[1]>25 else 60, color='#E0F0FF', alpha=0.4, zorder=0)  # 淡蓝色

    plt.axvline(5, color='gray', linestyle='--', linewidth=1)
    plt.text(5, plt.ylim()[0], '5', color='gray', fontsize=12, ha='center', va='bottom')
    plt.axvline(25, color='gray', linestyle='--', linewidth=1)
    plt.text(25, plt.ylim()[0], '25', color='gray', fontsize=12, ha='center', va='bottom')

    # 手动创建 legend
    handles = [Patch(color=color, label=f'Grade {grade}') for grade, color in grade_colors.items()]
    plt.legend(handles=handles, title="Grade")

    plt.title('Sperm Clustering by Avg Speed (VSL) and Mean Curvature')
    plt.xlabel('Avg Speed (VSL)')
    plt.ylabel('Mean Curvature')
    plt.tight_layout()
    plt.savefig('gmm_clusters_phys_by_grade.png', dpi=150)
    plt.show()

    # === 打印每个 grade 的数量统计 ===
    counter = Counter(grades)
    print('每个 grade 数量统计：')
    for grade, count in counter.items():
        print(f'Grade {grade}: {count}')

# ...前面聚类代码不变...

# 统计每个cluster有多少条轨迹
    unique_labels, counts = np.unique(labels, return_counts=True)
    print('\n每个聚类（Cluster Label）的轨迹数量：')
    for label, count in zip(unique_labels, counts):
        print(f'  Cluster {label}: {count} 条轨迹')

    # 统计每个分级有多少条轨迹
    grade_counter = Counter(grades)
    print('\n每个分级（A/B/C/D）的轨迹数量：')
    for grade in sorted(grade_counter.keys()):
        print(f'  Grade {grade}: {grade_counter[grade]} 条轨迹')




# ------------------------- 每个分级各画5条轨迹 -------------------------
plt.figure(figsize=(10, 8))
grade_colors = {'A': 'blue', 'B': 'green', 'C': 'orange', 'D': 'red'}

for grade in ['A', 'B', 'C', 'D']:
    idxs = [i for i, g in enumerate(grades) if g == grade]
    selected = random.sample(idxs, min(15, len(idxs)))
    if grade != 'D':
        for idx in selected:
            track = np.array(tracks[idx])
            plt.plot(track[:, 0], track[:, 1], 
                     color=grade_colors[grade], 
                     label=grade if idx == selected[0] else "",
                     alpha=0.7, linewidth=2)
    else:
        # 对于D类，画所有点，并用scatter加粗
        for idx in selected:
            track = np.array(tracks[idx])
            # 如果是完全不动的，只画第一个点就足够
            if np.all(track == track[0]):
                plt.scatter(track[0, 0], track[0, 1], color=grade_colors[grade], 
                            label=grade if idx == selected[0] else "", s=3, marker='o', alpha=0.8)
            else:
                # 万一有微小移动，也画一下连线和点
                plt.plot(track[:, 0], track[:, 1], 
                         color=grade_colors[grade], 
                         label=grade if idx == selected[0] else "",
                         alpha=0.7, linewidth=2)
                plt.scatter(track[:, 0], track[:, 1], color=grade_colors[grade], s=3, alpha=0.8)

plt.legend()
plt.title('Example Sperm Tracks from Each Grade')
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.savefig('example_tracks_by_grade.png', dpi=150)
plt.show()


grade_d_idxs = [i for i, g in enumerate(grades) if g == 'D']
print(f"Grade D 的轨迹数：{len(grade_d_idxs)}")
print(f"Grade D 的ID示例：{[ids[i] for i in grade_d_idxs[:10]]}")  # 只看前10个

# 如果需要保存所有 Grade D 的ID
with open('grade_D_ids.txt', 'w') as f:
    for i in grade_d_idxs:
        f.write(str(ids[i]) + '\n')

