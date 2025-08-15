import os
import json
import matplotlib.pyplot as plt
import numpy as np

def dataload_track_aligned_to_x_axis(root_dir, save_normalized=True, visualize=True):
    """
    对每个子目录下的 track.json 进行轨迹归一化，并将方向旋转对齐到正X轴。

    参数:
        root_dir (str): 数据集根目录
        save_normalized (bool): 是否保存归一化后的轨迹数据
        visualize (bool): 是否绘制所有归一化后的轨迹
    """
    all_tracks = []

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file == "track_normalized.json":
                file_path = os.path.join(subdir, file)

                # 加载轨迹数据
                with open(file_path, 'r') as f:
                    track_data = json.load(f)

                # 排序帧并提取坐标
                points = [tuple(v) for k, v in sorted(track_data.items(), key=lambda x: int(x[0]))]

                # 平移：起点设为 (0,0)
                origin_x, origin_y = points[0]
                normalized = [(x - origin_x, y - origin_y) for x, y in points]

                # 获取中点
                mid_index = len(normalized) // 2
                if len(normalized) % 2 == 0:
                    x1, y1 = normalized[mid_index - 1]
                    x2, y2 = normalized[mid_index]
                    xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
                else:
                    xm, ym = normalized[mid_index]

                # 第一步旋转：让中点方向对齐 x 轴
                angle = np.arctan2(ym, xm)
                theta = -angle
                cos_a, sin_a = np.cos(theta), np.sin(theta)

                rotated = [
                    (
                        x * cos_a - y * sin_a,
                        x * sin_a + y * cos_a
                    ) for x, y in normalized
                ]

                # 第二步：旋转后如果中点仍在负x轴，翻转180°
                if len(rotated) % 2 == 0:
                    x1, y1 = rotated[len(rotated)//2 - 1]
                    x2, y2 = rotated[len(rotated)//2]
                    rx, ry = (x1 + x2) / 2, (y1 + y2) / 2
                else:
                    rx, ry = rotated[len(rotated)//2]

                if rx < 0:
                    rotated = [(-x, -y) for x, y in rotated]

                # 收集轨迹
                all_tracks.append(rotated)

                # 保存新轨迹
                if save_normalized:
                    normalized_data = {f"{i:03}": [float(x), float(y)] for i, (x, y) in enumerate(rotated)}
                    new_file_path = os.path.join(subdir, "track_normalized_rotated.json")
                    with open(new_file_path, 'w') as f:
                        json.dump(normalized_data, f, indent=4)

    # 可视化所有轨迹
  # 可视化所有轨迹
    if visualize:
        # 全部轨迹图
        plt.figure(figsize=(12, 10))
        for track in all_tracks:
            x_vals, y_vals = zip(*track)
            plt.plot(x_vals, y_vals, alpha=0.5)
        plt.title("All Tracking Paths Aligned to X Axis")
        plt.xlabel("X (rotated)")
        plt.ylabel("Y (rotated)")
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

        # 前10条轨迹放大图
        plt.figure(figsize=(12, 8))
        offset_step = 20
        for i, track in enumerate(all_tracks[30:50]):
            x_vals, y_vals = zip(*track)
            y_vals_offset = [y + i * offset_step for y in y_vals] 
            plt.plot(x_vals, y_vals_offset, label=f'Track {i}')
        plt.title("First 10 Normalized and Rotated Tracks")
        plt.xlabel("X (rotated)")
        plt.ylabel("Y (rotated)")
        plt.legend()
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()


# 示例运行（请替换路径）
if __name__ == "__main__":
    dataload_track_aligned_to_x_axis("/mnt/d/thesis/dataset", save_normalized=True, visualize=True)

