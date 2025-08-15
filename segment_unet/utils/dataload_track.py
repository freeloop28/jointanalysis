import os
import json
import matplotlib.pyplot as plt


def dataload_track_save_separately(root_dir, save_normalized=True, visualize=True):
    """
    遍历所有子目录中的 track.json 文件，进行归一化，并可选保存到新的文件名，统一可视化所有轨迹。

    参数:
        root_dir (str): 根目录路径，例如 D:/thesis/new_dataset
        save_normalized (bool): 是否保存归一化后的数据（不覆盖原始文件）
        visualize (bool): 是否统一可视化所有轨迹
    """
    all_tracks = []

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file == "track.json":
                file_path = os.path.join(subdir, file)

                # 加载track.json
                with open(file_path, 'r') as f:
                    track_data = json.load(f)

                # 获取点坐标并归一化
                points = [tuple(v) for k, v in sorted(track_data.items(), key=lambda x: int(x[0]))]
                origin_x, origin_y = points[0]
                normalized_points = [(x - origin_x, y - origin_y) for x, y in points]
                all_tracks.append(normalized_points)

                # 保存为新的JSON文件
                if save_normalized:
                    normalized_data = {f"{i:03}": [x, y] for i, (x, y) in enumerate(normalized_points)}
                    new_file_path = os.path.join(subdir, "track_normalized.json")
                    with open(new_file_path, 'w') as f:
                        json.dump(normalized_data, f, indent=4)

    # 可视化所有轨迹
    if visualize:
        plt.figure(figsize=(10, 8))
        for track in all_tracks:
            x_vals, y_vals = zip(*track)
            plt.plot(x_vals, y_vals, marker='o', alpha=0.6)
        plt.title("All Normalized Tracking Paths")
        plt.xlabel("X (normalized)")
        plt.ylabel("Y (normalized)")
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

if __name__ == "__main__":
    dataload_track_save_separately("/mnt/d/thesis/dataset", save_normalized=True, visualize=True)

