import os
import json
import pandas as pd

video_dir = os.path.expanduser("~/projects/FairMOT/sperm_video")
output_root = os.path.expanduser("~/projects/FairMOT/MOT_dataset")
num_sequences = 50

for i in range(num_sequences):
    seq_id = f"{i:03d}"
    json_path = os.path.join(video_dir, f"{seq_id}_bboxes.json")
    if not os.path.exists(json_path):
        print(f"{json_path} not found, skip.")
        continue

    with open(json_path, "r") as f:
        data = json.load(f)

    mot_records = []
    for frame_name, objects in data.items():
        frame_id = int(frame_name.replace("Frame ", "")) + 1
        for obj in objects:
            track_id = obj["Track ID"]
            x, y = obj["BBox Top Left"]
            w, h = obj["BBox Size"]
            mot_records.append([frame_id, track_id, x, y, w, h, 1, 1, 1])

    gt_df = pd.DataFrame(mot_records, columns=["frame", "id", "x", "y", "w", "h", "conf", "class", "vis"])

    # 保存 gt.txt
    label_dir = os.path.join(output_root, "labels_with_ids", seq_id)
    os.makedirs(label_dir, exist_ok=True)
    gt_path = os.path.join(label_dir, "gt.txt")
    gt_df.to_csv(gt_path, index=False, header=False)
    print(f"Done: {gt_path}")

    # === 拆分成单帧 txt 文件 ===
    img1_label_dir = os.path.join(label_dir, "img1")
    os.makedirs(img1_label_dir, exist_ok=True)
    # 分组按frame
    for frame_id, group in gt_df.groupby("frame"):
        txt_filename = os.path.join(img1_label_dir, f"{int(frame_id):06d}.txt")
        # 保存当前帧所有目标
        with open(txt_filename, "w") as ftxt:
            for _, row in group.iterrows():
                # 这里可根据FairMOT的标准格式选择输出列，通常为: class_id, track_id, x, y, w, h
                # 有些实现只用id, x, y, w, h，具体根据你的模型来
                ftxt.write(f"0 {int(row['id'])} {row['x']} {row['y']} {row['w']} {row['h']}\n")

    print(f"已拆分所有帧到: {img1_label_dir}")

print("✅ 所有 gt.txt 和单帧标签均已生成。")
