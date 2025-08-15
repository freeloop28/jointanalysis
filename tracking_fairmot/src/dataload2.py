import os
import json
import cv2

video_dir = os.path.expanduser('~/projects/FairMOT/sperm_video/minkidataset')
output_root = os.path.expanduser('~/projects/FairMOT/minkidataset')
num_sequences = 50

for i in range(num_sequences):
    seq_id = f"{i:03d}"
    video_path = os.path.join(video_dir, f"{seq_id}_raw.mp4")
    json_path = os.path.join(video_dir, f"{seq_id}_bboxes.json")
    if not (os.path.exists(video_path) and os.path.exists(json_path)):
        print(f"{video_path} or {json_path} not found, skip.")
        continue

    # 创建目标文件夹
    seq_dir = os.path.join(output_root, seq_id)
    img1_dir = os.path.join(seq_dir, "img1")
    gt_dir = os.path.join(seq_dir, "gt")
    os.makedirs(img1_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    # 读取视频，逐帧保存图片
    cap = cv2.VideoCapture(video_path)
    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_name = os.path.join(img1_dir, f"{frame_idx:06d}.jpg")
        cv2.imwrite(img_name, frame)
        frame_idx += 1
    cap.release()
    print(f"Extracted {frame_idx-1} frames from {video_path}.")

    # 处理标签
    with open(json_path, "r") as f:
        data = json.load(f)

    for frame_name, objects in data.items():
        if not frame_name.startswith("Frame "):
            continue
        try:
            frame_id = int(frame_name.replace("Frame ", "")) + 1  # 注意和你bbox的frame对齐
        except Exception as e:
            print(f"frame_name error: {frame_name}, {e}")
            continue
        txt_filename = os.path.join(gt_dir, f"{frame_id:06d}.txt")
        with open(txt_filename, "w") as ftxt:
            for obj in objects:
                track_id = obj["Track ID"]
                x, y = obj["BBox Top Left"]
                w, h = obj["BBox Size"]
                # class_id 默认为0
                ftxt.write(f"0 {track_id} {x} {y} {w} {h}\n")
    print(f"Labels done: {gt_dir}")

for i in range(num_sequences):
    seq_id = f"{i:03d}"
    seq_dir = os.path.join(output_root, seq_id)
    img1_dir = os.path.join(seq_dir, "img1")
    seqinfo_path = os.path.join(seq_dir, "seqinfo.ini")

    # 找到第一张图片，获取分辨率
    img_files = sorted([f for f in os.listdir(img1_dir) if f.endswith(".jpg")])
    if not img_files:
        print(f"{img1_dir} 没有图片，跳过")
        continue
    first_img = cv2.imread(os.path.join(img1_dir, img_files[0]))
    imHeight, imWidth = first_img.shape[:2]
    seqLength = len(img_files)
    frameRate = 30  # 默认30，你可以改成自己的

    ini_text = f"""[Sequence]
name={seq_id}
imDir=img1
frameRate={frameRate}
seqLength={seqLength}
imWidth={imWidth}
imHeight={imHeight}
imExt=.jpg
"""
    with open(seqinfo_path, "w") as f:
        f.write(ini_text)
    print(f"已生成: {seqinfo_path}")

print("✅ 所有图片和标签已生成。")
