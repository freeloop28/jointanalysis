# ----------------------
# 1. 导入包
from collections import defaultdict
import os
import cv2
from PIL import Image
import torch
import numpy as np
from model import LightUNet  # 按你的实际路径
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import joblib
from skimage import measure
# ----------------------

# 2. 读取各类ID列表
blue_ids_path = '/home/ubuntu/projects/FairMOT/results/sperm_track_test/017_blue_ids.joblib'
green_ids_path = '/home/ubuntu/projects/FairMOT/results/sperm_track_test/017_green_ids.joblib'
yellow_ids_path = '/home/ubuntu/projects/FairMOT/results/sperm_track_test/017_yellow_ids.joblib'

blue_ids = set(joblib.load(blue_ids_path))
green_ids = set(joblib.load(green_ids_path))
yellow_ids = set(joblib.load(yellow_ids_path))

print(f'已加载蓝色（A类）ID，共{len(blue_ids)}个:', blue_ids)
print(f'已加载绿色（B类）ID，共{len(green_ids)}个:', green_ids)
print(f'已加载黄色（C类）ID，共{len(yellow_ids)}个:', yellow_ids)

# 3. 读取FairMOT跟踪结果，筛选60帧完整ID
mot_file = '/home/ubuntu/projects/FairMOT/results/sperm_track_test/017.txt'
id2frames = defaultdict(set)
all_records = []

with open(mot_file) as f:
    for line in f:
        fields = line.strip().split(',')
        frame = int(fields[0])
        id_ = int(fields[1])
        x1, y1, w, h = map(float, fields[2:6])
        id2frames[id_].add(frame)
        all_records.append({'frame': frame, 'id': id_, 'x1': x1, 'y1': y1, 'w': w, 'h': h})

valid_ids = [id_ for id_, frames in id2frames.items() if len(frames) == 60]
print('IDs with 60 frames:', valid_ids)

def filter_small_regions(mask, min_tail=20, min_head=10):
    """去除mask中的小连通区域，支持头和尾分别设置阈值"""
    filtered = np.zeros_like(mask)
    for label, min_area in zip([1, 2], [min_tail, min_head]):
        binary = (mask == label).astype(np.uint8)
        labeled = measure.label(binary, connectivity=1)
        props = measure.regionprops(labeled)
        for prop in props:
            if prop.area >= min_area:
                filtered[labeled == prop.label] = label
    return filtered

# 4. 加载UNet模型和推理函数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LightUNet(n_channels=3, n_classes=3, bilinear=True) 
checkpoint = torch.load('/home/ubuntu/projects/FairMOT/moranalysis/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
#model.load_state_dict(checkpoint['model_state'])

model.eval()
model.to(device)
preprocess = A.Compose([
    A.Resize(120, 120, interpolation=cv2.INTER_LINEAR),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
])
def predict_mask(roi_img):
    if isinstance(roi_img, Image.Image):
        roi_img = np.array(roi_img)
    transformed = preprocess(image=roi_img)
    input_tensor = transformed["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        pred_mask = torch.argmax(prob, dim=1)
        return pred_mask.squeeze().cpu().numpy()

# ----------------------

# 5. 批量ROI分割与保存（蓝/绿/黄三类分别存到不同目录）
base_output_dir = '/home/ubuntu/projects/FairMOT/MOT_dataset2/'
color_types = [
    ('blue', blue_ids, '/home/ubuntu/projects/FairMOT/MOT_dataset2/moranalysis_blue'),
    ('green', green_ids, '/home/ubuntu/projects/FairMOT/MOT_dataset2/moranalysis_green'),
    ('yellow', yellow_ids, '/home/ubuntu/projects/FairMOT/MOT_dataset2/moranalysis_yellow'),
]
MIN_TAIL_AREA = 150
MIN_HEAD_AREA = 80

#frames_dir = '/home/ubuntu/projects/FairMOT/outputs/sperm_track_test/017'
frames_dir = '/home/ubuntu/projects/FairMOT/minkidataset/017/img1'
for color, ids_set, output_dir in color_types:
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for rec in all_records:
        if rec['id'] not in valid_ids or rec['id'] not in ids_set:
            continue
        frame_id = rec['frame']
        id_ = rec['id']
        x1, y1, w, h = rec['x1'], rec['y1'], rec['w'], rec['h']
        img_path = os.path.join(frames_dir, f'{frame_id:06d}.jpg')
        img = Image.open(img_path).convert('RGB')
        cx = x1 + w / 2
        cy = y1 + h / 2
        half = 60
        left = max(0, int(cx - half))
        upper = max(0, int(cy - half))
        right = min(img.width, int(cx + half))
        lower = min(img.height, int(cy + half))
        roi = img.crop((left, upper, right, lower))
        mask = predict_mask(roi)
        mask_filtered = filter_small_regions(mask, min_tail=MIN_TAIL_AREA, min_head=MIN_HEAD_AREA)

        out_name = f'id{id_:03d}_frame{frame_id:05d}.png'
        roi.save(os.path.join(output_dir, f'roi_{out_name}'))
        Image.fromarray(mask_filtered.astype(np.uint8)).save(os.path.join(output_dir, f'mask_{out_name}'))
        # 彩色可视化
        class_colors = {0: [0, 0, 0], 1: [0, 255, 0], 2: [255, 0, 0]}
        mask_rgb = np.zeros((mask_filtered.shape[0], mask_filtered.shape[1], 3), dtype=np.uint8)
        for cls, color_val in class_colors.items():
            mask_rgb[mask_filtered == cls] = color_val
        Image.fromarray(mask_rgb).save(os.path.join(output_dir, f'mask_color_{out_name}'))
        count += 1
    print(f"全部分割完成（{color}类，共{count}个）！")

