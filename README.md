# Joint Analysis of Sperm Morphology and Motility

This repository integrates two main components for the joint analysis of sperm morphology and motility:

---

## 1️⃣ Segmentation Module (U-Net-based)

- **Source:** Adapted from [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)  
- **License:** MIT License  
- **Purpose:** Sperm head/tail segmentation to extract morphology-related features.  
- **Key Modifications:**
  - Added loss function variants (Dice, Combo) and implemented AP-based model selection strategy for more robust thresholding.
  - Customized dataset loader and augmentations (flipping, rotation, brightness variations) tailored for sperm microscopy images.
  - Added evaluation metrics (IoU, F1-score, Precision, Recall, AP) with per-class analysis.
  - Implemented post-processing for head-tail pairing.

---

## 2️⃣ Tracking Module (FairMOT-based)

- **Source:** Adapted from [ifzhang/FairMOT](https://github.com/ifzhang/FairMOT)  
- **License:** MIT License  
- **Purpose:** Multi-object tracking of sperm to extract motility-related features from video sequences.  
- **Key Modifications:**
  - Parameter tuning for microscopic sperm videos (e.g., detection confidence thresholds, tracking association parameters).
  - Implemented custom feature extraction (speed, linearity, curvature) for motility classification.
  - Integrated Gaussian Mixture Model (GMM) clustering to relate motility categories (A–D) to WHO standards.
  - Added visualization scripts for trajectory overlay and per-ID movement analysis.

---

## 📂 Project Structure
segment_unet/       # U-Net training/inference scripts  
tracking_fairmot/   # FairMOT-based tracking pipeline  
docs/               # Documentation, figures  
requirements.txt    # Python dependencies  
LICENSE             # License file  
README.md           # Project documentation  

---

## 🚀 Quick Start

### 1. Clone the repository
git clone https://github.com/YOURNAME/jointanalysis.git  
cd jointanalysis  

### 2. Install dependencies
pip install -r requirements.txt  

### 3. Train the segmentation model
cd segment_unet  
python train.py --epochs 50 --batch-size 32 

### 4. Run tracking on video data
cd tracking_fairmot  
tracking verification: python src/track.py mot     --load_model ./exp/mot/sperm_exp/model_last.pth     --data_cfg ./src/lib/cfg/sperm.json     --conf_thres 0.4     --input-video /home/ubuntu/projects/FairMOT/sperm_video/049_raw.mp4     --output-root ./demos
test veriification:  python src/test_det.py mot --data_cfg src/lib/cfg/sperm.json --load_model /home/ubuntu/projects/FairMOT/exp/mot/sperm_exp/model_last.pth
train:  python src/train.py mot \
    --exp_id sperm_exp \
    --data_cfg ./src/lib/cfg/sperm.json \
    --gpus 0 \
    --load_model ./exp/model_last.pth \
    --num_epochs 30 \
    --batch_size 4 \
    --input_h 640 \
    --input_w 640

video make:  ffmpeg -framerate 60 -start_number 0 -i MOT_dataset/outputs/sperm_track_test/049/%05d.jpg -c:v libx264 -pix_fmt yuv420p MOT_dataset/outputs/sperm_track_test/049_raw_result9.mp4

### 4. Joint analysis
cd ../moranalysis
python readingtracking.py

## 📊 Features Extracted

**Morphology Features (from U-Net segmentation):**
- Head area, aspect ratio, circularity
- Tail length, straightness
- Head-tail angle

**Motility Features (from FairMOT tracking):**
- Average speed, trajectory linearity
- Path curvature
- WHO-based A–D category clustering

---

## 📜 Acknowledgements

This project adapts and modifies code from the following repositories:

1. **U-Net segmentation** – [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) (MIT License)  
   Original work by [milesial](https://github.com/milesial).  
   Adapted for sperm microscopy images with new loss functions, augmentations, and evaluation metrics.

2. **FairMOT tracking** – [ifzhang/FairMOT](https://github.com/ifzhang/FairMOT) (MIT License)  
   Original work by [ifzhang](https://github.com/ifzhang).  
   Adapted for microscopic multi-object tracking, motility feature extraction, and classification.

---

## 📄 License
This repository is distributed under the [MIT License](LICENSE).  
Please also refer to the original licenses in the above repositories.

---

> **Author:** Xiao (2025)  
> Master's Thesis – Technical University of Denmark (DTU)  
> *Joint Analysis of Sperm Morphology and Motility Using Machine Learning*

