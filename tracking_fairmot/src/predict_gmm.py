# predict_gmm.py
import joblib
from sperm_cluster import load_and_align_tracks, extract_features_for_clustering

gmm = joblib.load('gmm_model.joblib')
label_to_grade = joblib.load('label_to_grade.joblib')

test_root_dir = 'your_val_or_test_dir'
tracks, ids = load_and_align_tracks(test_root_dir, return_ids=True)
features = extract_features_for_clustering(tracks)

labels = gmm.predict(features)
grades = [label_to_grade[label] for label in labels]

# 可保存或打印结果
track_id_to_grade = {tid: grade for tid, grade in zip(ids, grades)}
print(track_id_to_grade)
