import pandas as pd
import os
import matplotlib.pyplot as plt

# Read your existing csv
csv_path = 'all_sperm_features_full_blue_17.csv'
df = pd.read_csv(csv_path)
df_pair = df[df['type'] == 'pair']

# Features to check
features = [
    'head_area', 'head_aspect_ratio', 'head_circularity',
    'tail_length', 'tail_straightness', 'head_tail_angle'
]

# ===== Find outlier frames (IQR) =====
outlier_records = []
grouped = df_pair.groupby('global_id')

for gid, group in grouped:
    for feat in features:
        Q1 = group[feat].quantile(0.25)
        Q3 = group[feat].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = group[(group[feat] < lower) | (group[feat] > upper)]
        for idx, row in outliers.iterrows():
            outlier_records.append({
                'global_id': gid,
                'frame': row['frame'],
                'feature': feat,
                'value': row[feat],
                'lower_bound': lower,
                'upper_bound': upper
            })

# Save outlier frame list
outlier_df = pd.DataFrame(outlier_records)
outlier_df.to_csv('sperm_outlier_frames_blue_17.csv', index=False)
print('✅ Outlier frames saved to: sperm_outlier_frames.csv')

# ===== Remove outliers, save cleaned data =====
df_pair_clean = df_pair.copy()
for feat in features:
    Q1 = df_pair_clean.groupby('global_id')[feat].transform('quantile', 0.25)
    Q3 = df_pair_clean.groupby('global_id')[feat].transform('quantile', 0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df_pair_clean = df_pair_clean[(df_pair_clean[feat] >= lower) & (df_pair_clean[feat] <= upper)]

df_pair_clean.to_csv('all_sperm_features_pair_clean_blue_17.csv', index=False)
print('✅ Cleaned data saved to: all_sperm_features_pair_blue.csv')

# ===== Plot outlier count per ID =====
outlier_count = outlier_df.groupby('global_id').size().reset_index(name='outlier_count')

plt.figure(figsize=(12, 6))
plt.bar(outlier_count['global_id'], outlier_count['outlier_count'])
plt.xlabel('Global ID')
plt.ylabel('Number of Outlier Frames')
plt.title('Outlier Frame Count per Sperm ID')
plt.grid(True, axis='y')
plt.tight_layout()
os.makedirs('sperm_outlier_plots', exist_ok=True)
save_path = 'sperm_outlier_plots/outlier_count_per_id_blue_17.png'
plt.savefig(save_path)
plt.show()
print(f'✅ Outlier count figure saved to: {save_path}')
