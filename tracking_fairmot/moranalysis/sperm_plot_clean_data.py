import pandas as pd
import os
import matplotlib.pyplot as plt

# Load cleaned data
csv_path = 'all_sperm_features_pair_clean_green_17.csv'
df_clean = pd.read_csv(csv_path)

features = [
    'head_area', 'head_aspect_ratio', 'head_circularity',
    'tail_length', 'tail_straightness', 'head_tail_angle'
]

# ===== Trend plots for each ID =====
grouped = df_clean.groupby('global_id')

out_dir = 'sperm_id_trend_plots_clean_green_17'
os.makedirs(out_dir, exist_ok=True)

for gid, group in grouped:
    for feat in features:
        plt.figure(figsize=(10, 6))
        plt.plot(group['frame'], group[feat], marker='o')
        plt.title(f'Global ID {gid}: {feat} over frames (cleaned)')
        plt.xlabel('Frame')
        plt.ylabel(feat)
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(out_dir, f'id{gid:03d}_{feat}_trend_clean.png')
        plt.savefig(save_path)
        plt.close()
        print(f'Saved: {save_path}')


# ===== Per ID mean statistics plot (main & small scale) =====
summary_list = []
for gid, group in grouped:
    summary = {'global_id': gid}
    for col in features:
        mean = group[col].mean()
        summary[f'{col}_mean'] = mean
    summary_list.append(summary)

summary_df = pd.DataFrame(summary_list)

plot_dir = 'sperm_id_stat_plots_clean_green_17'
os.makedirs(plot_dir, exist_ok=True)

main_feats = ['head_area', 'tail_length', 'head_tail_angle']
side_feats = ['head_aspect_ratio', 'head_circularity', 'tail_straightness']


# ===== 全局 boxplot：将所有 ID 的数据拼一起统计 boxplot =====
out_dir_boxplot_global = 'sperm_global_boxplots_clean_green_17'
os.makedirs(out_dir_boxplot_global, exist_ok=True)

# 1️⃣ Main features 汇总 boxplot
plt.figure(figsize=(12, 6))
data_main = [df_clean[feat].dropna().values for feat in main_feats]
plt.boxplot(data_main, labels=main_feats, showmeans=True)
plt.ylabel('Area / Length / Angle')
plt.title('Cleaned: Global Distribution (Boxplot) - Main Features')
plt.grid(True, axis='y')
plt.tight_layout()
save_path_global_main = os.path.join(out_dir_boxplot_global, 'global_boxplot_main_features_clean_green_17.png')
plt.savefig(save_path_global_main)
plt.show()
print(f'Saved: {save_path_global_main}')


# 2️⃣ Small features 汇总 boxplot
plt.figure(figsize=(12, 6))
data_small = [df_clean[feat].dropna().values for feat in side_feats]
plt.boxplot(data_small, labels=side_feats, showmeans=True)
plt.ylabel('Aspect Ratio / Circularity / Straightness (0~1)')
plt.ylim(0, 3)
plt.title('Cleaned: Global Distribution (Boxplot) - Small Features')
plt.grid(True, axis='y')
plt.tight_layout()
save_path_global_small = os.path.join(out_dir_boxplot_global, 'global_boxplot_small_features_clean_green_17.png')
plt.savefig(save_path_global_small)
plt.show()
print(f'Saved: {save_path_global_small}')

