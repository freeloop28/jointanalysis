import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load CSVs
df_blue = pd.read_csv('all_sperm_features_pair_clean_blue_17.csv')
df_green = pd.read_csv('all_sperm_features_pair_clean_green_17.csv')
df_yellow = pd.read_csv('all_sperm_features_pair_clean_yellow_17.csv')

features_main = ['head_area', 'tail_length', 'head_tail_angle']
features_small = ['head_aspect_ratio', 'head_circularity', 'tail_straightness']

out_dir = 'sperm_global_boxplots_clean_compare'
os.makedirs(out_dir, exist_ok=True)


def plot_group_boxplot(features, ylabel, title, save_name, ylim=None):
    plt.figure(figsize=(14, 6))
    positions = [1, 2, 3]
    width = 0.2

    # 左→右顺序：黄、绿、蓝
    color_order = ['yellow', 'green', 'blue']
    colors = {'yellow': '#FFD700', 'green': '#32CD32', 'blue': '#1E90FF'}
    data_dict = {'yellow': df_yellow, 'green': df_green, 'blue': df_blue}

    for i, feat in enumerate(features):
        pos_base = positions[i]

        for j, group in enumerate(color_order):
            data = data_dict[group][feat].dropna()
            pos = pos_base - width + j * width
            bp = plt.boxplot(data, positions=[pos], widths=width, patch_artist=True, showmeans=True)
            for patch in bp['boxes']:
                patch.set_facecolor(colors[group])

    plt.xticks(positions, features)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis='y')
    if ylim:
        plt.ylim(*ylim)

    # Legend
    patches = [
        mpatches.Patch(color=colors['yellow'], label='Yellow (Slow)'),
        mpatches.Patch(color=colors['green'], label='Green (Medium/Special)'),
        mpatches.Patch(color=colors['blue'], label='Blue (Fast)')
    ]
    plt.legend(handles=patches)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, save_name))
    plt.show()


# 主特征
plot_group_boxplot(
    features_main,
    ylabel='Area / Length / Angle',
    title='Yellow vs Green vs Blue (Boxplot) - Main Features',
    save_name='global_boxplot_main_features_compare.png'
)

# 小特征
plot_group_boxplot(
    features_small,
    ylabel='Aspect Ratio / Circularity / Straightness (0~1)',
    title='Yellow vs Green vs Blue (Boxplot) - Small Features',
    save_name='global_boxplot_small_features_compare.png',
    ylim=(0, 3)
)
