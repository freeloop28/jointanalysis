import os

# 设置目标文件夹路径
folder_path = r"/home/ubuntu/seg/new_dataset"  # 改成你的文件夹路径

# 支持的图片扩展名
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

def count_images(path, recursive=True):
    count = 0
    if recursive:
        for root, _, files in os.walk(path):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    count += 1
    else:
        for file in os.listdir(path):
            if os.path.splitext(file)[1].lower() in image_extensions:
                count += 1
    return count

# 调用
total_images = count_images(folder_path, recursive=True)
print(f"图片总数: {total_images}")
