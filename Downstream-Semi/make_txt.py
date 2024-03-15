import os

def write_video_paths_to_txt(directory, output_file):
    with open(output_file + 'train.txt', 'w') as f:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mov')):
                    video_path = os.path.join(root, file)
                    f.write(f"{video_path},0\n")

# 指定要遍历的目录路径
directory_path = 'data/unlabeled/mostgan_cholec/'

# 指定输出的.txt文件路径
output_file_path = 'data/unlabeled/mostgan_cholec/splits/'

os.makedirs(output_file_path, exist_ok=True)


# 调用函数将视频路径写入.txt文件
write_video_paths_to_txt(directory_path, output_file_path)