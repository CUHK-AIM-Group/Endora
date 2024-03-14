import os
from tqdm import tqdm
import argparse


def get_filelist(file_path):
    Filelist = []
    for home, dirs, files in os.walk(file_path):
        for filename in files:
            Filelist.append(os.path.join(home, filename))
            # Filelist.append(filename)
    return Filelist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get the corresponding file')
    parser.add_argument('-f', '--frames_dir', type=str, default="", help='Path to the video frames')
    parser.add_argument('-t', '--text_dir', type=str, default="", help='Where to save corresponding text')
    args = parser.parse_args()

    image_root_path = args.frames_dir
    image_txt_path = args.text_dir



    files_list = get_filelist(image_root_path)

    for i in tqdm(files_list):
        relative_path = i.split(image_root_path)[-1]
        with open(image_txt_path, 'a+') as f:
            f.writelines(i + "\n")

