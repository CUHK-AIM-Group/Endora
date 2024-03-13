import os
from tqdm import tqdm

ffs_image_root = '/data1/tempt01/code/data/Kvasir-Capsule_frames'
ffs_image_txt = '/data1/tempt01/code/data/Kvasir-Capsule_frames/train_128_list.txt'

def get_filelist(file_path):
    Filelist = []
    for home, dirs, files in os.walk(file_path):
        for filename in files:
            Filelist.append(os.path.join(home, filename))
            # Filelist.append(filename)
    return Filelist

ffs_files = get_filelist(ffs_image_root)

for i in tqdm(ffs_files):
    relative_path = i.split(ffs_image_root)[-1]
    with open(ffs_image_txt, 'a+') as f:
        # f.writelines(relative_path + '\n')
        f.writelines(i + "\n")

# CUDA_VISIBLE_DEVICES=4 python process_list.py