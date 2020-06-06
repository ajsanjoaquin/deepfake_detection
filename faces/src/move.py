import os,shutil
from tqdm import tqdm
import argparse

DATASETS = {
    'original': 'original_sequences/youtube',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'DeepFakeDetection': 'manipulated_sequences/DeepFakeDetection',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures'
    }
parser = argparse.ArgumentParser(description='Move File')
parser.add_argument('--d','--data',type=str, default='original', choices=list(DATASETS.keys()), help='input_path')
args = parser.parse_args()

subfolders= [f.path for f in os.scandir(os.path.join(DATASETS[args.d],'c23/images')) if f.is_dir()]
if args.d=='original':
    out_path='/scratch/users/nus/dcsduxi/real_raw'
else:
    out_path='/scratch/users/nus/dcsduxi/fake_raw'
if not os.path.exists(out_path):
        os.makedirs(out_path)
total=0
for folder in tqdm(subfolders):
  for file in os.listdir(folder):
    if file.endswith('.png'):
      in_path = os.path.join(folder, file)
      total+=1
      shutil.move(in_path, os.path.join(out_path,'{}.png'.format(total)))
print(total)