import json
import shutil
import os.path.join as join
import argparse
parser = argparse.ArgumentParser(description='Move File according to JSON')
parser.add_argument('--json',type=str, default='.', help='json file')
parser.add_argument('--type',type=str,default='real',help='dataset type')
parser.add_argument('--o',type=str,default='test_vids/real')
args = parser.parse_args()

with open(args.json) as f:
  data = json.load(f)
if args.type=='fake':
  paired=['_'.join(pair) for pair in data]
  for pair in paired:
    shutil.copy(join('manipulated_sequences/NeuralTextures/c23/videos',pair+'.mp4'),args.o)
elif args.type=='real':
  for pair in data:
    for vid in pair:
      shutil.copy(join('original_sequences/youtube/c23/videos',vid+'.mp4'),args.o) 