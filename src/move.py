import json
import shutil
from os.path import join
import argparse
parser = argparse.ArgumentParser(description='Move File according to JSON')
parser.add_argument('--json',type=str, default='.', help='json file')
parser.add_argument('--type',type=str,default='real',help='dataset type')
parser.add_argument('--i',type=str,default='test_vids/real')
parser.add_argument('--o',type=str,default='test_vids/real')
args = parser.parse_args()

path=os.listdir(args.i)
with open(args.json) as f:
  data = json.load(f)
if args.type=='fake':
  count=0
  paired=['_'.join(pair) for pair in data]
  for pair in paired:
    for file in path:
        if pair+'.mp4' in file:
            shutil.copy(join(args.i,file),args.o)
            count+=1
        if count == 100:
            break
  count=0
  paired2=['_'.join(reversed(pair)) for pair in data]
  for pair in paired2:
    for file in path:
        if pair+'.mp4' in file:
            shutil.copy(join(args.i,file),args.o)
            count+=1
        if count == 100:
            break

elif args.type=='real':
  count=0
  for pair in data:
    for vid in pair:
        for file in path:
            if vid+'.mp4' in file:
                shutil.copy(join(args.i,file),args.o)
                count+=1
            if count == 100:
                break