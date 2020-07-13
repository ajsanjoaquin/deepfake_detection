import json
import shutil
import os
from os.path import join
import argparse

parser = argparse.ArgumentParser(description='Move File according to JSON')
parser.add_argument('--json',type=str, default='.', help='json file')
parser.add_argument('--type',type=str,default='real',help='dataset type')
parser.add_argument('--i',type=str,default='test_vids/real')
parser.add_argument('--o',type=str,default='test_vids/real')
args = parser.parse_args()

if not os.path.isdir(args.o):
    os.mkdir(args.o)
path=os.listdir(args.i)
with open(args.json) as f:
  data = json.load(f)
if args.type=='fake':
  paired=['_'.join(pair) for pair in data]
  for pair in (paired):
    count=0
    for file in path:
        if pair+'.mp4' in file:
            shutil.copy(join(args.i,file),args.o)
            count+=1
        if count == 100:
            break
  paired2=['_'.join(reversed(pair)) for pair in data]
  for pair in (paired2):
    count=0
    for file in path:
        if pair+'.mp4' in file:
            shutil.copy(join(args.i,file),args.o)
            count+=1
        if count == 100:
            break

elif args.type=='real':
  for pair in (data):
    for vid in pair:
        count=0
        for file in path:
            if vid+'.mp4' in file:
                shutil.copy(join(args.i,file),args.o)
                count+=1
            if count == 100:
                break