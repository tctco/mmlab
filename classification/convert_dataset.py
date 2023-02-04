import os
from torch.utils.data import random_split
import torch
from pathlib import Path
import shutil

torch.manual_seed(0)
DATA_ROOT = Path('./flower_datasets/')
TRAIN_RATIO = 0.8

for flower in os.listdir(DATA_ROOT):
  fnames = os.listdir(os.path.join(DATA_ROOT, flower))
  n_train = round(TRAIN_RATIO*len(fnames))
  train, val = random_split(fnames, [n_train, len(fnames)-n_train])
  target_dir = Path(f'./flower_dataset/train/{flower}')
  target_dir.mkdir(parents=True, exist_ok=True)
  for img in train:
    shutil.copy(DATA_ROOT/flower/img, target_dir/img)

  target_dir = Path(f'./flower_dataset/val/{flower}')
  target_dir.mkdir(parents=True, exist_ok=True)
  for img in val:
    shutil.copy(DATA_ROOT/flower/img, target_dir/img)

with open('./flower_dataset/classes.txt', 'w') as f:
  f.write('\n'.join(os.listdir(DATA_ROOT)))