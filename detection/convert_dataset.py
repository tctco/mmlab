import json
from PIL import Image
from pathlib import Path
import numpy as np
import cv2

DATA_ROOT = Path('./data/balloon/')
for d in ['train', 'val']:
  p = DATA_ROOT/d

  with open(p/'via_region_data.json', 'r') as f:
    data = json.load(f)

  images = []
  annotations = []
  categories = [{'id': 0, 'name': 'balloon', 'supercategory': 'none'}]
  ann_id = 0
  for i, v in enumerate(data.values()):
    img = Image.open(p/v['filename'])
    images.append({'id': i, 'file_name': v['filename'], 'width': img.width, 'height': img.height})
    for r in v['regions'].values():
      points = r['shape_attributes']
      points = np.array([points['all_points_x'], points['all_points_y']]).T
      area = cv2.contourArea(points.astype(np.float32))
      xmin, ymin = points.min(axis=0).tolist()
      xmax, ymax = points.max(axis=0).tolist()
      annotations.append({
        'id': ann_id,
        'image_id': i,
        'category_id': 0,
        'segmentation': [points.ravel().tolist()],
        'area': area,
        'bbox': [xmin, ymin, xmax-xmin, ymax-ymin],
        'iscrowd': 0
      })
      ann_id += 1
  with open(DATA_ROOT/f'{d}.json', 'w') as f:
    json.dump({
      'images':images, 
      'categories': categories, 
      'annotations': annotations
    }, f)