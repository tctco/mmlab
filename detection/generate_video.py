from mmdet.apis import init_detector, inference_detector
import matplotlib.pyplot as plt
from functools import reduce
import mmcv
import cv2
import numpy as np

model = init_detector('mask_rcnn_r50_fpn_fp16_1x_balloon.py', './work_dirs/mask_rcnn_r50_fpn_fp16_1x_balloon/latest.pth', device='cuda:0')

video = mmcv.VideoReader('./test_video.mp4')
color_splash = cv2.VideoWriter('color_splash.mp4', int(cv2.VideoWriter_fourcc(*'mp4v')), video.fps, video.resolution)
frame = video.read()
while frame is not None:
  result = inference_detector(model, frame)
  gray = mmcv.bgr2gray(frame)
  splashed = np.repeat(gray[..., None], 3, 2)
  mask = reduce(np.logical_or, result[1][0])
  splashed[mask] = frame[mask]
  color_splash.write(splashed)
  frame = video.read()

color_splash.release()