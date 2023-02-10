_base_ = ['./mask_rcnn_r50_fpn_fp16_1x_coco.py']

model = dict(
  roi_head=dict(
    bbox_head=dict(num_classes=1),
    mask_head=dict(num_classes=1)
  )
)

data = dict(
  train=dict(
    ann_file='/home/tc/datasets/mouse/annotations/train.json',
    img_prefix='/home/tc/datasets/mouse/images/train/',
    classes=('mouse',)
  ),
  val=dict(
    ann_file='/home/tc/datasets/mouse/annotations/val.json',
    img_prefix='/home/tc/datasets/mouse/images/val/',
    classes=('mouse',)
  ),
  test=dict(
    ann_file='/home/tc/datasets/mouse/annotations/test.json',
    img_prefix='/home/tc/datasets/mouse/images/test/',
    classes=('mouse',)
  )
)

evaluation = dict(interval=1, metric=['bbox', 'segm'], save_best='segm_mAP')

optimizer = dict(type='SGD', lr=0.02/10, momentum=0.9, weight_decay=0.0001)
lr_config = None
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=20)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
load_from = './mask_rcnn_r50_fpn_fp16_1x_coco.pth'