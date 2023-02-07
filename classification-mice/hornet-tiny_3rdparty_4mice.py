model = dict(
    type='ImageClassifier',
    backbone=dict(type='HorNet', arch='tiny', drop_path_rate=0.2),
    head=dict(
        type='LinearClsHead',
        num_classes=4,
        in_channels=512,
        init_cfg=None,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.2, mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.0),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0),
        dict(type='Constant', layer=['LayerScale'], val=1e-06)
    ],
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=4, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=4, prob=0.5)
    ]))


data = dict(
    samples_per_gpu=12,
    workers_per_gpu=2,
    train=dict(
        type='CustomDataset',
        data_prefix='data/mouse-cls/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='SquarePad', pad_val=(104, 116, 124)),
            dict(type='Resize', size=(224, -1)),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='RandAugment',
                policies=[
                    dict(
                        type='Rotate',
                        magnitude_key='angle',
                        magnitude_range=(0, 180)),
                    dict(
                        type='Contrast',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.9)),
                    dict(
                        type='Brightness',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.9)),
                    dict(
                        type='Sharpness',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.9)),
                    dict(
                        type='Shear',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.3),
                        direction='horizontal'),
                    dict(
                        type='Shear',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.3),
                        direction='vertical'),
                    dict(
                        type='Translate',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.45),
                        direction='horizontal'),
                    dict(
                        type='Translate',
                        magnitude_key='magnitude',
                        magnitude_range=(0, 0.45),
                        direction='vertical')
                ],
                num_policies=2,
                total_level=10,
                magnitude_level=9,
                magnitude_std=0.5,
                hparams=dict(pad_val=[104, 116, 124],
                             interpolation='bicubic')),
            dict(
                type='RandomErasing',
                erase_prob=0.25,
                mode='rand',
                min_area_ratio=0.02,
                max_area_ratio=0.3333333333333333,
                fill_color=[103.53, 116.28, 123.675],
                fill_std=[57.375, 57.12, 58.395]),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='CustomDataset',
        data_prefix='data/mouse-cls/test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='SquarePad', pad_val=(104, 116, 124)),
            dict(type='Resize', size=(224, -1)),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        type='CustomDataset',
        data_prefix='data/mouse-cls/test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='SquarePad', pad_val=(104, 116, 124)),
            dict(type='Resize', size=(224, -1)),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
evaluation = dict(interval=1, metric='accuracy', save_best='accuracy_top-1', metric_options={'topk':(1,)}) # 由于总类为4，这里必须指明metric_options
paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys=dict({
        '.absolute_pos_embed': dict(decay_mult=0.0),
        '.relative_position_bias_table': dict(decay_mult=0.0)
    }))
optimizer = dict(
    type='AdamW',
    lr=0.004/10,
    weight_decay=0.05,
    eps=1e-08,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys=dict({
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        })))
optimizer_config = dict(grad_clip=dict(max_norm=100.0))
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=0.01,
    warmup='linear',
    warmup_ratio=0.001,
    warmup_iters=20,
    warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=150)
checkpoint_config = dict(interval=30)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook'), dict(type='MMClsWandbHook', init_kwargs=dict(project='mice-cls')),])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = './hornet-tiny_3rdparty_in1k.pth'
resume_from = None
workflow = [('train', 1)]
custom_hooks = [dict(type='EMAHook', momentum=4e-05, priority='ABOVE_NORMAL')]
