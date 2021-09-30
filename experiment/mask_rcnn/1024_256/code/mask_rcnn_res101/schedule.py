# optimizer
optimizer = dict(type='Adam', lr=0.0005, weight_decay=0.0001)
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config=dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[9, 11])
    # step=[16, 22])
    
# lr_config = dict(
# policy='cyclic',
# target_ratio=(10, 1e-4),
# cyclic_times=1,
# step_ratio_up=0.4,
# )
# momentum_config = dict(
#     policy='cyclic',
#     target_ratio=(0.85 / 0.95, 1),
#     cyclic_times=1,
#     step_ratio_up=0.4,
# )
runner = dict(type='EpochBasedRunner', max_epochs=12)
