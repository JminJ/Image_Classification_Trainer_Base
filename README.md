# Image_Classification_Base_Code
## Intro
Boilerplate code for Image Classification task. (**using EfficientNet-B2**)
## Model Train
### using model
* [Ross Wightman/EfficientNet-B2](https://github.com/rwightman/pytorch-image-models)

### how to train?
```BASH
python codes/train.py \
    --learning_rate=5e-05 \
    --train_dataset_path=TRAIN_DATASET_PATH \
    --valid_dataset_path=VALID_DATASET_PATH \
    --device=DEVICE \
    --train_batch_size=64 \
    --valid_batch_size=32 \
    --epoch=5
```

### parameters
| parameter | type | description | default |
| ---------- | ---------- | ---------- | --------- |
| dropout_percent | float | value for dropout percent | 0.25 |
| use_loss_weight | bool | decising apply loss_weight | False |
| use_weight_sampler | bool | decising apply weight_sampler | False |
| train_dataset_path | str | train dataset path | - |
| valid_dataset_path | str | valid dataset path | - |
| use_custom_normalize | bool | decising apply temp data-only normalize | False |
| train_batch_size | int | batch size for model train | 64 |
| valid_batch_size | int | batch size for model valid | 32 |
| learning_rate | float | decise learning rate for train | 5e-05 |
| lr_warmup_rate | float | percent for learning-rate warmup | 0.1 |
| lr_warmup_steps | int | steps for learning-rate warmup | None |
| weight_decay_rate | float | percent for weight_decay | 0.0 |
| epoch | int | epoch use in training | 5 |
| device | str | device use in training | "cuda" |
| base_checkpoint_path | str | EfficientNet-B2 base model checkpoint path | None |
| wandb_name | str | A name using for Wandb project | "Image_Classification_Project" |
```
NOTE) 
1. Only one of "use_loss_weight" and "use_weight_sampler" can be True.
2. if base_checkpoint_path is None, code will loading EfficientNet-B2 from "timm" library.
```
</br>

## Contact
* jminju254@gmail.com
## Reference
* [Ross Wightman/EfficientNet-B2](https://github.com/rwightman/pytorch-image-models)
* [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)