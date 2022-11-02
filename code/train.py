import os
import time
import argparse
from PIL import ImageFile
from image_classifition_trainer import ImageTrainer

# seed 고정
import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_train_parameters():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d_p", "--dropout_percent", default=0.25, type=float)
    parser.add_argument("-l_w", "--use_loss_weight", default=False, type=bool)
    parser.add_argument("-w_s", "--use_weight_sampler", default=False, type=bool)
    parser.add_argument("-t_p", "--train_dataset_path", type=str)
    parser.add_argument("-v_p", "--valid_dataset_path", type=str)
    parser.add_argument("-c_norm", "--use_custom_normalize", default=False, type=bool)
    parser.add_argument("-tbs", "--train_batch_size", default=64, type=int)
    parser.add_argument("-vbs", "--valid_batch_size", default=32, type=int)
    parser.add_argument("-lr", "--learning_rate", default=5e-05, type=float)
    parser.add_argument("-lr_wr", "--lr_warmup_rate", default=0.1, type=float)
    parser.add_argument("-lr_ws", "--lr_warmup_steps", default=None, type=int)
    parser.add_argument("-e", "--epochs", default=5, type=int)
    parser.add_argument("-w_dr", "--weight_decay_rate", default=0, type=float)
    parser.add_argument("-d", "--device", default="cuda", type=str)
    parser.add_argument("-b_ckpt", "--base_checkpoint_path", default=None, type=str)

    args = parser.parse_args()

    return args

def check_dataset_type(target_dataset_path:str):
    dataset_type = target_dataset_path.split("/")[-2]
    
    return dataset_type

def make_save_dir_name(train_arguments:argparse.Namespace)->str:
    temp_lr = train_arguments.learning_rate
    use_loss_weight = train_arguments.use_loss_weight
    use_weight_sampler = train_arguments.use_weight_sampler
    use_custom_normalize = train_arguments.use_custom_normalize
    weight_decay_rate = train_arguments.weight_decay_rate
    dataset_type = check_dataset_type(train_arguments.train_dataset_path)

    if use_loss_weight and use_weight_sampler:
        raise AttributeError("if 'use_loss_weight' is True, 'use_weight_sampler' should be False.")
    elif not use_loss_weight and not use_weight_sampler:
        if use_custom_normalize:
            save_dir_name = f"image_classification_{dataset_type}_lr_{temp_lr}__custom_norm__wd_rate_{weight_decay_rate}"
        else:
            save_dir_name = f"image_classification_{dataset_type}_lr_{temp_lr}__wd_rate_{weight_decay_rate}"
    elif use_loss_weight:
        if use_custom_normalize:
            save_dir_name = f"image_classification_{dataset_type}_lr_{temp_lr}__loss_weight__custom_norm__wd_rate_{weight_decay_rate}"
        else:
            save_dir_name = f"image_classification_{dataset_type}_lr_{temp_lr}__loss_weight__wd_rate_{weight_decay_rate}"
    else:
        if use_custom_normalize:
            save_dir_name = f"image_classification_{dataset_type}_lr_{temp_lr}__weight_sampler__custom_norm__wd_rate_{weight_decay_rate}"
        else:
            save_dir_name = f"image_classification_{dataset_type}_lr_{temp_lr}__weight_sampler__wd_rate_{weight_decay_rate}"

    return save_dir_name

def make_save_dir(train_arguments:argparse.Namespace, base_save_path:str):
    temp_save_dir_name = os.path.join(base_save_path, make_save_dir_name(train_arguments=train_arguments))

    if not os.path.exists(temp_save_dir_name):
        os.makedirs(temp_save_dir_name)
        print(f"temp_save_path is made.")
    else:
        print("temp_save_dir_path is already existing.")
        make_next_path = input("You want to make 'temp_save_dir_path + _ + n'? :")
        if make_next_path in ["Yes", "yes", "Y", "y"]:
            while True:
                n = 2
                next_path = f"{temp_save_dir_name}_{n}"
                if os.path.exists(next_path):
                    n += 1
                else:
                    os.makedirs(next_path)
                    break
        else:
            print("WARNING!) Your past ckpts will be covered!!")
            print("Program sleep in 5 sec...")
            time.sleep(5)

    return temp_save_dir_name

if __name__ == "__main__":
    train_args = get_train_parameters()
    BASE_MODEL_SAVE_PATH = "" # 학습시킨 모델이 저장될 base 저장소 경로
    model_save_dir_path = make_save_dir(train_arguments=train_args, base_save_path=BASE_MODEL_SAVE_PATH)

    trainer = ImageTrainer(train_arguments=train_args, model_save_path=model_save_dir_path)
    trainer.train()