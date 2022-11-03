import os
import wandb
import torch
import torch.nn as nn
import argparse
import numpy as np
import pandas as pd
from typing import List, Tuple
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import get_cosine_schedule_with_warmup
from transformers import AdamW
from sklearn.metrics import f1_score

from image_dataset import ImageDataset
from image_classifier import ImageClassifier

class BaseOperation:
    def __init__(self, loss_function:nn.modules.loss, drop_p:float, device:str, base_ckpt_path: str=None):
        self.image_sensitive_classifier = ImageClassifier(drop_p, base_model_ckpt=base_ckpt_path)
        self.image_sensitive_classifier = self.image_sensitive_classifier.to(device)
        self.loss_function = loss_function
        
    def calc_loss(self, classifier_result:torch.Tensor, label:torch.Tensor)->torch.Tensor:
        # print(f"classifier_result : {classifier_result}")
        # print(f"\t{classifier_result[0][0]}")
        # print(f"\t{type(classifier_result[0][0])}")
        # print(f"label : {label}")
        # print("----------")
        # print(self.loss_function)
        # print("----------")
        temp_step_loss = self.loss_function(classifier_result, label)

        return temp_step_loss

    def get_collect_result(self, classifier_result:torch.Tensor, label:torch.Tensor)->List:
        result_max_index = torch.argmax(classifier_result, dim = 1)
        temp_correct_result_list = [False] * len(label)
        
        for i in range(len(label)):
            temp_label_val = label[i].item()
            temp_result_val = result_max_index[i].item()

            if temp_label_val == temp_result_val:
                temp_correct_result_list[i] = True

        return temp_correct_result_list

    def calc_acc(self, classifier_result:torch.Tensor, label:torch.Tensor)->torch.Tensor:
        y_pred = torch.argmax(classifier_result, dim=1)
        return (y_pred == label).to(torch.float).mean()

    def calc_f1_score(self, classifier_result:torch.Tensor, label:torch.Tensor)->Tuple[float, List]:
        label = label.detach().cpu().numpy() # numpy로 변경
        
        y_pred = torch.argmax(classifier_result, dim=1).detach().cpu().numpy() # numpy로 변경
        step_default_f1_score = f1_score(label, y_pred, average="weighted")
        step_class_f1_score = f1_score(label, y_pred, average=None)

        return step_default_f1_score, step_class_f1_score

    def operation_func(self, input_batch)->Tuple[torch.Tensor, torch.Tensor, List]:
        labels = input_batch["label"]
        image_tensors = input_batch["image_tensor"]

        classifier_result = self.image_sensitive_classifier(image_tensors)

        temp_step_loss = self.calc_loss(classifier_result=classifier_result, label=labels)
        temp_step_acc = self.calc_acc(classifier_result=classifier_result, label=labels)
        temp_step_collect_list = self.get_collect_result(classifier_result=classifier_result, label=labels)
        temp_d_f1_score, temp_c_f1_score = self.calc_f1_score(classifier_result=classifier_result, label=labels)

        return temp_step_loss, temp_step_acc, temp_step_collect_list, temp_d_f1_score, temp_c_f1_score

'''
    train_arguments:
        - dropout_percent(float)=0.25
        - use_loss_weight(bool)=False -> 아래와 상반
        - use_weight_sampler(bool)=False -> 위와 상반
        - train_dataset_path(str)
        - valid_dataset_path(str)
        - use_custom_normalize(bool) = False => 제거해야 할수(너무 오래 걸림)
        - train_batch_size(int) = 64
        - valid_batch_size(int) = 32  
        - learning_rate(float) = 5e-05 
        - lr_warmup_rate(float) = 0.1 
        - lr_warmup_steps(int) = False
        - epochs(int) = 5
        - weight_decay_rate(float) = False
        - device(str) = cuda
'''
class ImageTrainer:
    def __init__(self, train_arguments:argparse.Namespace, model_save_path:str):
        self.train_arguments = train_arguments
        self.pd_train_set = pd.read_csv(train_arguments.train_dataset_path, sep = "\t")
        self.pd_valid_set = pd.read_csv(train_arguments.valid_dataset_path, sep = "\t")
        self.model_save_path = model_save_path

        self.device = train_arguments.device
        self.learning_rate = train_arguments.learning_rate
        self.epochs = train_arguments.epochs

        self.loss_function = self.get_loss_func()
        self.classifier_operation = BaseOperation(loss_function = self.loss_function, drop_p = self.train_arguments.dropout_percent, device="cuda", base_ckpt_path=train_arguments.base_checkpoint_path)

        self.transform = self.get_transforms()
        self.train_dataloader, self.valid_dataloader = self.get_dataloaders()
        self.optimizer = self.get_optimizer(self.learning_rate, train_arguments.weight_decay_rate)
        self.lr_scheduler = self.get_lr_scheduler(step_num=train_arguments.lr_warmup_steps)

        self.wandb_init()

    def wandb_init(self):
        wandb_init_args = {
            "learning_rate" : self.train_arguments.learning_rate,
            "train_batch_size" : self.train_arguments.train_batch_size,
            "valid_batch_size" : self.train_arguments.valid_batch_size,
            "use_custom_normalize" : self.train_arguments.use_custom_normalize,
            "use_weight_sampler" : self.train_arguments.use_weight_sampler,
            "use_loss_weight" : self.train_arguments.use_loss_weight,
            "lr_warmup_rate" : self.train_arguments.lr_warmup_rate,
            "lr_warmup_steps" : self.train_arguments.lr_warmup_steps,
            "weight_decay_rate" : self.train_arguments.weight_decay_rate,
            "model_save_dir" : self.model_save_path
        }
        wandb.init(project = "Mindlogic Image Sensitive Model 221031", 
                    config=wandb_init_args)

        wandb.watch(self.classifier_operation.image_sensitive_classifier)

    def calc_class_f1_score(self, class_f1_score_list:List, base_class_f1_score:List, batch_label_convert_dict:dict, each_class_apperance:List)->Tuple[List, List]:
        # print(f"\tclass_f1_score_list : {class_f1_score_list}")

        for i in range(len(class_f1_score_list)):
            if class_f1_score_list[i] == 0:
                continue
            temp_key = str(i)
            each_class_apperance[batch_label_convert_dict[temp_key]] += 1
            base_class_f1_score[batch_label_convert_dict[temp_key]] += class_f1_score_list[i]
            base_class_f1_score[batch_label_convert_dict[temp_key]] /= each_class_apperance[batch_label_convert_dict[temp_key]]

        return base_class_f1_score, each_class_apperance

    def train(self):
        self.classifier_operation.image_sensitive_classifier.train()

        train_all_loss = 0
        train_all_acc = 0
        train_all_default_f1_score = 0
        train_all_class_f1_score = [0, 0, 0]
        train_each_class_appearance = [0, 0, 0] # 각 class 별 f1_score 구하기 위한 값.
        train_all_steps = 0
        train_all_examples = 0


        for e in range(self.epochs):
            print(f"========== Start {e} Epoch ==========")
            for _, batch in enumerate(self.train_dataloader, 0):
                step_loss, step_acc, collect_list, step_d_f1_score, step_c_f1_score = self.classifier_operation.operation_func(batch)
                temp_labels = batch["label"]
                temp_label_convert_dict = self.check_label_unique(temp_labels)

                train_all_loss += step_loss.item()
                train_all_acc += step_acc.item()
                train_all_default_f1_score += step_d_f1_score
                train_all_steps += 1
                train_all_examples += len(batch["label"])
                train_all_class_f1_score, train_each_class_appearance = self.calc_class_f1_score(step_c_f1_score, train_all_class_f1_score, temp_label_convert_dict, each_class_apperance=train_each_class_appearance)

                train_loss_mean = train_all_loss / train_all_steps
                train_acc_mean = train_all_acc / train_all_steps # -> 이미 .mean()으로 구하므로
                train_default_f1_score = train_all_default_f1_score / train_all_steps

                wandb.log({
                    "train_loss_mean" : train_loss_mean,
                    "train_acc_mean" : train_acc_mean,
                    "train_default_f1_score" : train_default_f1_score
                })
            
                ## update
                self.optimizer.zero_grad()
                step_loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
            
            self.valid(epoch=e)
            self.save_model(epoch=e)

    def valid(self, epoch):
                
        self.classifier_operation.image_sensitive_classifier.eval()
        
        valid_all_loss = 0
        valid_all_acc = 0
        valid_all_steps = 0
        valid_all_examples = 0
        valid_all_default_f1_score = 0
        valid_all_class_f1_score = [0, 0, 0]
        valid_each_class_appearance = [0, 0, 0] # 각 class 별 f1_score 구하기 위한 값.

        for _, batch in enumerate(self.valid_dataloader, 0):
            with torch.no_grad():
                step_loss, step_acc, collect_list, step_d_f1_score, step_c_f1_score = self.classifier_operation.operation_func(batch)
                temp_labels = batch["label"]
                temp_label_convert_dict = self.check_label_unique(temp_labels)

                valid_all_loss += step_loss.item()
                valid_all_acc += step_acc.item()
                valid_all_default_f1_score += step_d_f1_score
                valid_all_steps += 1
                valid_all_examples += len(batch["label"])
                # valid_all_class_f1_score = [valid_all_class_f1_score[i]+step_c_f1_score[i] for i in range(len(step_c_f1_score))]
                valid_all_class_f1_score, valid_each_class_appearance = self.calc_class_f1_score(step_c_f1_score, valid_all_class_f1_score, temp_label_convert_dict, each_class_apperance=valid_each_class_appearance)

        valid_loss_mean = valid_all_loss / valid_all_steps
        valid_acc_mean = valid_all_acc / valid_all_steps
        valid_default_f1_score_mean = valid_all_default_f1_score / valid_all_steps

        print(f"=========== Valid {epoch} Epoch Relsult ==========")
        print(f"valid_loss_mean : {valid_loss_mean}")
        print(f"valid_acc_mean : {valid_acc_mean}")
        print(f"valid_default_f1_score_mean : {valid_default_f1_score_mean}")
        print(f"\nvalid_class_f1_score_mean : {valid_all_class_f1_score}") # 함수에서 평균까지 구해서 반환함

        wandb.log({
            "valid_loss_mean" : valid_loss_mean,
            "valid_acc_mean" : valid_acc_mean,
            "valid_default_f1_score_mean" : valid_default_f1_score_mean
        })

    def save_model(self, epoch):
        model_save_dir_path = self.model_save_path
        torch.save({
            "epoch" : epoch,
            "model_state_dict" : self.classifier_operation.image_sensitive_classifier.state_dict(),
            "optimizer_state_dict" : self.optimizer.state_dict()
        }, os.path.join(model_save_dir_path, f"image_sensitive_{epoch}.pt"))

        print(f"Epoch {epoch} Model Saved.")


    # Initialize Functions
    def get_dataloaders(self):
        train_Dataset = ImageDataset(self.pd_train_set, self.transform, self.device)
        valid_Dataset = ImageDataset(self.pd_valid_set, self.transform, self.device)

        if self.train_arguments.use_weight_sampler:
            dataloader_args = {
                "num_workers" : 0, 
                "shuffle" : False
            }

            train_sampler, valid_sampler = self.get_weight_sampler()
            train_DataLoader=DataLoader(
                            train_Dataset,
                            batch_size=self.train_arguments.train_batch_size,
                            sampler=train_sampler,
                            **dataloader_args
                        )
            valid_DataLoader=DataLoader(
                            valid_Dataset,
                            batch_size=self.train_arguments.valid_batch_size,
                            sampler=valid_sampler,
                            **dataloader_args
                        )
        else:
            dataloader_args = {
                "num_workers" : 0, 
                "shuffle" : True
            }

            train_DataLoader=DataLoader(
                            train_Dataset,
                            batch_size=self.train_arguments.train_batch_size,
                            **dataloader_args
                        )
            valid_DataLoader=DataLoader(
                            valid_Dataset,
                            batch_size=self.train_arguments.valid_batch_size,
                            **dataloader_args
                        )

        return train_DataLoader, valid_DataLoader



    def get_optimizer(self, learning_rate:float, weight_decay_rate:float)->AdamW:
        if weight_decay_rate == False:
            optimizer = AdamW(params= self.classifier_operation.image_sensitive_classifier.parameters(), lr=learning_rate)
        else:
            optimizer = AdamW(params= self.classifier_operation.image_sensitive_classifier.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)

        return optimizer



    def get_normalize_mean_std(self)->Tuple[List, List]:
        def get_full_image_paths(train_dataset:pd.DataFrame, valid_dataset:pd.DataFrame)->List:
            train_image_path = list(train_dataset.loc[:, "image_path"])
            valid_image_path = list(valid_dataset.loc[:, "image_path"])

            all_image_path = train_image_path + valid_image_path

            return all_image_path

        def calc_custom_norm_mean(image_all_list:List)->List:
            meanRGB = np.array([np.mean(x.numpy(), axis=(1,2)) for x in image_all_list])
            
            meanR = np.mean([m[0] for m in meanRGB])
            meanG = np.mean([m[1] for m in meanRGB])
            meanB = np.mean([m[2] for m in meanRGB])

            return [meanR, meanG, meanB]

        def calc_custom_norm_std(image_all_list:List)->List:
            stdRGB = np.array([np.std(x.numpy(), axis=(1,2)) for x in image_all_list])

            stdR = np.mean([s[0] for s in stdRGB])
            stdG = np.mean([s[1] for s in stdRGB])
            stdB = np.mean([s[2] for s in stdRGB])

            return  [stdR, stdG, stdB]

        if self.train_arguments.use_custom_normalize:
            train_dataset = self.pd_train_set
            valid_dataset = self.pd_valid_set

            all_image_path_list = get_full_image_paths(train_dataset, valid_dataset)
            make_tensor_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                    ])
            all_image_tensor_list = [make_tensor_transform(Image.open(p).convert("RGB")) for p in all_image_path_list]
            
            custom_norm_mean = calc_custom_norm_mean(image_all_list=all_image_tensor_list)
            custom_norm_std = calc_custom_norm_std(image_all_list=all_image_tensor_list)

            return custom_norm_mean, custom_norm_std
        else:
            default_norm_mean = [0.485, 0.456, 0.406]
            default_norm_std = [0.229, 0.224, 0.225]

            return default_norm_mean, default_norm_std
            
    def get_transforms(self)->transforms.Compose:
        normalize_mean, normalize_std = self.get_normalize_mean_std()

        transform = transforms.Compose([
            transforms.Resize([260, 260]),
            transforms.ToTensor(),
            transforms.Normalize(mean = normalize_mean, std = normalize_std)
        ])
        
        return transform



    def count_label_counts(self, target_dataset:pd.DataFrame)->List:
        data_labels = list(target_dataset.loc[:, "label"])
        label_uniques, label_counts = np.unique(data_labels, return_counts=True)

        return label_counts

    def calc_loss_weight(self, target_dataset:pd.DataFrame)->torch.Tensor:
        target_label_counts = self.count_label_counts(target_dataset=target_dataset)
        label_weight_float = [max(target_label_counts)/c for c in target_label_counts]
        # make float
        label_weight = torch.tensor(label_weight_float).float()
        print(f"\tlabel weight : {label_weight}")

        return label_weight

    def calc_sampler_weight(self, target_dataset:pd.DataFrame)->torch.Tensor:
        target_label_counts = self.count_label_counts(target_dataset=target_dataset)

        class_weight = [sum(target_label_counts) / c for c in target_label_counts]
        target_label_weight = [class_weight[int(w)] for w in list(target_dataset.loc[:, "label"])]

        return target_label_weight

    def get_loss_func(self)->nn.modules.loss.CrossEntropyLoss:
        if self.train_arguments.use_loss_weight:
            if self.train_arguments.use_weight_sampler:
                raise AttributeError("if 'use_loss_weight' is True, 'use_weight_sampler' should be False.")
            
            label_weight = self.calc_loss_weight(self.pd_train_set)
            label_weight = label_weight.to(self.device)
            loss_func = nn.CrossEntropyLoss(weight=label_weight)

        else:
            loss_func = nn.CrossEntropyLoss()

        return loss_func

    def get_weight_sampler(self)->Tuple[WeightedRandomSampler, WeightedRandomSampler]:
        if self.train_arguments.use_loss_weight:
            raise AttributeError("if 'use_weight_sampler' is True, 'use_loss_weight' should be False.")
            
        train_label_weight = self.calc_sampler_weight(self.pd_train_set)
        valid_label_weight = self.calc_sampler_weight(self.pd_valid_set)

        train_weight_sampler = WeightedRandomSampler(train_label_weight, num_samples=len(self.pd_train_set), replacement=True)
        valid_weight_sampler = WeightedRandomSampler(valid_label_weight, num_samples=len(self.pd_valid_set), replacement=True)

        return train_weight_sampler, valid_weight_sampler


    
    def calc_warmup_steps(self, warmup_rate:float)->Tuple[int, int]:
        full_train_steps = int(np.ceil(len(self.pd_train_set) / self.train_arguments.train_batch_size) * self.train_arguments.epochs)
        warmup_steps = int(round(full_train_steps * warmup_rate))

        return full_train_steps, warmup_steps

    def get_lr_scheduler(self, step_num=None):
        full_train_steps, warmup_steps = self.calc_warmup_steps(self.train_arguments.lr_warmup_rate)

        if step_num == None:
            lr_scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=full_train_steps)
        else:
            lr_scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=step_num, num_training_steps=full_train_steps)

        return lr_scheduler