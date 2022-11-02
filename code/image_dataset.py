from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torch

class ImageDataset(Dataset):
    def __init__(self, image_dataframe:pd.DataFrame, transforms, device:str):
        super(ImageDataset, self).__init__()
        self.image_datasets = self.make_label_to_int(image_dataframe)
        self.transforms = transforms
        self.device = device

    def make_label_to_int(self, image_datasets)->pd.DataFrame:
        for l in range(len(image_datasets)):
            temp_label = image_datasets.loc[l, "label"]
            try:
                image_datasets.loc[l, "label"] = int(temp_label)
            except ValueError as E:
                float_temp_label = float(temp_label)
                image_datasets.loc[l, "label"] = int(float_temp_label)

        return image_datasets

    def __len__(self):
        return len(self.image_datasets)

    def __getitem__(self, index):
        temp_image_full_path = self.image_datasets.loc[index, "image_path"]
        temp_image_name = self.image_datasets.loc[index, "file_name"]
        temp_image_label = torch.tensor(self.image_datasets.loc[index, "label"])
        temp_image_label = temp_image_label.to(self.device)
        temp_image_label = temp_image_label.to(torch.int64)

        image = Image.open(temp_image_full_path).convert("RGB")

        transformed_image = self.transforms(image)
        transformed_image = transformed_image.to(self.device) # device로 적재됨

        return {"id" : temp_image_name, "label" : temp_image_label, "image_tensor" : transformed_image}