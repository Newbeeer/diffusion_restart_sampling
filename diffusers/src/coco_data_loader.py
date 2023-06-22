import torch
import torch.utils
import pandas as pd
import os
import open_clip
from PIL import Image
import clip

class text_image_pair(torch.utils.data.Dataset):
    def __init__(self, dir_path, csv_path):
        """

        Args:
            dir_path: the path to the stored images
            file_path:
        """
        self.dir_path = dir_path
        df = pd.read_csv(csv_path)
        self.text_description = df['caption']
        _, _, self.preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k')
        _, self.preprocess2 = clip.load("ViT-L/14", device='cuda')  # RN50x64
        # tokenizer = open_clip.get_tokenizer('ViT-g-14')

    def __len__(self):
        return len(self.text_description)

    def __getitem__(self, idx):

        img_path = os.path.join(self.dir_path, f'{idx}.png')
        raw_image = Image.open(img_path)
        image = self.preprocess(raw_image).squeeze().float()
        image2 = self.preprocess2(raw_image).squeeze().float()
        text = self.text_description[idx]
        return image, image2, text


if __name__ == "__main__":
    # test

    text_image_pair(dir_path='/scratch/ylxu/datasets/coco/subset', csv_path='/scratch/ylxu/datasets/coco/subset.csv')