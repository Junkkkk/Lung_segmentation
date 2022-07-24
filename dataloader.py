import torch
import torchvision

import pandas as pd
import numpy as np

from PIL import Image


class LungDataset(torch.utils.data.Dataset):
    def __init__(self, config, mode, img_list, mask_list, transforms=None):
        self.config = config
        self.img_list = img_list[mode]
        self.mask_list = mask_list[mode]
        self.transforms = transforms

    def __getitem__(self, idx):
        img_batch = self.img_list[idx]
        mask_batch = self.mask_list[idx]

        img = Image.open(img_batch).convert('P')
        mask = Image.open(mask_batch).convert('P')

        img = torchvision.transforms.functional.to_tensor(img)
        mask = np.array(mask)
        mask = np.expand_dims(mask, axis=0)
        mask = (torch.tensor(mask) > 128).long()

        # resize = Resize(output_size=self.config.resize_dim)
        # img, mask = resize((img, mask))

        if self.transforms is not None:
            img = self.transforms(img)
            mask = self.transforms(mask)

        return img, mask, len(self.img_list)

        #return torch.unsqueeze(img, 0), torch.unsqueeze(mask, 0)

    def __len__(self):
        return len(self.img_list)