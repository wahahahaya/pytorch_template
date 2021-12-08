import glob
import os

from torch.utils.data import Dataset
import cv2

import numpy as np
from imgaug import augmenters as iaa

class CustomDataset(Dataset):
    def __init__(self, root, transform=None, mode="train"):
        self.files_img = sorted(glob.glob(os.path.join(root, "images")+"/*.*"))
        self.files_tooth = sorted(glob.glob(os.path.join(root, "masks")+"/*TOOTH.*"))

        self.transform = transform
        self.mode = mode

    def __getitem__(self, index):
        img_row = np.expand_dims(np.expand_dims(cv2.imread(self.files_img[index % len(self.files_img)], cv2.IMREAD_GRAYSCALE),2),0)
        img_tooth = np.expand_dims(np.expand_dims(cv2.imread(self.files_tooth[index % len(self.files_tooth)], cv2.IMREAD_GRAYSCALE),2),0)

        if self.mode=="train":
            imgaug = iaa.Sequential([
                iaa.LinearContrast(alpha=(0.5, 1.5)),
                iaa.Multiply(mul=(0.7, 1.3)),
                iaa.Affine(scale=(0.8, 1.2), rotate=(-10, 10), shear=(-10, 10)),
                iaa.ElasticTransformation(alpha=(0, 100), sigma=10),
                iaa.GaussianBlur(sigma=(0.0, 5.0)),
                iaa.Cutout(size=0.1), # without segmentation map
                iaa.Resize(size={'height': 192, 'width': 512}),
                iaa.PadToFixedSize(height=192, width=512),
                iaa.CropToFixedSize(height=192, width=512),
            ])

        if self.mode=="test":
            imgaug = iaa.Sequential([
                iaa.Resize(size={'height': 192, 'width': 512}),
            ])

                
        seg_map = img_tooth

        aug_row, aug_seg = imgaug(images=img_row, segmentation_maps=seg_map) # output: [list]
        aug_row = np.array(aug_row, dtype="uint8").squeeze()
        aug_seg = np.array(aug_seg, dtype="uint8").squeeze()

        aug_tooth = aug_seg


        item_row = self.transform(aug_row)
        item_tooth = self.transform(aug_tooth)


        return {"row": item_row, "tooth": item_tooth}

    def __len__(self):
        return len(self.files_img)