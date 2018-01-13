import os
from glob import glob

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image


class ImageDir(Dataset):
    r"""Generic Dataset for an image classification task

    Args:
        image_glob: regular expression that matches all files needed
        label_fun: lambda function that maps names to class numbers
        img_trans: predefined torchvision transform that's used for converting Image to Tensor


    __getitem__ calls return three objects: image tensor, class label as LongTensor and a full path to file

    """

    def __init__(self, image_glob, label_fun=None, img_trans=None):
        self.imgs = glob(image_glob)
        if label_fun is not None:
            self.label_fun = label_fun  # a lambda that maps filename to the class number
        else:
            import re
            self.label_fun = lambda x: int(re.findall('\d+', x)[-1])  # the last number in the filename is the label
        if img_trans:
            self.img_trans = img_trans
        else:
            self.img_trans = T.ToTensor()

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        label = self.label_fun(os.path.basename(self.imgs[index]))
        if not isinstance(label, torch.LongTensor):
            label = torch.LongTensor([label]).view(-1)
        return self.img_trans(img), label, self.imgs[index]

    def __len__(self):
        return len(self.imgs)
