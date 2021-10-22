import os
import random
from random import Random

import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets as dset
from torchvision.transforms.transforms import ToTensor
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset.reconstruct import prepare_data


def train_validation_loader(train_dir, val_dir, batch_size, augment, candi, shuffle, seed, num_workers):

    train_dataset = dset.ImageFolder(train_dir)
    train_dataset = OmniglotTrain(train_dataset, augment)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    val_dataset = dset.ImageFolder(val_dir)
    val_dataset = OmniglotTest(val_dataset, candi, seed)
    val_loader = DataLoader(val_dataset, batch_size=candi, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


def testset_loader(test_dir, candi, seed, num_workers):
    
    test_dataset = dset.ImageFolder(test_dir)
    test_dataset = OmniglotTest(test_dataset, candi=candi, seed=seed)
    test_loader = DataLoader(test_dataset, batch_size=candi, shuffle=False, num_workers=num_workers)

    return test_loader


# adapted from https://github.com/fangpin/siamese-network
class OmniglotTrain(Dataset):
    def __init__(self, dataset, augment=False):
        self.dataset = dataset
        self.augment = augment
        self.mean = 0.8444
        self.std = 0.5329

    def __len__(self):
        return len(self.dataset.imgs)

    def __getitem__(self, index):

        # get image from same class
        if index % 2 == 1:

            label = 1.0
            cls = random.randint(0, len(self.dataset.classes) - 1)

            imgs = [x for x in self.dataset.imgs if x[1] == cls]

            img1 = random.choice(imgs)
            img2 = random.choice(imgs)
            
            while img1[1] != img2[1]:
                img2 = random.choice(imgs)

        # get image from different class
        else:
            label = 0.0

            img1 = random.choice(self.dataset.imgs)
            img2 = random.choice(self.dataset.imgs)

            while img1[1] == img2[1]:
                img2 = random.choice(self.dataset.imgs)

        # apply transformation
        if self.augment:
            trans = A.Compose([
                A.Rotate((-15, 15), p=1),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ])
        else:
            trans = A.Compose([
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ])

        image1 = cv2.imread(img1[0], cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(img2[0], cv2.IMREAD_GRAYSCALE)

        image1 = trans(image=image1)['image']
        image2 = trans(image=image2)['image']

        label = torch.from_numpy(np.array([label], dtype=np.float32))

        return image1, image2, label


class OmniglotTest:
    def __init__(self, dataset, candi, seed=0):
        self.dataset = dataset
        self.candi = candi
        self.seed = seed
        self.img1 = None
        self.mean = 0.8444
        self.std = 0.5329

    def __len__(self):
        return len(self.dataset.imgs)

    def __getitem__(self, index):
        rand = Random(self.seed + index)

        # get image pair from same class
        if index % self.candi == 0:

            label = 1.0

            idx = rand.randint(0, len(self.dataset.classes) - 1)

            imgs = [x for x in self.dataset.imgs if x[1] == idx]

            self.img1 = rand.choice(imgs)
            img2 = rand.choice(imgs)

            while self.img1[1] != img2[1]:
                img2 = rand.choice(imgs)

        # get image pair from different class
        else:
            label = 0.0

            img2 = random.choice(self.dataset.imgs)

            while self.img1[1] == img2[1]:
                img2 = random.choice(self.dataset.imgs)

        trans = A.Compose([
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2(),
        ])

        image1 = cv2.imread(self.img1[0], cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(img2[0], cv2.IMREAD_GRAYSCALE)
        image1 = trans(image=image1)['image']
        image2 = trans(image=image2)['image']

        label = torch.from_numpy(np.array([label], dtype=np.float32))
        return image1, image2, label