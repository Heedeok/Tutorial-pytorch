import os
import argparse
path= os.path.dirname(os.path.abspath(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms.transforms import ToTensor

from models.model import My_siamese
from utils.utils import increment_path
from pathlib import Path

from tqdm import tqdm
import wandb

import numpy as np
import matplotlib.pyplot as plt

from dataset.reconstruct import prepare_data
from dataset.loader import train_validation_loader
from dataset.loader import testset_loader


def imshow(img,text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    return plt 

def run(model, loss_fn, optimizer, train_loader, val_loader, test_loader, epochs, save_dir):

    best_loss = 0

    for epoch in range(epochs):
        
        # Training
        model.train()
        train_loss = 0

        for i, (img1, img2, label) in tqdm(enumerate(train_loader), desc="[Epoch {}]".format(epoch), total=len(train_loader)):

            output = model(img1.cuda(), img2.cuda())
            loss = loss_fn(output, label.cuda())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss

            if (i+1) % (len(train_loader)//5) == 0 or i == (len(train_loader) -1):
                wandb.log({"Train_loss":train_loss/i})
        
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():         
            for i, (img1, img2, label) in tqdm(enumerate(val_loader), desc="[Epoch {}]".format(epoch), total=len(val_loader)):

                output = model(img1.cuda(), img2.cuda())
                loss = loss_fn(output, label.cuda())

                val_loss += loss

                if (i+1) % (len(val_loader)//5) == 0 or i == (len(val_loader) -1):
                    # concatenated = torch.cat((img1[0],img2[0]),-1)
                    # distance = F.pairwise_distance(output1, output2)

                    # plt = imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}\nLabel : {}'.format(distance[0].item(),str(label[0])))
                    
                    # wandb.log({"Val_loss":val_loss/i, "Example":wandb.Image(plt)})
                    wandb.log({"Val_loss":val_loss/i})
            
            val_loss /= len(val_loader)

        # Model save
        if best_loss == 0 or best_loss > val_loss:
            best_loss = val_loss
        
        if val_loss == best_loss:
            ckpt = {'model':model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'epoch':epoch}
            save = os.path.join(save_dir, './omniglot_cnn_best.pt')
            torch.save(ckpt, save)

        else:
            ckpt = {'model':model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'epoch':epoch}
            save = os.path.join(save_dir, './omniglot_cnn_last.pt')
            torch.save(ckpt, save)

        print("Train loss : {}".format(train_loss))
        print("Val loss : {}".format(val_loss))




def main(opt):

    print("==========[Siamese Network(CNN) - Omniglot]==========")
    print(opt)

    # Wandb setting    
    save_dir = str(increment_path(Path(opt.project)/opt.name))
    wandb.init(project=opt.project, config=opt)
    wandb.run.name=save_dir.split("/")[-1]
    wandb.run.save()
    wandb.config.update(opt)

    # Siamese Network
    net = My_siamese().cuda()
    wandb.watch(net)

    # Loss & Optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)

    # Data Loader
    data_dir = os.path.join(path, "data")
    datasets.Omniglot(data_dir, background=True, download=True, transform=ToTensor())
    datasets.Omniglot(data_dir, background=False, download=True, transform=ToTensor())

    train_dir, val_dir, test_dir = prepare_data(data_dir, opt.seed)
    train_loader, val_loader = train_validation_loader(train_dir, val_dir, opt.batch, opt.augment, opt.candi, opt.shuffle, opt.seed, opt.workers)
    test_loader = testset_loader(test_dir, opt.candi, opt.seed, opt.workers)
    
    run(net, loss_fn, optimizer, train_loader, val_loader, test_loader, opt.epoch, save_dir)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--margin', type=float, default=2.0, help='magin of siamese')
    parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate')
    parser.add_argument('--epoch', type=int, default=50, help='# of epoch')
    parser.add_argument('--batch', type=int, default=16, help='# of batch')
    parser.add_argument('--project', default='tutorial', help='save to project/name')
    parser.add_argument('--name', default='siamese_omniglot_cnn', help='save to project/name')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--augment', action='store_true', help='Data augmentation')
    parser.add_argument('--candi', type=int, default=5)
    parser.add_argument('--shuffle', action='store_false', help='training data shuffle')
    parser.add_argument('--seed', type=int, default=3, help='random seed integer')

    opt = parser.parse_args()
    return opt


if __name__=="__main__":
    opt = parse_opt()
    main(opt)