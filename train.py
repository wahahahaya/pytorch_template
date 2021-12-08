import numpy as np
from datetime import datetime
from pathlib import Path
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
from torchvision.utils import make_grid

from util import read_json, write_json
from model import *
from data import CustomDataset

def train():
    # SET DEVICE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # SET TENSORBOARD
    writer = SummaryWriter()
    # LOAD CONFIG
    config = Namespace(**read_json(Path("config.json")))
    # SCALER (reduce memory)
    scaler = torch.cuda.amp.GradScaler()
    # HYPERPARAMETER
    epochs = config.trainer["epoch"]


    # DATA
    root = "../../"+config.data_loader["root"]

    tfs = transforms.Compose([
        transforms.ToTensor()
    ])

    custom_ds = CustomDataset(root, transform=tfs, mode="train")
    train_spilt = config.data_loader["train_split"]
    train_ds, val_ds = data.random_split(custom_ds, [int(len(custom_ds)*train_spilt), len(custom_ds)-(int(len(custom_ds)*train_spilt))])

    train_loader = DataLoader(
        train_ds,
        batch_size=config.data_loader["batch_size"],
        num_workers=config.data_loader["num_workers"],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.data_loader["batch_size"],
        num_workers=config.data_loader["num_workers"],
        pin_memory=True
    )

    # MODEL
    out_channels = config.arch["out_channels"]
    model = eval(config.arch["model"])(out_channels).to(device)
    config.arch["params"] = sum(p.numel() for p in model.parameters())

    # LOSS
    loss_bce = eval(config.trainer["loss"])()

    # OPTIMIZER
    opt = eval(config.trainer["opt"])(model.parameters(), lr=config.trainer["lr"])

    # FUNCTION
    sigmoid = nn.Sigmoid()

    # VAL
    def val(data):
        model.eval()
        with torch.no_grad():
            loss=[]
            for i, batch in enumerate(data):
                input = batch['row'].to(device)
                mask_tooth = batch['tooth'].to(device)

                output = sigmoid(model(input))

                loss_batch = loss_bce(output, mask_tooth)
                loss.append(loss_batch.item())
            loss_mean = np.mean(loss)
        return loss_mean

    # TRAIN
    for epoch in range(epochs):
        model.train()
        loss=[]
        for i, batch in enumerate(train_loader):
            input = batch['row'].to(device)
            mask_tooth = batch['tooth'].to(device)

            output = sigmoid(model(input))

            loss_batch = loss_bce(output, mask_tooth)
            loss.append(loss_batch.item())

            opt.zero_grad()
            scaler.scale(loss_batch).backward()
            scaler.step(opt)
            scaler.update()

        loss_mean = np.mean(loss)
        print("train: {}/{}, loss={:.4f}".format(epoch,epochs,loss_mean))

        img_input = make_grid(input[0])
        img_output = make_grid(output[0])
        img_mask = make_grid(mask_tooth[0])

        writer.add_image("input_img", img_input, epoch)
        writer.add_image("output_img", img_output, epoch)
        writer.add_image("mask_img", img_mask, epoch)
        writer.add_scalar('train_loss', loss_mean, epoch)
        writer.add_scalar('val_loss', val(val_loader), epoch)
            
        if (epoch+1)%10==0:
            torch.save(model.state_dict(), "saved_model/model_%d.pth"%(epoch))
            torch.save(opt.state_dict(), "saved_model/opt_%d.pth"%(epoch))


    config.date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    write_json(vars(config), Path("saved/config.json"))

if __name__ == "__main__":
    train()