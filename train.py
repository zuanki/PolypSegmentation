import argparse
import os
import sys
import time
import pandas as pd
import importlib
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from src.datasets.polyp_dataset import PolypDataset
from src.models.res_unet import ResUNet
from src.losses import BCEDiceLoss


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Segmentation Model')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='dataset path including images and masks')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--save_dir', type=str,
                        default='checkpoints', help='directory to save models')

    args = parser.parse_args()

    return args


def main(args):
    DATA_DIR = args.data_dir
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    LR = args.lr
    SAVE_DIR = args.save_dir
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create save directory if it doesn't exist
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Create dataset
    dataset = PolypDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Create model
    model = ResUNet(channel=3).to(DEVICE)

    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # Create loss function
    criterion = BCEDiceLoss()

    # Train model
    model.train()

    for epoch in tqdm(range(NUM_EPOCHS)):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            # Get inputs and labels
            image = data['image'].to(DEVICE)
            mask = data['mask'].to(DEVICE)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(image)
            loss = criterion(outputs, mask)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()

        print(f'Epoch {epoch+1} loss: {running_loss/len(dataloader)}')

    # Save model
    # Add some information to the save directory
    name_to_save = f'{time.strftime("%Y%m%d-%H%M%S")}_res_unet'
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, name_to_save))


if __name__ == '__main__':
    args = parse_args()
    main(args)
