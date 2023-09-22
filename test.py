from src.datasets.polyp_dataset import PolypDataset
from src.models.res_unet import ResUNet

import torch
import matplotlib.pyplot as plt


def test():
    # Create dataset
    dataset = PolypDataset('data')
    print("Dataset length: ", len(dataset))  # "Dataset length:  100
    print("Image shape: ", dataset[0]['image'].shape)
    print("Mask shape: ", dataset[0]['mask'].shape)

    # Create model
    model = ResUNet(channel=3)
    # Load model
    model.load_state_dict(torch.load('checkpoints/res_unet_epoch_10.pth'))

    output = model(dataset[0]['image'].unsqueeze(0))

    # Plot image, mask and output
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(dataset[0]['image'].permute(1, 2, 0))
    ax[0].set_title('Image')
    ax[1].imshow(dataset[0]['mask'].squeeze(), cmap='gray')
    ax[1].set_title('Mask')
    ax[2].imshow(output.squeeze().detach().numpy(), cmap='gray')
    ax[2].set_title('Output')
    plt.show()


if __name__ == '__main__':
    test()
