# Polyp Segmentation Neural Network using Residual U-Net

This repository contains a neural network for polyp segmentation using the Residual U-Net architecture. The goal of this project is to accurately segment polyps from the Kvasir-SEG dataset.

## Dataset

The Kvasir-SEG dataset is utilized in this project. It provides annotated images of the gastrointestinal tract for polyp segmentation. The dataset is publicly available and can be obtained from [Kvasir-SEG dataset](https://datasets.simula.no/kvasir-seg/).

## Method

The Residual U-Net architecture is employed for polyp segmentation in this project. Residual U-Net is a variant of the U-Net model that incorporates residual connections, enabling the network to effectively capture both local and global features. The architecture has been proven to be successful in various segmentation tasks.

## Results

The trained model achieved a Dice score of 0.65 and an Intersection over Union (IoU) score of 0.51. These metrics are commonly used to evaluate the accuracy of segmentation models. A higher Dice score and IoU score indicate better segmentation performance.