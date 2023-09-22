# Polyp Segmentation Neural Network using Residual U-Net

This repository presents a polyp segmentation neural network implemented using the Residual U-Net architecture. The main objective of this project is to accurately segment polyps in the Kvasir-SEG dataset.

To achieve this goal, we implemented the Residual U-Net model based on the paper titled "Road Extraction by Deep Residual U-Net" (link: [PDF](https://arxiv.org/pdf/1711.10684.pdf)). The Residual U-Net architecture is a modified version of the U-Net model that incorporates residual connections. These connections enable the network to effectively capture both local and global features, resulting in improved segmentation performance. The Residual U-Net architecture has proven successful in various segmentation tasks.

![Residual U-Net architecture](https://github.com/zuanki/PolypSegmentation/blob/main/assets/ResUNet.png)

## Dataset

The Kvasir-SEG dataset, which provides annotated images of the gastrointestinal tract for polyp segmentation, is utilized in this project. The dataset is publicly available and can be obtained from the [Kvasir-SEG dataset](https://datasets.simula.no/kvasir-seg/).

## Results

The trained model achieved a Dice score of 0.65 and an Intersection over Union (IoU) score of 0.51. These metrics are commonly used to evaluate the accuracy of segmentation models. A higher Dice score and IoU score indicate better segmentation performance.

![Streamlit app](https://github.com/zuanki/PolypSegmentation/blob/main/assets/App.png)

Please refer to the original repository for further details and implementation code.