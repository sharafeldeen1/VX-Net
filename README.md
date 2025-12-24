### [3D VX-Net](https://doi.org/10.1109/ICIP55913.2025.11084276)
This repository contains the official PyTorch implementation of 3D VX-Net, validated for segmenting pulmonary regions in 3D computed tomography (CT) scans, as described in the paper: [A Novel Automated System for Pathological Lung Segmentation Using Modified Local Binary Patterns and Hierarchical Transformers](https://doi.org/10.1109/ICIP55913.2025.11084276)

This study introduced a novel lung segmentation system by proposing a modified local binary pattern, called Cylinder Binary Pattern (CBP), which was concatenated with the 3D CT volume and fed into a novel network called VX-Net. This network combines the V-Net architecture with the capability of hierarchical transformers to improve segmentation accuracy and robustness.

The network is highly configurable and accepts several input arguments to customize its architecture and behavior:


```python
from VXNet.vx_net import VXNet

model = VXNet(
        in_chans=3, 
        num_classes=2, 
        dims=[32, 64, 128], 
        depths=[2, 2, 2], 
        encoder_nConv=[2, 3, 2], 
        decoder_nConv=[1, 1, 2], 
        dropout=[False, True, True], 
        depth_reduction=False, 
        elu=True, 
        nll=False
    )
```

| Parameter         | Type         | Default             | Description                                                                           |
| ----------------- | ------------ | ------------------- | ------------------------------------------------------------------------------------- |
| `in_chans`        | int          | 3                   | Number of input image channels (e.g., 1 for grayscale CT, 3 for multi-channel inputs) |
| `num_classes`     | int          | 2                   | Number of segmentation classes (e.g., background vs. lung)                            |
| `dims`            | list of int  | [32, 64, 128]       | Feature dimensions at each stage of the network                                       |
| `depths`          | list of int  | [2, 2, 2]           | Number of blocks at each stage                                                        |
| `encoder_nConv`   | list of int  | [2, 3, 2]           | Number of convolutional layers per stage in the encoder                               |
| `decoder_nConv`   | list of int  | [1, 1, 2]           | Number of convolutional layers per stage in the decoder                               |
| `dropout`         | list of bool | [False, True, True] | Dropout applied at each stage for regularization                                      |
| `depth_reduction` | bool         | False               | Downsamples the depth dimension if `True`                                             |
| `elu`             | bool         | True                | Use ELU activation if `True`, PReLU if `False`                                        |
| `nll`             | bool         | False               | Use log softmax output if `True`, softmax if `False`                                  |

**Example**
```python
from VXNet.vx_net import VXNet
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.randn(1, 2, 50, 512, 512)

model = VXNet(in_chans=2,elu=False, nll=True).to(device)
y=model(x)

print(y.shape)

```
## Installation / Required Packages
To run the VX-Net implementation, you need the following Python packages:
```bash
# PyTorch
pip install torch torchvision torchaudio

# TIMM 
pip install timm

```

## Results
The proposed system and network were evaluated on patients with varying COVID-19 severity and compared with different state-of-the-art segmentation methods, as summarized in the following table:


| Model           | DSC (%)       | IoU (%)       | AVD          | HD          |
|-----------------|---------------|---------------|--------------|-------------|
| nnFormer        | 87.2 ± 10.3   | 78.6 ± 14.8   | 17.8 ± 15.6  | 14.2 ± 14   |
| Swin-Unet       | 92.2 ± 6.8    | 86.2 ± 10.6   | 11 ± 10.7    | 14.8 ± 17.9 |
| 3D U-Net        | 93.4 ± 4.5    | 87.9 ± 7.3    | 10.3 ± 12    | 33.6 ± 41   |
| UX-Net          | 94.9 ± 2.8    | 90.5 ± 4.8    | 4.7 ± 4.9    | 10.5 ± 9.3  |
| 2D U-Net        | 95 ± 1.7      | 90.5 ± 3.1    | 6.6 ± 3.1    | 15.4 ± 23.4 |
| 3D V-Net        | 95.1 ± 7.1    | 91.4 ± 10.4   | 5.9 ± 11     | 8.2 ± 8     |
| **VX-Net**      | 96.6 ± 1.5    | 93.4 ± 2.8    | 3.5 ± 3      | 4.1 ± 1.4   |
| **Our System**  | 97.6 ± 1      | 95.4 ± 1.9    | 2 ± 1.9      | 3 ± 1.2     |

**Acronyms:**  
- DSC: Dice Similarity Coefficient  
- IoU: Intersection over Union  
- AVD: Absolute Volume Distance  
- HD: Hausdorff Distance
  
## Citation
If you find this repository helpful, please consider citing:
```
@inproceedings{sharafeldeen2025novel,
  title={A Novel Automated System for Pathological Lung Segmentation Using Modified Local Binary Patterns and Hierarchical Transformers},
  author={Sharafeldeen, Ahmed and Taher, Fatma and Ghazal, Mohammed and Khalil, Ashraf and Mahmoud, Ali and Contractor, Sohail and El-Baz, Ayman},
  booktitle={2025 IEEE International Conference on Image Processing (ICIP)},
  pages={1642--1647},
  year={2025},
  organization={IEEE}
}
```
