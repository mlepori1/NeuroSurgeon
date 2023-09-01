# NeuroSurgeon
<p align="center">
    <img src=https://github.com/mlepori1/NeuroSurgeon/assets/25048682/9e96d377-7a65-4441-8492-e6b0c635886f>
</p>

NeuroSurgeon is a python toolkit built to enable deep learning researchers to easily uncover and manipulate subnetworks within trained models. NeuroSurgeon provides a simple API to inject differentiable binary masks techniques into linear, attention, and convolution layers in BERT, GPT, ResNet, and ViT-style models within Huggingface Transformers. Differentiable masking has a variety of use cases for deep learning research, such as:

- Pruning to uncover functional subnetworks
    - Relevant Papers:
        - [Csordás et al. 2021](https://arxiv.org/abs/2010.02066)
        - [Lepori et al. 2023](https://arxiv.org/abs/2301.10884)
        - [Panigrahi et al. 2023](https://arxiv.org/abs/2302.06600)
- Subnetwork Probing
    - Relevant Papers:
        - [Cao et al. 2021](https://arxiv.org/abs/2104.03514)
        - [Conmy et al. 2023](https://arxiv.org/abs/2304.14997)
- Training with L0 Regularization
    - Relevant Papers:
        - [Louizos et al. 2018](https://arxiv.org/abs/1712.01312)
        - [Savarese et al. 2020](https://arxiv.org/abs/1912.04427)
     
## Documentation     
[Read the NeuroSurgeon Documentation](https://neurosurgeon.readthedocs.io/en/latest/)

## Tutorial
To get started with NeuroSurgeon, check out the tutorial [here](https://colab.research.google.com/drive/190_zGHEI8BjPgdxWaRrogFmbgruP6y4l?usp=sharing). This covers the basic workflow for using NeuroSurgeon to uncover functional subnetworks within a trained model.

## Install

NeuroSurgeon requires python 3.9 or higher and several libraries, including Transformers and PyTorch. Installation can be done using PyPi:

`pip install NeuroSurgeon`

### Logo Prompt
NeuroSurgeon's logo was created with the help of DALL-E 2, using the prompt "A cute cartoon robot doctor smiling with a stethoscope". 


