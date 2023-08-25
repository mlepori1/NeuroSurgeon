# NeuroSurgeon
<p align="center">
    <img src=https://github.com/mlepori1/NeuroSurgeon/assets/25048682/9e96d377-7a65-4441-8492-e6b0c635886f>
</p>

NeuroSurgeon is a python toolkit built to enable deep learning researchers to easily uncover and manipulate subnetworks within trained models. NeuroSurgeon provides a simple API to inject differentiable binary masks techniques into linear, attention, and convolution layers in BERT, GPT, ResNet, and ViT-style models within Huggingface Transformers. Differential masking has a variety of use cases for deep learning research, such as:

- Pruning to uncover functional subnetworks
    - Relevant Papers:
        - [Csordás et al. 2021](https://arxiv.org/abs/2010.02066)
        - [Lepori et al. 2023](https://arxiv.org/abs/2301.10884)
- Subnetwork Probing
    - Relevant Papers:
        - [Cao et al. 2021](https://arxiv.org/abs/2104.03514)
- Training with L0 Regularization
    - Relevant Papers:
        - [Louizos et al. 2018](https://arxiv.org/abs/1712.01312)
        - [Savarese et al. 2020](https://arxiv.org/abs/1912.04427)  


## Install

NeuroSurgeon requires python 3.9 or higher and several libraries including Transformers and Pytorch. Installation can be done using Pypi:

`pip install NeuroSurgeon`


