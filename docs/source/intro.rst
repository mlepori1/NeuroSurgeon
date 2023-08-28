Introduction
============

``NeuroSurgeon`` is a python toolkit built to enable deep learning researchers to easily uncover and manipulate subnetworks within trained models. NeuroSurgeon provides a simple API to inject differentiable binary masks techniques into linear, attention, and convolution layers in BERT, GPT, ResNet, and ViT-style models within Huggingface Transformers.
Differentiable masking has several use cases, including:

* Pruning to uncover functional subnetworks
    * Relevant Papers:
        * `Csordás et al. 2021 <https://arxiv.org/abs/2010.02066>`_
        * `Lepori et al. 2023 <https://arxiv.org/abs/2301.10884>`_
        * `Panigrahi et al. 2023 <https://arxiv.org/abs/2302.06600>`_
* Subnetwork Probing
    * Relevant Papers:
        * `Cao et al. 2021 <https://arxiv.org/abs/2104.03514>`_
        * `Conmy et al. 2023 <https://arxiv.org/abs/2304.14997>`_
* Training with L0 Regularization
    * Relevant Papers:
        * `Louizos et al. 2018 <https://arxiv.org/abs/1712.01312>`_
        * `Savarese et al. 2020 <https://arxiv.org/abs/1912.04427>`_

Installation
*************

``NeuroSurgeon`` is tested in python3.9, and requires several packages, such as PyTorch and the Huggingface Transformers libary. Install using pip:
``pip install NeuroSurgeon``

Support
*********
This is a research library, designed to be extended and improved as the field advances. Please report bugs, issues, and feature requests `here <https://github.com/mlepori1/NeuroSurgeon>`_!

