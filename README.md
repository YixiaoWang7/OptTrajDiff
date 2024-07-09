# OptTrajDiff

This repository is the official Pytorch implementation for [OptTrajDiff](https://arxiv.org/abs/2406.08850).

 <!-- [![Project Page](https://img.shields.io/badge/Project-Website-orange)](https://cove-video.github.io/) [![arXiv](https://img.shields.io/badge/arXiv-COVE-b31b1b.svg)](https://arxiv.org/abs/2406.08850)  -->

> **OptTrajDiff: Optimizing Diffusion Models for Joint Trajectory Prediction and Controllable Generation**  
> [Yixiao Wang](https://scholar.google.com/citations?user=HoKoCv0AAAAJ&hl=en),
> [Chen Tang](https://scholar.google.com/citations?user=kwBR1ygAAAAJ&hl=en&oi=ao),
> [Lingfeng Sun](https://scholar.google.com/citations?user=2p6GCEEAAAAJ&hl=en&oi=ao),
> [Simone Rossi](https://scholar.google.com/citations?user=oakZP0cAAAAJ&hl=en&oi=ao),
> [Yichen Xie](https://scholar.google.com/citations?user=-P9LwcgAAAAJ&hl=en&oi=ao),
> [Chensheng Peng](https://scholar.google.com/citations?user=Xrh1OIUAAAAJ&hl=en&oi=ao),
> [Thomas Hannagan](https://scholar.google.com/citations?user=Xrh1OIUAAAAJ&hl=en&oi=ao),
> [Stefano Sabatini](https://scholar.google.com/citations?user=Xrh1OIUAAAAJ&hl=en&oi=ao),
> [Nicola Poerio](https://scholar.google.com/citations?user=Xrh1OIUAAAAJ&hl=en&oi=ao),
> [Masayoshi Tomizuka](https://scholar.google.com/citations?user=Xrh1OIUAAAAJ&hl=en&oi=ao),
> [Wei Zhan](https://scholar.google.com/citations?user=Xrh1OIUAAAAJ&hl=en&oi=ao),


<p>
<!-- <img src="assets/repo_figures/Picture1.jpg" width="1080px"/> -->

 Diffusion models are promising for joint trajectory prediction and controllable generation in autonomous driving, but they face challenges of inefficient inference time and high computational demands. To tackle these challenges, we introduce Optimal Gaussian Diffusion (OGD) and Estimated Clean Manifold (ECM) Guidance. OGD optimizes the prior distribution for a small diffusion time $T$ and starts the reverse diffusion process from it. ECM directly injects guidance gradients to the estimated clean manifold, eliminating extensive gradient backpropagation throughout the network. Our methodology streamlines the generative process, enabling practical applications with reduced computational overhead. Experimental validation on the large-scale Argoverse 2 dataset demonstrates our approach's superior performance, offering a viable solution for computationally efficient, high-quality joint trajectory generation and controllable generation for autonomous driving.

</p>

## News
- [2024.7.1] Paper is accepted by [ECCV 2024](https://eccv2024.ecva.net/)!

## ToDo
- □ Release code



## Citation

If you find our work helpful, please **star 🌟** this repo and **cite 📑** our paper. BibTex will be updated soon. Thanks for your support!

