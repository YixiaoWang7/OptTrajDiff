# OptTrajDiff

This repository is the official github repository for [OptTrajDiff](https://yixiaowang7.github.io/OptTrajDiff_Page/).

 <!-- [![Project Page](https://img.shields.io/badge/Project-Website-orange)](https://cove-video.github.io/) [![arXiv](https://img.shields.io/badge/arXiv-COVE-b31b1b.svg)](https://arxiv.org/abs/2406.08850)  -->

> **OptTrajDiff: Optimizing Diffusion Models for Joint Trajectory Prediction and Controllable Generation**  
> [Yixiao Wang](https://yixiaowang7.github.io),
> [Chen Tang](https://scholar.google.com/citations?user=x78TL58AAAAJ&hl=en&oi=ao),
> [Lingfeng Sun](https://scholar.google.com/citations?user=Uxb6wbkAAAAJ&hl=en&oi=ao),
> [Simone Rossi](https://scholar.google.com/citations?hl=en&user=lTt86awAAAAJ),
> [Yichen Xie](https://scholar.google.com/citations?user=SdX6DaEAAAAJ&hl=en&oi=ao),
> [Chensheng Peng](https://scholar.google.com/citations?user=DbZxclcAAAAJ&hl=en&oi=ao),
> Thomas Hannagan,
> Stefano Sabatini,
> Nicola Poerio,
> [Masayoshi Tomizuka](https://scholar.google.com/citations?user=8m8taGEAAAAJ&hl=en&oi=ao),
> [Wei Zhan](https://scholar.google.com/citations?user=xVN3UxYAAAAJ&hl=en&oi=ao),


<p>
<!-- <img src="assets/repo_figures/Picture1.jpg" width="1080px"/> -->

 Diffusion models are promising for joint trajectory prediction and controllable generation in autonomous driving, but they face challenges of inefficient inference time and high computational demands. To tackle these challenges, we introduce Optimal Gaussian Diffusion (OGD) and Estimated Clean Manifold (ECM) Guidance. OGD optimizes the prior distribution for a small diffusion time $T$ and starts the reverse diffusion process from it. ECM directly injects guidance gradients to the estimated clean manifold, eliminating extensive gradient backpropagation throughout the network. Our methodology streamlines the generative process, enabling practical applications with reduced computational overhead. Experimental validation on the large-scale Argoverse 2 dataset demonstrates our approach's superior performance, offering a viable solution for computationally efficient, high-quality joint trajectory generation and controllable generation for autonomous driving.

</p>

## News
- [2024.7.1] Paper is accepted by [ECCV 2024](https://eccv2024.ecva.net/)!

## ToDo
<!-- - □ Release code -->
- [x] Release code

## Initial Setup

**Step 1**: Download the code by cloning the repository:
```
git clone https://github.com/YixiaoWang7/OptTrajDiff.git && cd OptTrajDiff
```

**Step 2**: Set up a new conda environment and install required packages:
```
conda env create -f environment.yml
conda activate OptTrajDiff
```

**Step 3**: Implement the [Argoverse 2 API](https://github.com/argoverse/av2-api) and access the [Argoverse 2 Motion Forecasting Dataset](https://www.argoverse.org/av2.html). Please see the [Argoverse 2 User Guide](https://argoverse.github.io/user-guide/getting_started.html).



## Joint Trajectory Prediction with Optimal Gaussian Diffusion

### Training Command
```sh
python train_diffnet_tb.py --root <Path to dataset> --train_batch_size 16 --val_batch_size 4 --test_batch_size 4 --dataset argoverse_v2 --num_historical_steps 50 --num_future_steps 60 --num_recurrent_steps 3 --pl2pl_radius 150 --time_span 10 --pl2a_radius 50 --a2a_radius 50 --num_t2m_steps 30 --pl2m_radius 150 --a2m_radius 150 --devices "4,5,6" --qcnet_ckpt_path <Path to QCNet checkpoint> --num_workers 4 --num_denoiser_layers 3 --num_diffusion_steps 100 --T_max 30 --max_epochs 30 --lr 0.005 --beta_1 0.0001 --beta_T 0.05 --diff_type opd --sampling ddim --sampling_stride 10 --num_eval_samples 6 --choose_best_mode FDE --std_reg 0.3 --check_val_every_n_epoch 3 --path_pca_s_mean 'pca/imp_org/s_mean_10.npy' --path_pca_VT_k 'pca/imp_org/VT_k_10.npy' --path_pca_V_k 'pca/imp_org/V_k_10.npy' --path_pca_latent_mean 'pca/imp_org/latent_mean_10.npy' --path_pca_latent_std 'pca/imp_org/latent_std_10.npy'
```
Below are the significant arguments related to our work:

- `--devices`: Specifies the GPUs you want to use.
- `--qcnet_ckpt_path`: Provides the path to the QCNet checkpoints.
- `--num_denoiser_layers`: Defines the number of layers in the diffusion network.
- `--num_diffusion_steps`: Sets the number of diffusion steps.
- `--max_epochs`: Determines the total number of training epochs.
- `--lr`: Sets the learning rate.
- `--beta_1`: Specifies the  $\beta_1$, the diffusion schedule parameter.
- `--beta_T`: Specifies the $\beta_T$, the diffusion schedule parameter.
- `--sampling_stride`: Defines the sampling stride for DDIM.
- `--num_eval_samples`: Indicates the number of evaluation samples.


### Validation Command
```sh
python val_diffnet.py --root <Path to dataset> --ckpt_path <Path to diffusion network checkpoint> --devices '5,' --batch_size 8 --sampling ddim --sampling_stride 10 --num_eval_samples 128 --std_reg 0.3 --path_pca_V_k 'pca/imp_org/V_k_10.npy' --network_mode 'val'
```



## Controllable Generation with ECM and ECMR
```sh
python val_diffnet.py --root <Path to dataset> --ckpt_path <Path to diffusion network checkpoint> --devices '2,' --batch_size 16 --sampling ddim --sampling_stride 10 --num_eval_samples 128 --std_reg 0.3 --path_pca_V_k 'pca/imp_org/V_k_10.npy' --network_mode 'val' --guid_sampling 'guid' --guid_task 'rand_goal_5s' --guid_method <Guided sampling method> --guid_plot plot --cost_param_costl 10.0 --cost_param_threl 1.0
```
Below are the key arguments relevant to our work:
- `--guid_method`: Specifies the guided sampling method you wish to use. Options are `['none', 'ECM', 'ECMR']`.
- `--guid_task`: Defines the controllable tasks you plan to test. Options include `['none', 'goal', 'goal_5s', 'goal_at5s', 'rand_goal', 'rand_goal_5s', 'rand_goal_at5s']`.


## Citation
If you find our work helpful, please **star 🌟** this repo and **cite 📑** our paper. BibTex will be updated soon. Thanks for your support!
```
Coming soon.
```

This repository is developed based on [Query-Centric Trajectory Prediction](https://github.com/ZikangZhou/QCNet).
Please also consider citing:
```
@inproceedings{zhou2023query,
  title={Query-Centric Trajectory Prediction},
  author={Zhou, Zikang and Wang, Jianping and Li, Yung-Hui and Huang, Yu-Kai},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```