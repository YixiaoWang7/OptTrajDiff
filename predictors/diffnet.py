# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from itertools import chain
from itertools import compress
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from losses import MixtureNLLLoss
from losses import NLLLoss
from metrics import Brier
from metrics import MR
from metrics import minADE, BestMinJADE, BestMinJFDE, minJADE, minJFDE,meanJADE,meanJFDE, minJLDE, meanJLDE
from metrics import minAHE
from metrics import minFDE
from metrics import minFHE
from metrics import meanJFDEG, minJFDEG
from metrics import meanVelRate, maxVelRate, targetVelminError, targetVelmeanError, KinematicFeasibleRate
from predictors import QCNet
from modules import JointDiffusion
import numpy as np
from time import time
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture as GMM

try:
    from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission
except ImportError:
    ChallengeSubmission = object

from av2.datasets.motion_forecasting import scenario_serialization
from visualization import *
from av2.map.map_api import ArgoverseStaticMap

import os

class DiffNet(pl.LightningModule):

    def __init__(self,
                 args,
                 **kwargs) -> None:
        super(DiffNet, self).__init__()
        self.save_hyperparameters()
        self.dataset = args.dataset
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        self.output_head = args.output_head
        self.num_historical_steps = args.num_historical_steps
        self.num_future_steps = args.num_future_steps
        self.num_modes = args.num_modes
        self.num_recurrent_steps = args.num_recurrent_steps
        self.num_freq_bands = args.num_freq_bands
        self.num_map_layers = args.num_map_layers
        self.num_agent_layers = args.num_agent_layers
        self.num_dec_layers = args.num_dec_layers
        self.num_heads = args.num_heads
        self.head_dim = args.head_dim
        self.dropout = args.dropout
        self.pl2pl_radius = args.pl2pl_radius
        self.time_span = args.time_span
        self.pl2a_radius = args.pl2a_radius
        self.a2a_radius = args.a2a_radius
        self.num_t2m_steps = args.num_t2m_steps
        self.pl2m_radius = args.pl2m_radius
        self.a2m_radius = args.a2m_radius
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.T_max = args.T_max
        self.submission_dir = args.submission_dir
        self.submission_file_name = args.submission_file_name
        self.diff_type = args.diff_type
        self.sampling = args.sampling
        self.sampling_stride = args.sampling_stride
        self.num_diffusion_steps = args.num_diffusion_steps
        self.num_eval_samples = args.num_eval_samples
        self.eval_mode_error_2 = args.eval_mode_error_2
        self.choose_best_mode = args.choose_best_mode
        self.train_agent = args.train_agent
        self.path_pca_s_mean = args.path_pca_s_mean
        self.path_pca_VT_k = args.path_pca_VT_k
        self.path_pca_latent_mean = args.path_pca_latent_mean
        self.path_pca_latent_std = args.path_pca_latent_std
        self.s_mean = None
        self.VT_k = None
        self.latent_mean = None
        self.latent_std = None
        self.m_dim = args.m_dim
        
        
        self.check_param()

        self.qcnet = QCNet.load_from_checkpoint(checkpoint_path=args.qcnet_ckpt_path)
        self.qcnet.freeze()
        
        self.linear = nn.Linear(10,2)
        
        self.joint_diffusion = JointDiffusion(args=args)

        self.reg_loss = NLLLoss(component_distribution=['laplace'] * args.output_dim + ['von_mises'] * args.output_head,
                                reduction='none')
        self.cls_loss = MixtureNLLLoss(component_distribution=['laplace'] * args.output_dim + ['von_mises'] * args.output_head,
                                       reduction='none')

        self.Brier = Brier(max_guesses=6)
        self.minADE = minADE(max_guesses=6)
        self.minAHE = minAHE(max_guesses=6)
        self.minFDE = minFDE(max_guesses=6)
        self.minFHE = minFHE(max_guesses=6)
        self.MR = MR(max_guesses=6)

        self.test_predictions = dict()
        self.BestMinJADE = BestMinJADE(max_guesses=6)
        self.BestMinJFDE = BestMinJFDE(max_guesses=6)
        
        self.minJADE = minJADE(max_guesses=6)
        self.minJFDE = minJFDE(max_guesses=6)
        
        self.minJADE_diff = minJADE(max_guesses=6)
        self.minJFDE_diff = minJFDE(max_guesses=6)
        
        self.minJADE_diff_c = minJADE(max_guesses=6)
        self.minJFDE_diff_c = minJFDE(max_guesses=6)
        
        self.minJLDE= minJLDE()
        self.meanJLDE= meanJLDE()
        
        self.minJFDEG= minJFDEG()
        self.meanJFDEG= meanJFDEG()
        
        self.meanJADE_diff_c = meanJADE(max_guesses=6)
        self.meanJFDE_diff_c = meanJFDE(max_guesses=6)
        
        self.minJADE_diff_gmm = minJADE(max_guesses=6)
        self.minJFDE_diff_gmm = minJFDE(max_guesses=6)
        
        self.meanVelRate = meanVelRate()
        self.maxVelRate = maxVelRate()
        self.KinematicFeasibleRate = KinematicFeasibleRate()
        
        self.KinematicConfortRate = KinematicFeasibleRate()
        
        self.targetVelminError = targetVelminError()
        self.targetVelmeanError = targetVelmeanError()
        
        self.num_all_agents = 0
        self.M_dis = []
        self.order_ac = []
        
        # self.automatic_optimization=False
        
    def add_extra_param(self, args):
        self.guid_sampling = args.guid_sampling
        self.guid_task = args.guid_task
        self.guid_method = args.guid_method
        self.guid_plot = args.guid_plot
        self.std_reg = args.std_reg
        self.path_pca_V_k = args.path_pca_V_k
        self.V_k = None
        self.cluster = args.cluster
        self.cluster_max_thre = args.cluster_max_thre
        self.cluster_mean_thre = args.cluster_mean_thre

        self.cond_norm = args.cond_norm
        self.cost_param_costl = args.cost_param_costl
        self.cost_param_threl = args.cost_param_threl
        
        
    def check_param(self):
        if self.sampling == 'ddpm':
            self.sampling_stride = 1
        elif self.sampling == 'ddim':
            self.sampling_stride = int(self.sampling_stride)
            if self.sampling_stride > self.num_diffusion_steps - 1:
                print('ddim stride > diffusion steps.')
                exit()
            scale = self.num_diffusion_steps / self.sampling_stride
            if abs(scale - int(scale)) > 0.00001:
                print('mod(diffusion steps, ddim stride) != 0')
                exit()

    def forward(self, data: HeteroData):
        scene_enc = self.qcnet.encoder(data)
        x = torch.ones(32,10).to(scene_enc['x_a'].device)
        return self.linear(x)

    def normalize(self, original_data, mean, std):
        if original_data.dim() == 2:
            if mean.dim() == 1:
                return (original_data - mean.unsqueeze(0))/(std.unsqueeze(0)+0.1)
            if mean.dim() == 2:
                return (original_data - mean)/(std+0.1)
        elif original_data.dim() == 3:
            if mean.dim() == 1:
                return (original_data - mean.unsqueeze(0).unsqueeze(0))/(std.unsqueeze(0).unsqueeze(0)+0.1)
            if mean.dim() == 2:
                return (original_data - mean.unsqueeze(1))/(std.unsqueeze(1)+0.1)
        else:
            raise ValueError('normalized data should 2-dimensional or 3-dimensional.')
    
    def unnormalize(self, original_data, mean, std):
        if original_data.dim() == 2:
            if mean.dim() == 1:
                return original_data*(std.unsqueeze(0)+0.1) + mean.unsqueeze(0)
            if mean.dim() == 2:
                return original_data*(std+0.1) + mean
        elif original_data.dim() == 3:
            if mean.dim() == 1:
                return original_data * (std.unsqueeze(0).unsqueeze(0)+0.1) + mean.unsqueeze(0).unsqueeze(0)
            if mean.dim() == 2:
                return original_data * (std.unsqueeze(1)+0.1) + mean.unsqueeze(1)
        else:
            raise ValueError('normalized data should 2-dimensional or 3-dimensional.')
    
    def training_step(self,
                      data,
                      batch_idx):
        
        print_flag = False
        if batch_idx % 100 == 0:
            print_flag = True
        
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        
        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        pred, scene_enc = self.qcnet(data)
        if self.output_head:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)

        # note that I change the training data
        eval_mask = (data['agent']['category'] >= 2) & (reg_mask[:,-1] == True) & (reg_mask[:,0] == True)
        mask = eval_mask
        gt = gt[mask][..., :self.output_dim]
        reg_mask = reg_mask[mask]
        num_agent = gt.size(0)
        reg_start_list = []
        reg_end_list = []
        for i in range(num_agent):
            start = []
            end = []
            for j in range(59):
                if reg_mask[i,j] == True and reg_mask[i,j+1] == False:
                    start.append(j)
                elif reg_mask[i,j] == False and reg_mask[i,j+1] == True:
                    end.append(j+1)
            reg_start_list.append(start)
            reg_end_list.append(end)
            
        for i in range(num_agent):
            count = 0
            for j in range(59):
                if reg_mask[i,j] == False:
                    start_id = reg_start_list[i][count]
                    end_id = reg_end_list[i][count]
                    start_pt = gt[i, start_id]
                    end_pt = gt[i, end_id]
                    gt[i,j] = start_pt + (end_pt - start_pt) / (end_id - start_id) * (j-start_id)
                    if j == end_id - 1:
                        count += 1

        flat_gt = gt.reshape(gt.size(0), -1)
        if self.s_mean == None:
            s_mean = np.load(self.path_pca_s_mean)
            self.s_mean = torch.tensor(s_mean).to(gt.device)
            VT_k = np.load(self.path_pca_VT_k)
            self.VT_k = torch.tensor(VT_k).to(gt.device)
            if self.path_pca_V_k != 'none':
                V_k = np.load(self.path_pca_V_k)
                self.V_k = torch.tensor(V_k).to(gt.device)
            else:
                self.V_k = self.VT_k.transpose(0,1)
            latent_mean = np.load(self.path_pca_latent_mean)
            self.latent_mean = torch.tensor(latent_mean).to(gt.device)
            latent_std = np.load(self.path_pca_latent_std) * 2
            self.latent_std = torch.tensor(latent_std).to(gt.device)
                    
        target_mode = torch.matmul(flat_gt-self.s_mean, self.VT_k)
        target_mode = self.normalize(target_mode, self.latent_mean, self.latent_std)
        
        marginal_trajs = traj_refine[eval_mask,:,:,:2]
        marginal_trajs = marginal_trajs.view(marginal_trajs.size(0),self.num_modes,-1)
        marginal_mode = torch.matmul((marginal_trajs-self.s_mean.unsqueeze(1)).permute(1,0,2), self.VT_k.unsqueeze(0).repeat(self.num_modes,1,1))
        marginal_mode = marginal_mode.permute(1,0,2)

        marginal_mode = self.normalize(marginal_mode, self.latent_mean, self.latent_std)

        marg_mean = marginal_mode.mean(dim=1)
        marg_std = marginal_mode.std(dim=1) + self.std_reg
        
        if self.cond_norm == 1:
            marginal_mode = self.normalize(marginal_mode, marg_mean, marg_std)
            target_mode = self.normalize(target_mode, marg_mean, marg_std)

            mean = torch.zeros_like(marg_mean)
            std = torch.ones_like(marg_std)
        else:
            mean = marg_mean
            std = marg_std

        
        
        loss = self.joint_diffusion.get_loss(target_mode, data = data, scene_enc = scene_enc,mean=mean,std=std, mm = marginal_mode, mmscore = pi.exp()[eval_mask],eval_mask=eval_mask)
        
        self.log('train_loss', loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        
        if print_flag:
            print(batch_idx,loss)
        
        return loss

    def validation_step(self,
                    data,
                    batch_idx):
        if self.guid_sampling == 'no_guid':
            self.validation_step_norm(data, batch_idx)
        elif self.guid_sampling == 'guid':
            self.validation_step_guid(data, batch_idx)
    
    def validation_step_norm(self,
                    data,
                    batch_idx):
        print_flag = False
        if batch_idx % 1 == 0:
            print_flag = True
            
        
        data_batch = batch_idx
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        
        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        pred, scene_enc = self.qcnet(data)
        if self.output_head:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                    pred['loc_refine_head'],
                                    pred['scale_refine_pos'][..., :self.output_dim],
                                    pred['conc_refine_head']], dim=-1)
        else:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                    pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
        l2_norm = (torch.norm(traj_refine[..., :self.output_dim] -
                            gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
        
        if self.s_mean == None:
            s_mean = np.load(self.path_pca_s_mean)
            self.s_mean = torch.tensor(s_mean).to(gt.device)
            VT_k = np.load(self.path_pca_VT_k)
            self.VT_k = torch.tensor(VT_k).to(gt.device)
            if self.path_pca_V_k != 'none':
                V_k = np.load(self.path_pca_V_k)
                self.V_k = torch.tensor(V_k).to(gt.device)
            else:
                self.V_k = self.VT_k.transpose(0,1)
            latent_mean = np.load(self.path_pca_latent_mean)
            self.latent_mean = torch.tensor(latent_mean).to(gt.device)
            latent_std = np.load(self.path_pca_latent_std) * 2
            self.latent_std = torch.tensor(latent_std).to(gt.device)
        
        
        eval_mask = data['agent']['category'] >= 2
        
        mask = (data['agent']['category'] >= 2) & (reg_mask[:,-1] == True) & (reg_mask[:,0] == True)
        gt_n = gt[mask][..., :self.output_dim]
        gt_n[0,:,:] = (gt_n[0,:,:] - gt_n[0,0:1,:]) / 4 * 3 + gt_n[0,0:1,:]
        reg_mask_n = reg_mask[mask]
        num_agent = gt_n.size(0)
        reg_start_list = []
        reg_end_list = []
        for i in range(num_agent):
            start = []
            end = []
            for j in range(59):
                if reg_mask_n[i,j] == True and reg_mask_n[i,j+1] == False:
                    start.append(j)
                elif reg_mask_n[i,j] == False and reg_mask_n[i,j+1] == True:
                    end.append(j+1)
            reg_start_list.append(start)
            reg_end_list.append(end)
            
        for i in range(num_agent):
            count = 0
            for j in range(59):
                if reg_mask_n[i,j] == False:
                    start_id = reg_start_list[i][count]
                    end_id = reg_end_list[i][count]
                    start_pt = gt_n[i, start_id]
                    end_pt = gt_n[i, end_id]
                    gt_n[i,j] = start_pt + (end_pt - start_pt) / (end_id - start_id) * (j-start_id)
                    if j == end_id - 1:
                        count += 1
        
        flat_gt = gt_n.reshape(gt_n.size(0),-1)
        k_vector = torch.matmul(flat_gt-self.s_mean, self.VT_k)
        rec_flat_gt = torch.matmul(k_vector, self.V_k) + self.s_mean
        rec_gt = rec_flat_gt.view(-1,60,2)
        
        target_mode = torch.matmul(flat_gt-self.s_mean, self.VT_k)
        target_mode = self.normalize(target_mode, self.latent_mean, self.latent_std)
        
        # fde is valid for all eval agent
        fde_marginal_gt = torch.norm(traj_refine[:,:,-1,:2] - gt[:,-1,:2].unsqueeze(1),dim=-1)

        # [num_agents,] index of best mode
        if self.choose_best_mode == 'FDE':
            best_fde_mode = fde_marginal_gt.argmin(dim=-1)
            best_l2_mode = l2_norm.argmin(dim=-1)
            best_mode = best_l2_mode
            fde_valid = reg_mask[:,-1]==True
            best_mode[fde_valid] = best_fde_mode[fde_valid]
        elif self.choose_best_mode == 'ADE':
            best_l2_mode = l2_norm.argmin(dim=-1)
            best_mode = best_l2_mode

        if self.dataset == 'argoverse_v2':
            eval_mask = data['agent']['category'] >= 2
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        valid_mask_eval = reg_mask[eval_mask]

        pi_eval = F.softmax(pi[eval_mask], dim=-1)
        gt_eval = gt[eval_mask]
    
        marginal_trajs = traj_refine[eval_mask,:,:,:2]
        marginal_trajs = marginal_trajs.view(marginal_trajs.size(0),self.num_modes,-1)
        marginal_mode = torch.matmul((marginal_trajs-self.s_mean.unsqueeze(1)).permute(1,0,2), self.VT_k.unsqueeze(0).repeat(self.num_modes,1,1))
        marginal_mode = marginal_mode.permute(1,0,2)
        marginal_mode = self.normalize(marginal_mode, self.latent_mean, self.latent_std)
        marg_mean = marginal_mode.mean(dim=1)
        marg_std = marginal_mode.std(dim=1) + self.std_reg


        if self.cond_norm:
            marginal_mode = self.normalize(marginal_mode, marg_mean, marg_std)
            target_mode = self.normalize(target_mode, marg_mean, marg_std)

            mean = torch.zeros_like(marg_mean)
            std = torch.ones_like(marg_std)
        else:
            mean = marg_mean
            std = marg_std


        self.joint_diffusion.eval()
        num_samples = self.num_eval_samples

        if_output_diffusion_process = False
        if if_output_diffusion_process:
            reverse_steps = 100
            pred_modes = self.joint_diffusion.sample(num_samples, data = data, scene_enc = scene_enc, 
                                                    mean = mean, std = std, mm = marginal_mode, 
                                                    mmscore = pi.exp()[eval_mask],sampling=self.sampling,
                                                    stride=self.sampling_stride,eval_mask=eval_mask,
                                                    if_output_diffusion_process=if_output_diffusion_process,
                                                    reverse_steps=reverse_steps)
            inter_latents = pred_modes[::1]
            inter_trajs = []
            for latent in inter_latents:
                unnorm_pred_modes = self.unnormalize(latent,self.latent_mean, self.latent_std)
                rec_traj = torch.matmul(unnorm_pred_modes.permute(1,0,2), (self.V_k).unsqueeze(0).repeat(self.num_eval_samples,1,1)) + self.s_mean.unsqueeze(0)
                rec_traj = rec_traj.permute(1,0,2)
                rec_traj = rec_traj.view(rec_traj.size(0), rec_traj.size(1),self.num_future_steps,2)
                inter_trajs.append(rec_traj)
            
            
            
            pred_modes = pred_modes[-1]
            
            random_modes = torch.fmod(torch.randn_like(pred_modes),3) / 2
            unnorm_pred_modes = self.unnormalize(random_modes,self.latent_mean, self.latent_std)
            rec_traj = torch.matmul(unnorm_pred_modes.permute(1,0,2), (self.V_k).unsqueeze(0).repeat(self.num_eval_samples,1,1)) + self.s_mean.unsqueeze(0)
            rec_traj = rec_traj.permute(1,0,2)
            random_traj = rec_traj.view(rec_traj.size(0), rec_traj.size(1),self.num_future_steps,2)
            
            
            
            step_list = [0,10,20,30,40,50,60,70,80,90,100]
            mean_1 = mean.unsqueeze(1)
            std_1 = std.unsqueeze(1)
            # std = torch.ones_like(std)
            
            
            random_modes = torch.fmod(torch.randn_like(pred_modes),3) / 2 * std_1
            unnorm_pred_modes = self.unnormalize(random_modes,self.latent_mean, self.latent_std)
            rec_traj = torch.matmul(unnorm_pred_modes.permute(1,0,2), (self.V_k).unsqueeze(0).repeat(self.num_eval_samples,1,1)) + self.s_mean.unsqueeze(0)
            rec_traj = rec_traj.permute(1,0,2)
            random_kernel_traj = rec_traj.view(rec_traj.size(0), rec_traj.size(1),self.num_future_steps,2)
            
            opt_trajs = []
            for step in step_list:
                # c1 = torch.sqrt(1-self.joint_diffusion.var_sched.alpha_bars[step]).to(pred_modes.device)
                c1 = 1
                e_rand = torch.fmod(torch.randn_like(pred_modes),3) * std_1
                s_T = torch.sqrt(self.joint_diffusion.var_sched.alpha_bars[step].to(e_rand.device))* mean_1
                x_T = c1 * e_rand + s_T
                
                unnorm_pred_modes = self.unnormalize(x_T,self.latent_mean, self.latent_std)
                rec_traj = torch.matmul(unnorm_pred_modes.permute(1,0,2), (self.V_k).unsqueeze(0).repeat(self.num_eval_samples,1,1)) + self.s_mean.unsqueeze(0)
                rec_traj = rec_traj.permute(1,0,2)
                rec_traj = rec_traj.view(rec_traj.size(0), rec_traj.size(1),self.num_future_steps,2)
                opt_trajs.append(rec_traj)
                
            
            noised_gt_trajs = []
            for step in step_list:
                c1 = torch.sqrt(1-self.joint_diffusion.var_sched.alpha_bars[step]).to(pred_modes.device)
                e_rand = torch.fmod(torch.randn_like(pred_modes),3) * std_1
                s_T = torch.sqrt(self.joint_diffusion.var_sched.alpha_bars[step].to(e_rand.device))* target_mode.unsqueeze(1)
                x_T = c1 * e_rand + s_T
                
                unnorm_pred_modes = self.unnormalize(x_T,self.latent_mean, self.latent_std)
                rec_traj = torch.matmul(unnorm_pred_modes.permute(1,0,2), (self.V_k).unsqueeze(0).repeat(self.num_eval_samples,1,1)) + self.s_mean.unsqueeze(0)
                rec_traj = rec_traj.permute(1,0,2)
                rec_traj = rec_traj.view(rec_traj.size(0), rec_traj.size(1),self.num_future_steps,2)
                noised_gt_trajs.append(rec_traj)
                
            
        else:
            
            start_data = None
            reverse_steps = None
            pred_modes = self.joint_diffusion.sample(num_samples, data = data, scene_enc = scene_enc, 
                                                    mean = mean, std = std, mm = marginal_mode, 
                                                    mmscore = pi.exp()[eval_mask],sampling=self.sampling,
                                                    stride=self.sampling_stride,eval_mask=eval_mask,
                                                    start_data=start_data, reverse_steps=reverse_steps)
        
        

        
        if self.cond_norm:
            pred_modes = self.unnormalize(pred_modes, marg_mean, marg_std)

        unnorm_pred_modes = self.unnormalize(pred_modes,self.latent_mean, self.latent_std)
        rec_traj = torch.matmul(unnorm_pred_modes.permute(1,0,2), (self.V_k).unsqueeze(0).repeat(self.num_eval_samples,1,1)) + self.s_mean.unsqueeze(0)
        rec_traj = rec_traj.permute(1,0,2)
        rec_traj = rec_traj.view(rec_traj.size(0), rec_traj.size(1),self.num_future_steps,2)
        
        if True in torch.isnan(pred_modes):
            print('nan')
            print(data_batch)
            exit()
        
        first_mapping_in_latent = False
        if first_mapping_in_latent:
            mode_diff = pred_modes.unsqueeze(-2).repeat(1,1,self.num_modes,1) - marginal_mode.unsqueeze(1)
            # [num_agents, num_samples, num_modes]
            mode_diff = mode_diff.norm(dim=-1)
        else:
            mode_diff = rec_traj[:,:,-1,:2].unsqueeze(-2).repeat(1,1,self.num_modes,1) - traj_refine[eval_mask,:,-1,:2].unsqueeze(1)
            mode_diff = mode_diff.norm(dim=-1)
            
        # find the closest mode
        
        mode_joint_best = torch.argmin(mode_diff,dim=-1)

        # joint mode clustering
        device = mean.device
        batch_idx = data['agent']['batch'][eval_mask]
        num_scenes = batch_idx[-1].item()+1
        num_agents_per_scene = mode_joint_best.new_tensor([(batch_idx == i).sum() for i in range(num_scenes)])
        top_modes = torch.randint(0,self.num_modes,(pred_modes.size(0),self.num_modes)).to(pred_modes.device)
        refine_pi = torch.zeros(top_modes.size(0),self.num_modes).to(pred_modes.device)
        best_mode_eval = best_mode[eval_mask]
        best_mode_key = 'k'
        for it in best_mode_eval:
            best_mode_key += str(it.cpu().numpy())
        
        ### [num_agents, num_samples, 2]
        joint_goal_pts = traj_refine[eval_mask,:,-1,:2]
        for i in range(num_scenes):
            start_id = torch.sum(num_agents_per_scene[:i])
            end_id = torch.sum(num_agents_per_scene[:i+1])
            
            # initialize the modes for single agent
            if end_id - start_id == 1:
                for j in range(self.num_modes):
                    top_modes[start_id:end_id,j] = j
            
            # cluster the modes
            topk_keys = []
            topk_nums = []
            topk_joint_modes = []
            for j in range(num_samples):
                key = 'k'
                for it in mode_joint_best[start_id:end_id,j]:
                    key += str(it.cpu().numpy())
                
                try:
                    idx = topk_keys.index(key)
                    topk_nums[idx] += 1
                except ValueError:
                    topk_keys.append(key)
                    topk_nums.append(1)
                    topk_joint_modes.append(mode_joint_best[start_id:end_id,j:j+1])

            topk_nums = torch.tensor(topk_nums).to(device)
            topk_joint_modes = torch.cat(topk_joint_modes, dim=1)
            
            # sort
            ids = torch.argsort(topk_nums, descending=True)
            
            if self.cluster == 'normal':
                topk_ids = ids[:self.num_modes]
                total_num = torch.sum(topk_nums[topk_ids])
                top_modes[start_id:end_id,:topk_ids.size(0)] = topk_joint_modes[:, topk_ids]
                refine_pi[start_id:end_id,:topk_ids.size(0)] = topk_nums[topk_ids]/total_num
                
            elif self.cluster == 'traj':
                ids_init = ids
                nums_init = topk_nums
                # print('init',nums_init[ids_init])
                
                # based on sort, cluster in the trajectory space
                ### [num_agents, num_samples, 2]
                scene_joint_goal_pts = joint_goal_pts[start_id:end_id]
                num_agents = scene_joint_goal_pts.size(0)
                max_threshold = self.cluster_max_thre
                # print(max_threshold)
                mean_threshold = self.cluster_mean_thre
                queue = torch.ones(ids.shape[0],dtype=torch.int8)
                nms_nums = []
                nms_idx = []
                for j in range(ids.size(0)):
                    if queue[j] == 0:
                        continue
                    queue[j] = 0
                    idx = ids[j]
                    nms_idx.append(idx)
                    temp_nums = topk_nums[idx]
                    temp_modes = topk_joint_modes[:,idx]
                    target_jgp = scene_joint_goal_pts[torch.arange(num_agents), temp_modes]
                                        
                    # queue_ids = torch.arange(j+1,ids.size(0)).to(device)
                    queue_ids = torch.nonzero(queue).squeeze(1).to(device)
                    
                    
                    cand_ids = ids[queue_ids]        
                    num_cands = cand_ids.size(0)
                    cand_modes = topk_joint_modes[:,cand_ids]
                    cand_jgp = scene_joint_goal_pts[torch.arange(num_agents).unsqueeze(-1).repeat(1,num_cands), cand_modes]
                    diff =  torch.norm(target_jgp.unsqueeze(1) - cand_jgp,dim=-1)
                    max_diff = torch.max(diff, dim=0)[0]
                    group_ids = torch.nonzero(max_diff < max_threshold).squeeze(1)
                    cand_nums = topk_nums[cand_ids[group_ids]]
                    queue[queue_ids[group_ids]] = 0
                    nms_nums.append(temp_nums + torch.sum(cand_nums))
                    
                    
                # when clustering into less than num_modes groups, add more groups
                if len(nms_idx) < self.num_modes:
                    for idx in ids_init:
                        if idx not in nms_idx:
                            nms_idx.append(idx)
                            nms_nums.append(nums_init[idx])
                            if len(nms_idx) == self.num_modes:
                                break
                    
                # assign modes
                nms_idx = torch.tensor(nms_idx).to(device)
                nms_nums = torch.tensor(nms_nums).to(device)
                ids = torch.argsort(nms_nums, descending=True)
                topk_ids = ids[:self.num_modes]
                
                lack_nms = self.num_modes - topk_ids.size(0)
                if end_id - start_id == 1 and lack_nms > 0:
                    total_num = torch.sum(nms_nums[topk_ids]+lack_nms)
                    top_modes[start_id:end_id,:topk_ids.size(0)] = topk_joint_modes[:, nms_idx[topk_ids]]
                    refine_pi[start_id:end_id,:topk_ids.size(0)] = nms_nums[topk_ids]/total_num
                    for k in range(lack_nms):
                        for j in range(self.num_modes):
                            if j not in top_modes[start_id:end_id,:topk_ids.size(0)+k]:
                                top_modes[start_id:end_id,topk_ids.size(0)+k]=j
                    
                    
                else:
                    total_num = torch.sum(nms_nums[topk_ids])
                    top_modes[start_id:end_id,:topk_ids.size(0)] = topk_joint_modes[:, nms_idx[topk_ids]]
                    refine_pi[start_id:end_id,:topk_ids.size(0)] = nms_nums[topk_ids]/total_num

        
        
        num_mm = top_modes.size(1)
        traj_refine_eval = traj_refine[eval_mask]
        traj_refine_topk = traj_refine_eval[torch.arange(traj_refine_eval.size(0)).unsqueeze(-1).repeat(1,num_mm), top_modes][...,:self.output_dim]
        
        # joint metrics
        batch_agent_idx = data['agent']['batch'][eval_mask]
        self.minJADE_diff.update(batch_agent_idx = batch_agent_idx, 
                            pred=traj_refine_topk[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                           valid_mask=valid_mask_eval)
        self.log('val_minJADE_diff', self.minJADE_diff, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)
        
        
        self.minJFDE_diff.update(batch_agent_idx = batch_agent_idx, 
                            pred=traj_refine_topk[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                           valid_mask=valid_mask_eval)
        
        self.log('val_minJFDE_diff', self.minJFDE_diff, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)
        

        if print_flag:
            print('truncted JADE',self.minJADE_diff.compute())
            print('truncted JFDE',self.minJFDE_diff.compute())
        
        

        batch_agent_idx = data['agent']['batch'][eval_mask]
        self.minJADE_diff_c.update(batch_agent_idx = batch_agent_idx, 
                            pred=rec_traj, target=gt_eval[..., :self.output_dim], prob=pi_eval,
                        valid_mask=valid_mask_eval)
        self.log('val_minJADE_diff_c', self.minJADE_diff_c, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)
        
        
        self.minJFDE_diff_c.update(batch_agent_idx = batch_agent_idx, 
                            pred=rec_traj, target=gt_eval[..., :self.output_dim], prob=pi_eval,
                        valid_mask=valid_mask_eval)
        
        self.log('val_minJFDE_diff_c', self.minJFDE_diff_c, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)
        
        if print_flag:
            print('JADE_c',self.minJADE_diff_c.compute())
            print('JFDE_c',self.minJFDE_diff_c.compute())
        
        goal_point = gt_eval[:,-1,:2]
        
        plot = False
        if plot:
            origin_eval = data['agent']['position'][eval_mask, self.num_historical_steps - 1]
            theta_eval = data['agent']['heading'][eval_mask, self.num_historical_steps - 1]
            cos, sin = theta_eval.cos(), theta_eval.sin()
            rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device=self.device)
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = sin
            rot_mat[:, 1, 0] = -sin
            rot_mat[:, 1, 1] = cos
            rec_traj_world = torch.matmul(rec_traj[:, :, :, :2],
                                    rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
            
            marginal_trajs = traj_refine[eval_mask,:,:,:2]
            marg_traj_world = torch.matmul(marginal_trajs[:, :, :, :2],
                                    rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
            
            marg_traj_world=marg_traj_world.detach().cpu().numpy()
            
            
            if if_output_diffusion_process:
                inter_trajs_world = []
                for traj in inter_trajs:
                    traj_world = torch.matmul(traj[:, :, :, :2],
                                    rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
                    inter_trajs_world.append(traj_world.detach().cpu().numpy())

                opt_trajs_world = []
                for traj in opt_trajs:
                    traj_world = torch.matmul(traj[:, :, :, :2],
                                    rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
                    opt_trajs_world.append(traj_world.detach().cpu().numpy())
                
                
                noised_gt_trajs_world = []
                for traj in noised_gt_trajs:
                    traj_world = torch.matmul(traj[:, :, :, :2],
                                    rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
                    noised_gt_trajs_world.append(traj_world.detach().cpu().numpy())
                
                random_traj_world = torch.matmul(random_traj[:, :, :, :2],
                                    rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
                random_traj_world = random_traj_world.detach().cpu().numpy()
                
                random_kernel_traj_world = torch.matmul(random_kernel_traj[:, :, :, :2],
                                    rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
                random_kernel_traj_world = random_kernel_traj_world.detach().cpu().numpy()
            
            gt_eval_world = torch.matmul(gt_eval[:, :, :2],
                                    rot_mat) + origin_eval[:, :2].reshape(-1, 1, 2)
            gt_eval_world = gt_eval_world.detach().cpu().numpy()
            
            goal_point_world = torch.matmul(goal_point[:, None, None, :],
                                            rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
            goal_point_world = goal_point_world.squeeze(1).squeeze(1)
            goal_point_world = goal_point_world.detach().cpu().numpy()


            img_folder = 'visual'
            sub_folder = 'opd_method'
            rec_traj_world = rec_traj_world.detach().cpu().numpy()
            for i in range(num_scenes):
                start_id = torch.sum(num_agents_per_scene[:i])
                end_id = torch.sum(num_agents_per_scene[:i+1])
                
                
                
                if end_id - start_id == 1:
                    continue
                
                temp = gt_eval[start_id:end_id]
                temp_start = temp[:,0,:]
                temp_end = temp[:,-1,:]
                norm = torch.norm(temp_end-temp_start,dim=-1)
                if torch.max(norm) < 10:
                    continue
            
                scenario_id = data['scenario_id'][i]
                base_path_to_data = Path('/mnt/hdd1/trajectory_prediction/qcnet/val/raw')
                scenario_folder = base_path_to_data / scenario_id
                
                static_map_path = scenario_folder / f"log_map_archive_{scenario_id}.json"
                scenario_path = scenario_folder / f"scenario_{scenario_id}.parquet"
                # viz_save_path = viz_output_dir / f"{scenario_id}.mp4"

                scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
                static_map = ArgoverseStaticMap.from_json(static_map_path)
                
                viz_output_dir = Path(img_folder) / sub_folder
                os.makedirs(viz_output_dir,exist_ok=True)

                viz_save_path = viz_output_dir / ('b'+ str(data_batch)+'_s'+str(i)+'_'+self.sampling+'.jpg')
                
                additional_traj = {}
                additional_traj['gt'] = gt_eval_world[start_id:end_id]
                additional_traj['goal_point'] = goal_point_world
                
                additional_traj['marg_traj'] = marg_traj_world[start_id:end_id]
                additional_traj['rec_traj'] = rec_traj_world[start_id:end_id]
                
                traj_visible = {}
                traj_visible['gt'] = False
                traj_visible['gt_goal'] = False
                traj_visible['goal_point'] = False
                traj_visible['marg_traj'] = False
                traj_visible['rec_traj'] = True
                                
                visualize_scenario_prediction(scenario, static_map, additional_traj, traj_visible, viz_save_path)
                
                if if_output_diffusion_process:
                    for j in range(len(inter_trajs_world)):
                        traj = inter_trajs_world[j]
                        additional_traj['rec_traj'] = traj[start_id:end_id]
                        viz_save_path = viz_output_dir / ('b'+ str(data_batch)+'_s'+str(i)+'_'+self.sampling+'_'+'inter_'+str(j)+'_'+'reverse_'+str(reverse_steps)+'.jpg')
                        visualize_scenario_prediction(scenario, static_map, additional_traj, traj_visible, viz_save_path)

                    
                    for j in range(len(opt_trajs_world)):
                        traj = opt_trajs_world[j]
                        additional_traj['rec_traj'] = traj[start_id:end_id]
                        viz_save_path = viz_output_dir / ('b'+ str(data_batch)+'_s'+str(i)+'_'+self.sampling+'_'+'opt_'+str(j)+'.jpg')
                        visualize_scenario_prediction(scenario, static_map, additional_traj, traj_visible, viz_save_path)
                    
                    for j in range(len(noised_gt_trajs_world)):
                        traj = noised_gt_trajs_world[j]
                        additional_traj['rec_traj'] = traj[start_id:end_id]
                        viz_save_path = viz_output_dir / ('b'+ str(data_batch)+'_s'+str(i)+'_'+self.sampling+'_'+'noised_gt_'+str(j)+'.jpg')
                        visualize_scenario_prediction(scenario, static_map, additional_traj, traj_visible, viz_save_path)

                    
                    
                    additional_traj['rec_traj'] = random_traj_world[start_id:end_id]
                    viz_save_path = viz_output_dir / ('b'+ str(data_batch)+'_s'+str(i)+'_'+self.sampling+'_'+'random'+'.jpg')
                    visualize_scenario_prediction(scenario, static_map, additional_traj, traj_visible, viz_save_path)
                    
                    
                    additional_traj['rec_traj'] = random_kernel_traj_world[start_id:end_id]
                    viz_save_path = viz_output_dir / ('b'+ str(data_batch)+'_s'+str(i)+'_'+self.sampling+'_'+'random_kernel'+'.jpg')
                    visualize_scenario_prediction(scenario, static_map, additional_traj, traj_visible, viz_save_path)




    def validation_step_guid(self,
                        data,
                        batch_idx):
        print_flag = False
        if batch_idx % 1 == 0:
            print_flag = True
        
        data_batch = batch_idx
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        
        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        pred, scene_enc = self.qcnet(data)
        if self.output_head:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['loc_propose_head'],
                                      pred['scale_propose_pos'][..., :self.output_dim],
                                      pred['conc_propose_head']], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
        l2_norm = (torch.norm(traj_refine[..., :self.output_dim] -
                              gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
        
        

        
        if self.s_mean == None:
            s_mean = np.load(self.path_pca_s_mean)
            self.s_mean = torch.tensor(s_mean).to(gt.device)
            VT_k = np.load(self.path_pca_VT_k)
            self.VT_k = torch.tensor(VT_k).to(gt.device)
            if self.path_pca_V_k != 'none':
                V_k = np.load(self.path_pca_V_k)
                self.V_k = torch.tensor(V_k).to(gt.device)
            else:
                self.V_k = self.VT_k.transpose(0,1)
            latent_mean = np.load(self.path_pca_latent_mean)
            self.latent_mean = torch.tensor(latent_mean).to(gt.device)
            latent_std = np.load(self.path_pca_latent_std) * 2
            self.latent_std = torch.tensor(latent_std).to(gt.device)
        
        eval_mask = data['agent']['category'] >= 2
        
        mask = (data['agent']['category'] >= 2) & (reg_mask[:,-1] == True) & (reg_mask[:,0] == True)
        gt_n = gt[mask][..., :self.output_dim]
        reg_mask_n = reg_mask[mask]
        num_agent = gt_n.size(0)
        reg_start_list = []
        reg_end_list = []
        for i in range(num_agent):
            start = []
            end = []
            for j in range(59):
                if reg_mask_n[i,j] == True and reg_mask_n[i,j+1] == False:
                    start.append(j)
                elif reg_mask_n[i,j] == False and reg_mask_n[i,j+1] == True:
                    end.append(j+1)
            reg_start_list.append(start)
            reg_end_list.append(end)
            
        for i in range(num_agent):
            count = 0
            for j in range(59):
                if reg_mask_n[i,j] == False:
                    start_id = reg_start_list[i][count]
                    end_id = reg_end_list[i][count]
                    start_pt = gt_n[i, start_id]
                    end_pt = gt_n[i, end_id]
                    gt_n[i,j] = start_pt + (end_pt - start_pt) / (end_id - start_id) * (j-start_id)
                    if j == end_id - 1:
                        count += 1
        
        flat_gt = gt_n.reshape(gt_n.size(0),-1)
        k_vector = torch.matmul(flat_gt-self.s_mean, self.VT_k)
        rec_flat_gt = torch.matmul(k_vector, self.V_k) + self.s_mean
        rec_gt = rec_flat_gt.view(-1,60,2)
        
        target_mode = torch.matmul(flat_gt-self.s_mean, self.VT_k)
        target_mode = self.normalize(target_mode, self.latent_mean, self.latent_std)
        
        # fde is valid for all eval agent
        fde_marginal_gt = torch.norm(traj_refine[:,:,-1,:2] - gt[:,-1,:2].unsqueeze(1),dim=-1)

        # [num_agents,] index of best mode
        if self.choose_best_mode == 'FDE':
            best_fde_mode = fde_marginal_gt.argmin(dim=-1)
            best_l2_mode = l2_norm.argmin(dim=-1)
            best_mode = best_l2_mode
            fde_valid = reg_mask[:,-1]==True
            best_mode[fde_valid] = best_fde_mode[fde_valid]
        elif self.choose_best_mode == 'ADE':
            best_l2_mode = l2_norm.argmin(dim=-1)
            best_mode = best_l2_mode

        if self.dataset == 'argoverse_v2':
            eval_mask = data['agent']['category'] >= 2
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        valid_mask_eval = reg_mask[eval_mask]

        pi_eval = F.softmax(pi[eval_mask], dim=-1)
        gt_eval = gt[eval_mask]
        
            
        marginal_trajs = traj_refine[eval_mask,:,:,:2]
        marginal_trajs = marginal_trajs.view(marginal_trajs.size(0),self.num_modes,-1)
        marginal_mode = torch.matmul((marginal_trajs-self.s_mean.unsqueeze(1)).permute(1,0,2), self.VT_k.unsqueeze(0).repeat(self.num_modes,1,1))
        marginal_mode = marginal_mode.permute(1,0,2)
        marginal_mode = self.normalize(marginal_mode, self.latent_mean, self.latent_std)


        mean = marginal_mode.mean(dim=1)
        std = marginal_mode.std(dim=1) + self.std_reg

        
        # pred_modes [num_agents, num_samples, 128]
        self.joint_diffusion.eval()
        num_samples = self.num_eval_samples
        
        
        goal_point = gt_eval[:,-1,:2].detach().clone()
        
        task = self.guid_task
        if task == 'none':
            cond_gen = None
            grad_guid = None
            
            vel = (gt_eval[:,1:,:2] - gt_eval[:,:-1,:2]).detach().clone()
            vel = torch.abs(vel)
            max_vel = vel.max(-2)[0]
            
            vel = (gt_eval[:,1:,:2] - gt_eval[:,:-1,:2]).detach().clone()
            mean_vel = vel.mean(-2)
            
        elif task == 'goal':
            goal_point = gt_eval[:,-1,:2].detach().clone()
            grad_guid = [goal_point, self.s_mean, self.V_k, self.VT_k, self.latent_mean, self.latent_std]
            cond_gen = None
            
        elif task == 'goal_5s':
            goal_point = gt_eval[:,50,:2].detach().clone()
            grad_guid = [goal_point, self.s_mean, self.V_k, self.VT_k, self.latent_mean, self.latent_std]
            cond_gen = None
            
            
        elif task == 'goal_at5s':
            goal_point = gt_eval[:,-1,:2].detach().clone()
            grad_guid = [goal_point, self.s_mean, self.V_k, self.VT_k, self.latent_mean, self.latent_std]
            cond_gen = None
            
            
        elif task == 'rand_goal':
            torch.manual_seed(2023)
            num_agent = gt_eval.size(0)
            cands = traj_refine[eval_mask,:,:,:2]
            rand_ids = torch.randint(0,self.num_modes,(cands.size(0),))
            chosed_marginal_trajectory = cands[torch.arange(cands.size(0)), rand_ids]
            gt_eval = chosed_marginal_trajectory
            
            goal_point = gt_eval[:,-1,:2].detach().clone()
            grad_guid = [goal_point, self.s_mean, self.V_k, self.VT_k, self.latent_mean, self.latent_std]
            cond_gen = None
            
        elif task == 'rand_goal_5s':
            torch.manual_seed(2023) # 2023
            num_agent = gt_eval.size(0)
            cands = traj_refine[eval_mask,:,:,:2]
            rand_ids = torch.randint(0,self.num_modes,(cands.size(0),))
            chosed_marginal_trajectory = cands[torch.arange(cands.size(0)), rand_ids]
            gt_eval = chosed_marginal_trajectory
            
            goal_point = gt_eval[:,50,:2].detach().clone()
            grad_guid = [goal_point, self.s_mean, self.V_k, self.VT_k, self.latent_mean, self.latent_std]
            cond_gen = None
            
        elif task == 'rand_goal_at5s':
            torch.manual_seed(2023)
            num_agent = gt_eval.size(0)
            cands = traj_refine[eval_mask,:,:,:2]
            rand_ids = torch.randint(0,self.num_modes,(cands.size(0),))
            chosed_marginal_trajectory = cands[torch.arange(cands.size(0)), rand_ids]
            gt_eval = chosed_marginal_trajectory
            
            goal_point = gt_eval[:,-1,:2].detach().clone()
            grad_guid = [goal_point, self.s_mean, self.V_k, self.VT_k, self.latent_mean, self.latent_std]
            cond_gen = None
            
        else:
            raise print('unseen tasks.')
            
        
        guid_method = self.guid_method # none ECM ECMR
        guid_inner_loop = 0 # 111 testing
        guid_param = {}
        guid_param['task'] = task
        guid_param['guid_method'] = guid_method
        cost_param = {'cost_param_costl':self.cost_param_costl, 'cost_param_threl':self.cost_param_threl}
        guid_param['cost_param'] = cost_param
        
        sub_folder = 'important_scenes_with_guid_rs2023' 
        os.makedirs('visual/'+sub_folder, exist_ok=True)
        
        
        pred_modes = self.joint_diffusion.sample(num_samples, data = data, scene_enc = scene_enc, 
                                                 mean = mean, std = std, mm = marginal_mode, 
                                                 mmscore = pi.exp()[eval_mask],sampling=self.sampling,
                                                 stride=self.sampling_stride,eval_mask=eval_mask, 
                                                 grad_guid = grad_guid,cond_gen = cond_gen,
                                                 guid_param = guid_param)
        
        if True in torch.isnan(pred_modes):
            print('nan')
            print(data_batch)
            exit()
        
        
        batch_idx = data['agent']['batch'][eval_mask]
        num_scenes = batch_idx[-1].item()+1
        num_agents_per_scene = batch_idx.new_tensor([(batch_idx == i).sum() for i in range(num_scenes)])
        top_modes = torch.randint(0,self.num_modes,(pred_modes.size(0),self.num_modes)).to(pred_modes.device)
        refine_pi = torch.zeros(top_modes.size(0),self.num_modes).to(pred_modes.device)
        
        
        
        pred_modes = self.unnormalize(pred_modes,self.latent_mean, self.latent_std)
        rec_traj = torch.matmul(pred_modes.permute(1,0,2), (self.V_k).unsqueeze(0).repeat(self.num_eval_samples,1,1)) + self.s_mean.unsqueeze(0)
        rec_traj = rec_traj.permute(1,0,2)
        rec_traj = rec_traj.view(rec_traj.size(0), rec_traj.size(1),self.num_future_steps,2)
        
        
        plot = (self.guid_plot == 'plot')
        if plot:
            origin_eval = data['agent']['position'][eval_mask, self.num_historical_steps - 1]
            theta_eval = data['agent']['heading'][eval_mask, self.num_historical_steps - 1]
            cos, sin = theta_eval.cos(), theta_eval.sin()
            rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device=self.device)
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = sin
            rot_mat[:, 1, 0] = -sin
            rot_mat[:, 1, 1] = cos
            rec_traj_world = torch.matmul(rec_traj[:, :, :, :2],
                                    rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
            
            marginal_trajs = traj_refine[eval_mask,:,:,:2]
            marg_traj_world = torch.matmul(marginal_trajs[:, :, :, :2],
                                    rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
            
            marg_traj_world=marg_traj_world.detach().cpu().numpy()
            
            
            gt_eval_world = torch.matmul(gt_eval[:, :, :2],
                                    rot_mat) + origin_eval[:, :2].reshape(-1, 1, 2)
            gt_eval_world = gt_eval_world.detach().cpu().numpy()
            
            goal_point_world = torch.matmul(goal_point[:, None, None, :],
                                            rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
            goal_point_world = goal_point_world.squeeze(1).squeeze(1)
            goal_point_world = goal_point_world.detach().cpu().numpy()


            img_folder = 'images_g'
            os.makedirs(img_folder,exist_ok=True)
            rec_traj_world = rec_traj_world.detach().cpu().numpy()
            for i in range(num_scenes):
                start_id = torch.sum(num_agents_per_scene[:i])
                end_id = torch.sum(num_agents_per_scene[:i+1])
            
                scenario_id = data['scenario_id'][i]
                base_path_to_data = Path('/mnt/hdd1/trajectory_prediction/qcnet/val/raw')
                scenario_folder = base_path_to_data / scenario_id
                
                static_map_path = scenario_folder / f"log_map_archive_{scenario_id}.json"
                scenario_path = scenario_folder / f"scenario_{scenario_id}.parquet"
                # viz_save_path = viz_output_dir / f"{scenario_id}.mp4"

                scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
                static_map = ArgoverseStaticMap.from_json(static_map_path)
                
                viz_output_dir = Path('visual') / sub_folder
                viz_save_path = viz_output_dir / (task + '_b'+ str(data_batch)+'_s'+str(i)+'_'+guid_method+'_'+self.sampling+'.jpg')
                
                additional_traj = {}
                additional_traj['gt'] = gt_eval_world[start_id:end_id]
                additional_traj['goal_point'] = goal_point_world[start_id:end_id]
                
                additional_traj['marg_traj'] = marg_traj_world[start_id:end_id]
                additional_traj['rec_traj'] = rec_traj_world[start_id:end_id]
                
                traj_visible = {}
                traj_visible['gt'] = False
                traj_visible['gt_goal'] = False
                traj_visible['goal_point'] = True
                traj_visible['marg_traj'] = False
                traj_visible['rec_traj'] = True
                
                visualize_scenario_prediction(scenario, static_map, additional_traj, traj_visible, viz_save_path)
                
                
        
        
        if 'rand' in task:
            LDE_gt_trajs = gt_eval
        else:
            LDE_gt_trajs = gt_n[..., :self.output_dim]
            
        batch_agent_idx = data['agent']['batch'][eval_mask]
        self.minJLDE.update(task = task, batch_agent_idx = batch_agent_idx, 
                            pred=rec_traj, target=LDE_gt_trajs, prob=pi_eval,
                        valid_mask=valid_mask_eval)
        self.log('minJLDE', self.minJLDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)
        
        batch_agent_idx = data['agent']['batch'][eval_mask]
        self.meanJLDE.update(task = task, batch_agent_idx = batch_agent_idx, 
                            pred=rec_traj, target=LDE_gt_trajs, prob=pi_eval,
                        valid_mask=valid_mask_eval)
        self.log('meanJLDE', self.meanJLDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)
    
        
        
        
        if print_flag:
            print('minJLDE',self.minJLDE.compute())
            print('meanJLDE',self.meanJLDE.compute())
            
            
            
        batch_agent_idx = data['agent']['batch'][eval_mask]
        self.minJFDEG.update(task = task, batch_agent_idx = batch_agent_idx, 
                            pred=rec_traj, target=goal_point, prob=pi_eval,
                        valid_mask=valid_mask_eval)
        self.log('minJFDEG', self.minJFDEG, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)
        
        batch_agent_idx = data['agent']['batch'][eval_mask]
        self.meanJFDEG.update(task = task, batch_agent_idx = batch_agent_idx, 
                            pred=rec_traj, target=goal_point, prob=pi_eval,
                        valid_mask=valid_mask_eval)
        self.log('meanJFDEG', self.meanJFDEG, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(data['scenario_id']),sync_dist=True)
        
        if print_flag:
            print('minJFDEG',self.minJFDEG.compute())
            print('meanJFDEG',self.meanJFDEG.compute())
        
        print('GPU_incre_memory',np.mean(self.joint_diffusion.GPU_incre_memory))
        print('infer_time_per_step',np.mean(self.joint_diffusion.infer_time_per_step))
        

        

    def test_step(self,
                  data,
                  batch_idx):
        
        
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        
        pred, scene_enc = self.qcnet(data)
        
        if self.output_head:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        device = traj_refine.device
        pi = pred['pi']

        if self.dataset == 'argoverse_v2':
            eval_mask = data['agent']['category'] >= 2
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        
        origin_eval = data['agent']['position'][eval_mask, self.num_historical_steps - 1]
        theta_eval = data['agent']['heading'][eval_mask, self.num_historical_steps - 1]
        cos, sin = theta_eval.cos(), theta_eval.sin()
        rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device=self.device)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = sin
        rot_mat[:, 1, 0] = -sin
        rot_mat[:, 1, 1] = cos
        traj_eval = torch.matmul(traj_refine[eval_mask, :, :, :2],
                                 rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
        
        if self.s_mean == None:
            s_mean = np.load(self.path_pca_s_mean)
            self.s_mean = torch.tensor(s_mean).to(device)
            VT_k = np.load(self.path_pca_VT_k)
            self.VT_k = torch.tensor(VT_k).to(device)
            if self.path_pca_V_k != 'none':
                V_k = np.load(self.path_pca_V_k)
                self.V_k = torch.tensor(V_k).to(device)
            else:
                self.V_k = self.VT_k.transpose(0,1)
            latent_mean = np.load(self.path_pca_latent_mean)
            self.latent_mean = torch.tensor(latent_mean).to(device)
            latent_std = np.load(self.path_pca_latent_std) * 2
            self.latent_std = torch.tensor(latent_std).to(device)
            
        marginal_trajs = traj_refine[eval_mask,:,:,:2]
        marginal_trajs = marginal_trajs.view(marginal_trajs.size(0),self.num_modes,-1)
        marginal_mode = torch.matmul((marginal_trajs-self.s_mean.unsqueeze(1)).permute(1,0,2), self.VT_k.unsqueeze(0).repeat(self.num_modes,1,1))
        marginal_mode = marginal_mode.permute(1,0,2)
        marginal_mode = self.normalize(marginal_mode, self.latent_mean, self.latent_std)

        
        mean = marginal_mode.mean(dim=1)
        std = marginal_mode.std(dim=1) + self.std_reg

        self.joint_diffusion.eval()
        num_samples = self.num_eval_samples

        reverse_steps = 70 # 70
        
        pred_modes = self.joint_diffusion.sample(num_samples, data = data, scene_enc = scene_enc, 
                                                mean = mean, std = std, mm = marginal_mode, 
                                                mmscore = pi.exp()[eval_mask],sampling=self.sampling,
                                                stride=self.sampling_stride,
                                                reverse_steps=reverse_steps,
                                                eval_mask=eval_mask)
        
        mode_diff = pred_modes.unsqueeze(-2).repeat(1,1,self.num_modes,1) - marginal_mode.unsqueeze(1)
        # [num_agents, num_samples, num_modes]
        mode_diff = mode_diff.norm(dim=-1)
        
        # mode_joint_best [num_agents, num_samples]
        mode_joint_best = torch.argmin(mode_diff,dim=-1)
        
        # joint mode clustering
        device = mean.device
        batch_idx = data['agent']['batch'][eval_mask]
        num_scenes = batch_idx[-1].item()+1
        num_agents_per_scene = mode_joint_best.new_tensor([(batch_idx == i).sum() for i in range(num_scenes)])
        top_modes = torch.randint(0,self.num_modes,(pred_modes.size(0),self.num_modes)).to(pred_modes.device)
        refine_pi = torch.zeros(top_modes.size(0),self.num_modes).to(pred_modes.device)
        ### [num_agents, num_samples, 2]
        joint_goal_pts = traj_refine[eval_mask,:,-1,:2]
        for i in range(num_scenes):
            start_id = torch.sum(num_agents_per_scene[:i])
            end_id = torch.sum(num_agents_per_scene[:i+1])
            
            # initialize the modes for single agent
            if end_id - start_id == 1:
                for j in range(self.num_modes):
                    top_modes[start_id:end_id,j] = j
            
            # cluster the modes
            topk_keys = []
            topk_nums = []
            topk_joint_modes = []
            for j in range(num_samples):
                key = 'k'
                for it in mode_joint_best[start_id:end_id,j]:
                    key += str(it.cpu().numpy())
                
                try:
                    idx = topk_keys.index(key)
                    topk_nums[idx] += 1
                except ValueError:
                    topk_keys.append(key)
                    topk_nums.append(1)
                    topk_joint_modes.append(mode_joint_best[start_id:end_id,j:j+1])

            topk_nums = torch.tensor(topk_nums).to(device)
            topk_joint_modes = torch.cat(topk_joint_modes, dim=1)
            
            # sort
            ids = torch.argsort(topk_nums, descending=True)
            
            if self.cluster == 'normal':
                topk_ids = ids[:self.num_modes]
                total_num = torch.sum(topk_nums[topk_ids])
                top_modes[start_id:end_id,:topk_ids.size(0)] = topk_joint_modes[:, topk_ids]
                refine_pi[start_id:end_id,:topk_ids.size(0)] = topk_nums[topk_ids]/total_num
                
            elif self.cluster == 'traj':
                ids_init = ids
                nums_init = topk_nums
                # print('init',nums_init[ids_init])
                
                # based on sort, cluster in the trajectory space
                ### [num_agents, num_samples, 2]
                scene_joint_goal_pts = joint_goal_pts[start_id:end_id]
                num_agents = scene_joint_goal_pts.size(0)
                max_threshold = self.cluster_max_thre
                # print(max_threshold)
                mean_threshold = self.cluster_mean_thre
                queue = torch.ones(ids.shape[0],dtype=torch.int8)
                nms_nums = []
                nms_idx = []
                for j in range(ids.size(0)):
                    if queue[j] == 0:
                        continue
                    queue[j] = 0
                    idx = ids[j]
                    nms_idx.append(idx)
                    temp_nums = topk_nums[idx]
                    temp_modes = topk_joint_modes[:,idx]
                    target_jgp = scene_joint_goal_pts[torch.arange(num_agents), temp_modes]
                                        
                    # queue_ids = torch.arange(j+1,ids.size(0)).to(device)
                    queue_ids = torch.nonzero(queue).squeeze(1).to(device)
                    
                    
                    cand_ids = ids[queue_ids]        
                    num_cands = cand_ids.size(0)
                    cand_modes = topk_joint_modes[:,cand_ids]
                    cand_jgp = scene_joint_goal_pts[torch.arange(num_agents).unsqueeze(-1).repeat(1,num_cands), cand_modes]
                    diff =  torch.norm(target_jgp.unsqueeze(1) - cand_jgp,dim=-1)
                    max_diff = torch.max(diff, dim=0)[0]
                    group_ids = torch.nonzero(max_diff < max_threshold).squeeze(1)
                    cand_nums = topk_nums[cand_ids[group_ids]]
                    queue[queue_ids[group_ids]] = 0
                    nms_nums.append(temp_nums + torch.sum(cand_nums))
                    
                    
                # when clustering into less than num_modes groups, add more groups
                if len(nms_idx) < self.num_modes:
                    for idx in ids_init:
                        if idx not in nms_idx:
                            nms_idx.append(idx)
                            nms_nums.append(nums_init[idx])
                            if len(nms_idx) == self.num_modes:
                                break
                    
                # assign modes
                nms_idx = torch.tensor(nms_idx).to(device)
                nms_nums = torch.tensor(nms_nums).to(device)
                ids = torch.argsort(nms_nums, descending=True)
                topk_ids = ids[:self.num_modes]
                
                lack_nms = self.num_modes - topk_ids.size(0)
                if end_id - start_id == 1 and lack_nms > 0:
                    total_num = torch.sum(nms_nums[topk_ids]) + lack_nms
                    top_modes[start_id:end_id,:topk_ids.size(0)] = topk_joint_modes[:, nms_idx[topk_ids]]
                    refine_pi[start_id:end_id,:topk_ids.size(0)] = nms_nums[topk_ids]/total_num
                    for k in range(lack_nms):
                        for j in range(self.num_modes):
                            if j not in top_modes[start_id:end_id,:topk_ids.size(0)+k]:
                                top_modes[start_id:end_id,topk_ids.size(0)+k]=j
                                refine_pi[start_id:end_id,topk_ids.size(0)+k] = 1/total_num
                    
                    
                else:
                    total_num = torch.sum(nms_nums[topk_ids])
                    top_modes[start_id:end_id,:topk_ids.size(0)] = topk_joint_modes[:, nms_idx[topk_ids]]
                    refine_pi[start_id:end_id,:topk_ids.size(0)] = nms_nums[topk_ids]/total_num
                    
        
        num_mm = top_modes.size(1)
        traj_refine_topk = traj_eval[torch.arange(traj_eval.size(0)).unsqueeze(-1).repeat(1,num_mm), top_modes][...,:self.output_dim]
        traj_eval = traj_refine_topk.cpu().numpy()
        pi_eval = refine_pi.cpu().numpy()
        
        
        if self.dataset == 'argoverse_v2':
            eval_id = list(compress(list(chain(*data['agent']['id'])), eval_mask))
            if isinstance(data, Batch):
                for i in range(data.num_graphs):
                    start_id = torch.sum(num_agents_per_scene[:i])
                    end_id = torch.sum(num_agents_per_scene[:i+1])
                    scenario_trajectories = {}
                    for j in range(start_id,end_id):
                        scenario_trajectories[eval_id[j]] = traj_eval[j]
                    self.test_predictions[data['scenario_id'][i]] = (pi_eval[start_id,:], scenario_trajectories)
            else:
                self.test_predictions[data['scenario_id']] = (pi_eval[0], {eval_id[0]: traj_eval[0]})
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))


    def on_test_end(self):
        if self.dataset == 'argoverse_v2':
            ChallengeSubmission(self.test_predictions).to_parquet(
                Path(self.submission_dir) / f'{self.submission_file_name}.parquet')
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]
    
    def set_opt_lr(self, lr):
        [optimizer], [scheduler] = self.optimizers()
        for g in optimizer.param_groups:
            g['lr'] = 0.001
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('QCNet')
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument('--input_dim', type=int, default=2)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--output_dim', type=int, default=2)
        parser.add_argument('--output_head', action='store_true')
        parser.add_argument('--num_historical_steps', type=int, required=True)
        parser.add_argument('--num_future_steps', type=int, required=True)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--num_recurrent_steps', type=int, required=True)
        parser.add_argument('--num_freq_bands', type=int, default=64)
        parser.add_argument('--num_map_layers', type=int, default=1)
        parser.add_argument('--num_agent_layers', type=int, default=2)
        parser.add_argument('--num_dec_layers', type=int, default=2)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--head_dim', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--pl2pl_radius', type=float, required=True)
        parser.add_argument('--time_span', type=int, default=None)
        parser.add_argument('--pl2a_radius', type=float, required=True)
        parser.add_argument('--a2a_radius', type=float, required=True)
        parser.add_argument('--num_t2m_steps', type=int, default=None)
        parser.add_argument('--pl2m_radius', type=float, required=True)
        parser.add_argument('--a2m_radius', type=float, required=True)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)
        parser.add_argument('--submission_dir', type=str, default='./')
        parser.add_argument('--submission_file_name', type=str, default='submission')
        parser.add_argument('--qcnet_ckpt_path', type=str, required=True)
        parser.add_argument('--num_denoiser_layers', type=int, default=3)
        parser.add_argument('--num_diffusion_steps', type=int, default=10)
        parser.add_argument('--beta_1', type=float, default=1e-4)
        parser.add_argument('--beta_T', type=float, default=0.05)
        parser.add_argument('--diff_type', choices=['opsd', 'opd', 'vd']) 
        parser.add_argument('--sampling', choices=['ddpm','ddim'])
        parser.add_argument('--sampling_stride', type = int, default = 20)
        parser.add_argument('--num_eval_samples', type = int, default = 6)
        parser.add_argument('--eval_mode_error_2', type = int, default = 1)
        parser.add_argument('--choose_best_mode', choices=['FDE', 'ADE'],default = 'ADE')
        parser.add_argument('--train_agent', choices=['all', 'eval'],default = 'all')
        parser.add_argument('--path_pca_s_mean', type = str,default = 'pca/s_mean_10.npy')
        parser.add_argument('--path_pca_VT_k', type = str,default = 'pca/VT_k_10.npy')
        parser.add_argument('--path_pca_latent_mean', type = str,default = 'pca/latent_mean_10.npy')
        parser.add_argument('--path_pca_latent_std', type = str,default = 'pca/latent_std_10.npy')
        parser.add_argument('--m_dim', type = int,default = 10)
        
        return parent_parser
