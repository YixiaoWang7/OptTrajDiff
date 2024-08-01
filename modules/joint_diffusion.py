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
import math
from typing import Dict, List, Mapping, Optional
from pynvml import *
nvmlInit()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch_geometric.utils import dense_to_sparse
import copy
import time

from layers import AttentionLayer
from layers import FourierEmbedding
from layers import MLPLayer
from utils import angle_between_2d_vectors
from utils import bipartite_dense_to_sparse
from utils import weight_init
from utils import wrap_angle
import numpy as np

class JointDiffusion(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.diff_type = args.diff_type
        self.net = Denoiser(
            dataset=args.dataset,
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            output_head=args.output_head,
            num_historical_steps=args.num_historical_steps,
            num_future_steps=args.num_future_steps,
            num_samples=6,
            num_recurrent_steps=args.num_recurrent_steps,
            num_t2m_steps=args.num_t2m_steps,
            pl2m_radius=args.pl2m_radius,
            a2m_radius=args.a2m_radius,
            num_freq_bands=args.num_freq_bands,
            num_layers=args.num_denoiser_layers,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            dropout=args.dropout,
            diff_type=args.diff_type,
            m_dim = args.m_dim
        )
        
        self.var_sched = VarianceSchedule(
                num_steps = args.num_diffusion_steps,
                beta_1 = args.beta_1,
                beta_T = args.beta_T,
                mode = 'linear'
            )
        self.infer_time_per_step = []
        self.GPU_incre_memory = []

    def get_loss(self, 
                 m,
                 data: HeteroData,
                 scene_enc: Mapping[str, torch.Tensor],
                 mean = None,
                 std = None,
                 mm = None,
                 mmscore = None,
                 eval_mask = None) -> Dict[str, torch.Tensor]:
        
        if self.diff_type == 'vd':
            return self.get_loss_vd(m,data,scene_enc,mean,std,mm,mmscore,eval_mask)
        elif self.diff_type == 'opd':
            return self.get_loss_opd(m,data,scene_enc,mean,std,mm,mmscore,eval_mask)
        elif self.diff_type == 'opsd':
            return self.get_loss_opsd(m,data,scene_enc,mean,std,mm,mmscore,eval_mask)
        
    def get_loss_vd(self, 
                 m,
                 data: HeteroData,
                 scene_enc: Mapping[str, torch.Tensor],
                 mean = None,
                 std = None,
                 mm = None,
                 mmscore = None,
                 eval_mask = None) -> Dict[str, torch.Tensor]:
        # m: [num_agents, d_latent]
        x_0 = m
        device = mean.device
        
        num_agents = data['agent']['position'][eval_mask].size(0)
        batch_idx = data['agent']['batch'][eval_mask]
        num_scenes = batch_idx[-1].item()+1
        
        t = torch.tensor(self.var_sched.uniform_sample_t(num_scenes)).to(device).unsqueeze(0).repeat(num_agents,1)
        t = t[torch.arange(num_agents), batch_idx]

        alpha_bar = self.var_sched.alpha_bars[t][:,None].to(device)
        beta = self.var_sched.betas[t][:,None].to(device)
        
        c0 = torch.sqrt(alpha_bar)
        c1 = torch.sqrt(1 - alpha_bar)
        
        e_rand = torch.randn_like(x_0).to(device)
        
        x_t = c0 * x_0 + c1 * e_rand
            
        g_theta = self.net(x_t, beta, data, scene_enc, num_samples = 1, mm = mm, mmscore = mmscore, eval_mask=eval_mask).squeeze(1)
        
        loss = ((e_rand- g_theta) ** 2).mean()
        
        return loss
    
    def get_loss_opsd(self, 
                 m,
                 data: HeteroData,
                 scene_enc: Mapping[str, torch.Tensor],
                 mean = None,
                 std = None,
                 mm = None,
                 mmscore = None,
                 eval_mask = None) -> Dict[str, torch.Tensor]:
        # m: [num_agents, d_latent]

        x_0 = m
        device = m.device
        num_agents = data['agent']['position'][eval_mask].size(0)
        batch_idx = data['agent']['batch'][eval_mask]
        num_scenes = batch_idx[-1].item()+1
        
        t = torch.tensor(self.var_sched.uniform_sample_t(num_scenes)).to(device).unsqueeze(0).repeat(num_agents,1)
        t = t[torch.arange(num_agents), batch_idx]
        alpha_bar = self.var_sched.alpha_bars[t][:,None].to(device)
        beta = self.var_sched.betas[t][:,None].to(device)
        

        c0 = torch.sqrt(alpha_bar)
        c1 = torch.sqrt(1-alpha_bar)
        kt = self.var_sched.kt[t][:,None].to(device)
        
        std = std
        e_rand = torch.randn_like(x_0).to(device) * std
        s_t = kt * mean
        
        x_t = c0 * x_0 + s_t + c1 * e_rand
        # target = (x_t-c0*x_0)/c1
        target = e_rand
        g_theta = self.net(x_t, beta, data, scene_enc, num_samples = 1, mm = mm, mmscore = mmscore, eval_mask=eval_mask).squeeze(1)
        
        loss = (1/std**2*(target-g_theta)**2).mean()
        
        return loss
    
    def get_loss_opd(self, 
                 m,
                 data: HeteroData,
                 scene_enc: Mapping[str, torch.Tensor],
                 mean = None,
                 std = None,
                 mm = None,
                 mmscore = None,
                 eval_mask = None) -> Dict[str, torch.Tensor]:
        # m: [num_agents, d_latent]

        x_0 = m

        device = m.device
        num_agents = data['agent']['position'][eval_mask].size(0)
        batch_idx = data['agent']['batch'][eval_mask]
        num_scenes = batch_idx[-1].item()+1
        
        t = torch.tensor(self.var_sched.uniform_sample_t(num_scenes)).to(device).unsqueeze(0).repeat(num_agents,1)
        t = t[torch.arange(num_agents), batch_idx]
        alpha_bar = self.var_sched.alpha_bars[t][:,None].to(device)
        beta = self.var_sched.betas[t][:,None].to(device)
        
        c0 = torch.sqrt(alpha_bar)
        c1 = torch.sqrt(1 - alpha_bar)
        
        e_rand = torch.randn_like(x_0).to(device) * std
        
        x_t = c0 * x_0 + c1 * e_rand
            
        g_theta = self.net(x_t, beta, data, scene_enc, num_samples = 1, mm = mm, mmscore = mmscore, eval_mask=eval_mask).squeeze(1)
        
        loss = ((e_rand- g_theta) ** 2).mean()
        
        return loss
    

    def sample(self, 
               num_samples: int,
               data: HeteroData,
               scene_enc: Mapping[str, torch.Tensor],
               mean = None,
               std = None,
               mm = None,
               mmscore = None,
               if_output_diffusion_process = False,
               start_data = None,
               reverse_steps = None,
               eval_mask = None,
               sampling="ddpm", 
               stride=20,
               grad_guid = None,
               cond_gen = None,
               guid_param = None,
               uc = None,
               ) -> Dict[str, torch.Tensor]:
        
        if self.diff_type == 'opsd':
            
            return self.sample_opsd(num_samples, data, scene_enc, mean, std, 
                                    mm, mmscore, if_output_diffusion_process, start_data,reverse_steps,
                                    eval_mask, sampling, stride, grad_guid)
        elif self.diff_type == 'opd':
            return self.sample_opd(num_samples, data, scene_enc, mean, std, 
                                    mm, mmscore, if_output_diffusion_process, start_data,reverse_steps,
                                    eval_mask, sampling, stride, grad_guid,cond_gen = cond_gen, guid_param = guid_param, uc = uc)
        elif self.diff_type == 'vd':
            return self.sample_vd(num_samples, data, scene_enc, mean, std, 
                                    mm, mmscore, if_output_diffusion_process, start_data,reverse_steps,
                                    eval_mask, sampling, stride)
        
    def sample_opsd(self, 
               num_samples: int,
               data: HeteroData,
               scene_enc: Mapping[str, torch.Tensor],
               mean = None,
               std = None,
               mm = None,
               mmscore = None,
               if_output_diffusion_process = False,
               start_data = None,
               reverse_steps = None,
               eval_mask = None,
               sampling="ddpm", 
               stride=20,
               grad_guid = None,
               ) -> Dict[str, torch.Tensor]:

        device = mean.device

        num_agents = mean.size(0)
        num_dim = mean.size(1)

        mean = mean.unsqueeze(1)
        std = std.unsqueeze(1)
        # std = torch.ones_like(std)
        e_rand = torch.randn([num_agents, num_samples, num_dim]).to(device) * std


        s_T = self.var_sched.kt[-1].to(device)* mean
        if start_data == None:
            x_T = e_rand + s_T
        else:
            c0 = torch.sqrt(self.var_sched.alpha_bars[-1]).to(device)
            c1 = torch.sqrt(1-self.var_sched.alpha_bars[-1]).to(device)
            kt = self.var_sched.kt[-1].to(device)
            s_t = kt * mean
            x_T = c0 * start_data.unsqueeze(1) + s_t + c1 * e_rand
            
        x_t_list = [x_T]
        torch.cuda.empty_cache()
        
        for t in range(self.var_sched.num_steps, 0, -stride):
            z = torch.randn_like(x_T) * std if t > 1 else torch.zeros_like(x_T)

            beta = self.var_sched.betas[t]
            st = self.var_sched.kt[t] * mean
            st_1 = self.var_sched.kt[t-1] * mean
            inv_sqrt_alpha = self.var_sched.inv_sqrt_alpha[t]
            co_g = self.var_sched.co_g[t]
            co_st = self.var_sched.co_st[t]
            co_z = self.var_sched.co_z[t]
            
            x_t = x_t_list[-1]
            with torch.no_grad():
                beta = beta.to(device).repeat(num_agents * num_samples).unsqueeze(-1)
                g_theta = self.net(copy.deepcopy(x_t), beta, data, scene_enc, num_samples = num_samples, mm = mm, mmscore = mmscore, eval_mask=eval_mask)

            if sampling == 'ddpm':
                x_next = inv_sqrt_alpha*(x_t - co_g * g_theta) - co_st * st + st_1 + co_z * z
            else:
                alpha_bar = self.var_sched.alpha_bars[t]
                inv_1ma = (1-alpha_bar).sqrt()
                alpha_bar_next = self.var_sched.alpha_bars[t-stride]
                inv_1ma_next = (1-alpha_bar_next).sqrt()
                s_next = self.var_sched.kt[t-stride] * mean
                x_next = inv_sqrt_alpha*(x_t - inv_1ma * g_theta) + s_next + inv_1ma_next*(g_theta-st/inv_1ma)
                
            if True in torch.isnan(x_next):
                print('nan:',t)

            x_t_list.append(x_next.detach())
            if not if_output_diffusion_process:
                x_t_list.pop(0)
            
        if if_output_diffusion_process:
            return x_t_list
        else:
            return x_t_list[-1]
        
    def lat2traj(self, latent, V_k, s_mean, num_samples):
        return torch.matmul(latent, V_k.unsqueeze(0).repeat(num_samples,1,1)) + s_mean.unsqueeze(0)
    
            
    
    def sample_opd(self, 
               num_samples: int,
               data: HeteroData,
               scene_enc: Mapping[str, torch.Tensor],
               mean = None,
               std = None,
               mm = None,
               mmscore = None,
               if_output_diffusion_process = False,
               start_data = None,
               reverse_steps = None,
               eval_mask = None,
               sampling="ddpm", 
               stride=20,
               grad_guid = None,
               cond_gen = None,
               guid_param = None,
               uc = None,
               ) -> Dict[str, torch.Tensor]:
        
        if grad_guid != None:
            task = guid_param['task']
            guid_method = guid_param['guid_method']
            cost_param_costl = guid_param['cost_param']['cost_param_costl']
            cost_param_threl = guid_param['cost_param']['cost_param_threl']
            [guid_label, s_mean, V_k, VT_k, latent_mean, latent_std] = grad_guid
        
        if reverse_steps is None:
            reverse_steps = self.var_sched.num_steps
        
        device = mean.device

        num_agents = mean.size(0)
        num_dim = mean.size(1)

        mean = mean.unsqueeze(1)
        std = std.unsqueeze(1)
        # std = torch.ones_like(std)
        e_rand = torch.randn([num_agents, num_samples, num_dim]).to(device) * std

        s_T = torch.sqrt(self.var_sched.alpha_bars[reverse_steps].to(device))* mean
        if start_data == None:
            # c1 = torch.sqrt(1-self.var_sched.alpha_bars[reverse_steps]).to(device)
            # print(c1)
            # c1 = 1
            c1 = 1
            x_T = c1 * e_rand + s_T
        else:
            c0 = torch.sqrt(self.var_sched.alpha_bars[reverse_steps]).to(device)
            c1 = torch.sqrt(1-self.var_sched.alpha_bars[reverse_steps]).to(device)

            if start_data.dim() == 2:
                x_T = c0 * start_data.unsqueeze(1) + c1 * e_rand
            elif start_data.dim() == 3:
                x_T = c0 * start_data + c1 * e_rand
            
        x_t_list = [x_T]
        torch.cuda.empty_cache()
        pt = time.time()
        h = nvmlDeviceGetHandleByIndex(2)
        info = nvmlDeviceGetMemoryInfo(h)
        pu = info.used/ (1024 ** 3)
        for t in range(reverse_steps, 0, -stride):

            z = torch.randn_like(x_T) * std if t > 1 else torch.zeros_like(x_T)
            beta = self.var_sched.betas[t]
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            alpha_bar_next = self.var_sched.alpha_bars[t-stride]
            c0 = 1 / torch.sqrt(alpha)
            c1 = (1-alpha) / torch.sqrt(1 - alpha_bar)
            sigma = self.var_sched.get_sigmas(t, 0)
            
            x_t = x_t_list[-1]
            if cond_gen != None:
                [idx, target_mode] = cond_gen
                x_t[idx,:,:] = target_mode.unsqueeze(0).repeat(num_samples,1)
            
            if grad_guid == None:
                with torch.no_grad():
                    beta_emb = beta.to(device).repeat(num_agents * num_samples).unsqueeze(-1)
                    g_theta = self.net(copy.deepcopy(x_t), beta_emb, data, scene_enc, num_samples = num_samples, mm = mm, mmscore = mmscore, eval_mask=eval_mask)
            
                if sampling == 'ddpm':
                    x_next = c0 * (x_t - c1 * g_theta) + sigma * z
                elif sampling == 'ddim':
                    x0_t = (x_t - g_theta * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                    x_next = alpha_bar_next.sqrt() * x0_t + (1 - alpha_bar_next).sqrt() * g_theta
                
            else:
                if sampling == 'ddpm':
                    print('guided sampling not support ddpm yet.')
                    exit()
                elif sampling == 'ddim':

                    if guid_method == 'none':
                        with torch.no_grad():
                            beta_emb = beta.to(device).repeat(num_agents * num_samples).unsqueeze(-1)
                            g_theta = self.net(copy.deepcopy(x_t), beta_emb, data, scene_enc, num_samples = num_samples, mm = mm, mmscore = mmscore, eval_mask=eval_mask)
                    
                        x0_t = (x_t - g_theta * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                        x_next = alpha_bar_next.sqrt() * x0_t + (1 - alpha_bar_next).sqrt() * g_theta
                    
                    if guid_method == 'ECMR':
                        with torch.no_grad():
                            beta_emb = beta.to(device).repeat(num_agents * num_samples).unsqueeze(-1)
                            g_theta = self.net(copy.deepcopy(x_t), beta_emb, data, scene_enc, num_samples = num_samples, mm = mm, mmscore = mmscore, eval_mask=eval_mask)
                                
                        with torch.inference_mode(False):
                            ### Marginal Mapping
                            temp_e = g_theta.clone().detach().float()
                            temp_x_t = x_t.clone().detach().float()
                            temp_x_0 = (temp_x_t - torch.sqrt(1-alpha_bar).clone() * temp_e) / torch.sqrt(alpha_bar).clone()
                            temp_x_0.requires_grad = False
                            
                            temp_x_0_unnor = temp_x_0 * (latent_std.unsqueeze(0).unsqueeze(0)+0.1) + latent_mean.unsqueeze(0).unsqueeze(0)
                            rec_traj = torch.matmul(temp_x_0_unnor.permute(1,0,2), V_k.unsqueeze(0).repeat(num_samples,1,1)) + s_mean.unsqueeze(0)
                            rec_traj = rec_traj.permute(1,0,2)
                            rec_traj = rec_traj.view(rec_traj.size(0), rec_traj.size(1),60,2)
                            diff = self.task_diff(task, rec_traj, guid_label)

                            temp_mm = mm.clone().detach().float().requires_grad_(True)
                            temp_mm_unnor = temp_mm * (latent_std.unsqueeze(0).unsqueeze(0)+0.1) + latent_mean.unsqueeze(0).unsqueeze(0)
                            marg_traj = torch.matmul(temp_mm_unnor.permute(1,0,2), V_k.unsqueeze(0).repeat(temp_mm.size(1),1,1)) + s_mean.unsqueeze(0)
                            marg_traj = marg_traj.permute(1,0,2)
                            marg_traj = marg_traj.view(marg_traj.size(0), marg_traj.size(1),60,2)
                            marg_diff = self.task_diff(task, marg_traj, guid_label)
                            dis = diff
                            mm_dis = marg_diff
                            
                            min_idx = torch.argmin(mm_dis,dim=-1)
                            dis_mm_mean = dis > mm_dis[torch.arange(mm_dis.size(0)), min_idx].unsqueeze(1)
                            
                            larg_idx = torch.nonzero(dis_mm_mean)
                            
                            temp_mm = (mm[torch.arange(mm.size(0)), min_idx]).detach().clone()
                            temp_mm = temp_mm.unsqueeze(1).repeat(1,num_samples,1)
                            temp_x_0[larg_idx[:,0],larg_idx[:,1]] = temp_mm[larg_idx[:,0],larg_idx[:,1]]
                            
                            
                            ### Gradident
                            temp_x_0.requires_grad = True
                            temp_x_0_unnor = temp_x_0 * (latent_std.unsqueeze(0).unsqueeze(0)+0.1) + latent_mean.unsqueeze(0).unsqueeze(0)
                            rec_traj = torch.matmul(temp_x_0_unnor.permute(1,0,2), V_k.unsqueeze(0).repeat(num_samples,1,1)) + s_mean.unsqueeze(0)
                            rec_traj = rec_traj.permute(1,0,2)
                            rec_traj = rec_traj.view(rec_traj.size(0), rec_traj.size(1),60,2)
                            diff = self.task_diff(task, rec_traj, guid_label)
                            error = diff.mean()
                            error.backward()
                            # note it is the gradient of x0
                            grad = temp_x_0.grad
                            
                        grad = grad * cost_param_costl
                        scale = 1 * cost_param_threl
                        grad = torch.clip(grad, min = -scale, max = scale)
                        m_0 = temp_x_0 - grad
                                                
                        x_next = alpha_bar_next.sqrt() * m_0 + (1 - alpha_bar_next).sqrt() * g_theta
                                            

                    if guid_method == 'ECM':       
                        with torch.no_grad():
                            beta_emb = beta.to(device).repeat(num_agents * num_samples).unsqueeze(-1)
                            g_theta = self.net(copy.deepcopy(x_t), beta_emb, data, scene_enc, num_samples = num_samples, mm = mm, mmscore = mmscore, eval_mask=eval_mask)
                         
                        with torch.inference_mode(False):
                            temp_e = g_theta.clone().detach().float()
                            temp_x_t = x_t.clone().detach().float()
                            temp_x_0 = (temp_x_t - torch.sqrt(1-alpha_bar).clone() * temp_e) / torch.sqrt(alpha_bar).clone()
                            temp_x_0.requires_grad = True
                            temp_x_0_unnor = temp_x_0 * (latent_std.unsqueeze(0).unsqueeze(0)+0.1) + latent_mean.unsqueeze(0).unsqueeze(0)
                            rec_traj = torch.matmul(temp_x_0_unnor.permute(1,0,2), V_k.unsqueeze(0).repeat(num_samples,1,1)) + s_mean.unsqueeze(0)
                            rec_traj = rec_traj.permute(1,0,2)
                            rec_traj = rec_traj.view(rec_traj.size(0), rec_traj.size(1),60,2)
                            diff = self.task_diff(task, rec_traj, guid_label)
                            error = diff.mean()
                            error.backward()
                            # note it is the gradient of x0
                            grad = temp_x_0.grad
                            
                        grad = grad * cost_param_costl
                        scale = 1 * cost_param_threl
                        grad = torch.clip(grad, min = -scale, max = scale)
                        m_0 = temp_x_0 - grad
                        
                        # g_theta = torch.randn_like(x_T) * std
                        x_next = alpha_bar_next.sqrt() * m_0 + (1 - alpha_bar_next).sqrt() * g_theta

            if True in torch.isnan(x_next):
                print('nan:',t)
            x_t_list.append(x_next.detach())
            if not if_output_diffusion_process:
                x_t_list.pop(0)
            
        cost_time = (time.time() - pt)/10
        h = nvmlDeviceGetHandleByIndex(2)
        info = nvmlDeviceGetMemoryInfo(h)
        cu = info.used/ (1024 ** 3)
        cost_u = cu - pu
        self.GPU_incre_memory.append(cost_u)
        self.infer_time_per_step.append(cost_time)
            
        if if_output_diffusion_process:
            return x_t_list
        else:
            return x_t_list[-1]
        
    def task_diff(self, task, traj, label):
        if 'goal' in task:
            goal_pt = label
            if 'goal_at5s' in task:
                index = 50
            else:
                index = 59
                
            goal_diff = ((traj[:,:,index,:] - goal_pt.unsqueeze(1))**2)

            goal_diff[...,1] = goal_diff[...,1] * 1
            goal_diff = goal_diff.mean(-1)
            return goal_diff
            
        
    def sample_vd(self, 
               num_samples: int,
               data: HeteroData,
               scene_enc: Mapping[str, torch.Tensor],
               mean = None,
               std = None,
               mm = None,
               mmscore = None,
               if_output_diffusion_process = False,
               start_data = None,
               reverse_steps = None,
               eval_mask = None,
               sampling="ddpm", 
               stride=20,
               ) -> Dict[str, torch.Tensor]:
        
        if reverse_steps is None:
            reverse_steps = self.var_sched.num_steps
            
        device = mean.device

        num_agents = mean.size(0)
        num_dim = mean.size(1)

        e_rand = torch.randn([num_agents, num_samples, num_dim]).to(device)

        if start_data == None:
            x_T = e_rand
        else:
            c0 = torch.sqrt(self.var_sched.alpha_bars[reverse_steps]).to(device)
            c1 = torch.sqrt(1-self.var_sched.alpha_bars[reverse_steps]).to(device)
            x_T = c0 * start_data.unsqueeze(1) + c1 * e_rand
            
        x_t_list = [x_T]
        torch.cuda.empty_cache()
            
        for t in range(reverse_steps, 0, -stride):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            beta = self.var_sched.betas[t]
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            alpha_bar_next = self.var_sched.alpha_bars[t-stride]
            c0 = 1 / torch.sqrt(alpha)
            c1 = (1-alpha) / torch.sqrt(1 - alpha_bar)
            sigma = self.var_sched.get_sigmas(t, 0)
            
            x_t = x_t_list[-1]
            with torch.no_grad():
                beta = beta.to(device).repeat(num_agents * num_samples).unsqueeze(-1)
                g_theta = self.net(copy.deepcopy(x_t), beta, data, scene_enc, num_samples = num_samples, mm = mm, mmscore = mmscore, eval_mask=eval_mask)


            if sampling == 'ddpm':
                x_next = c0 * (x_t - c1 * g_theta) + sigma * z
            elif sampling == 'ddim':
                x0_t = (x_t - g_theta * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                x_next = alpha_bar_next.sqrt() * x0_t + (1 - alpha_bar_next).sqrt() * g_theta
            
            if True in torch.isnan(x_next):
                print('nan:',t)
            x_t_list.append(x_next.detach())
            if not if_output_diffusion_process:
                x_t_list.pop(0)
            
        if if_output_diffusion_process:
            return x_t_list
        else:
            return x_t_list[-1]


        


class Denoiser(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 output_head: bool,
                 num_historical_steps: int,
                 num_future_steps: int,
                 num_samples: int,
                 num_recurrent_steps: int,
                 num_t2m_steps: Optional[int],
                 pl2m_radius: float,
                 a2m_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 diff_type: str,
                 m_dim: int) -> None:
        super(Denoiser, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_samples = num_samples
        self.num_recurrent_steps = num_recurrent_steps
        self.num_t2m_steps = num_t2m_steps if num_t2m_steps is not None else num_historical_steps
        self.pl2m_radius = pl2m_radius
        self.a2m_radius = a2m_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.diff_type = diff_type
        self.m_dim = m_dim

        input_dim_r_t = 4
        input_dim_r_pl2m = 3
        input_dim_r_a2m = 3
        
        self.proj_in = nn.Linear(self.m_dim, self.hidden_dim)
        self.proj_in_mm = nn.Linear(self.m_dim, self.hidden_dim)
        
        self.proj_out = nn.Linear(self.hidden_dim, self.m_dim)
        
        self.r_t2m_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pl2m_emb = FourierEmbedding(input_dim=input_dim_r_pl2m, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a2m_emb = FourierEmbedding(input_dim=input_dim_r_a2m, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        
        noise_dim = 1
        self.noise_emb = FourierEmbedding(input_dim=noise_dim, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        
        score_dim = 1
        self.score_emb = FourierEmbedding(input_dim=score_dim, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)

        self.t2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.m2m_propose_attn_layer = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.ms2m_propose_attn_layer = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=False) for _ in range(num_layers)]
        )
        
        
        # self.mlp1 = MyMLP(d_in = hidden_dim, d_h = hidden_dim, d_out = hidden_dim)
        # self.mlp2 = MyMLP(d_in = hidden_dim, d_h = hidden_dim, d_out = hidden_dim)
        # self.linear1 = nn.Linear(hidden_dim*3,hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim*3,hidden_dim)
        # self.to_out = MyMLP(d_in = hidden_dim, d_h = hidden_dim, d_out = hidden_dim)
        
        self.mlp1 = SkipMLP(d_model=hidden_dim)
        self.mlp2 = SkipMLP(d_model=hidden_dim)
        self.linear1 = nn.Linear(hidden_dim*3,hidden_dim)
        self.linear2 = nn.Linear(hidden_dim*3,hidden_dim)
        self.to_out = SkipMLP(d_model=hidden_dim)
        
        # self.to_out = SkipMLP(d_model=hidden_dim)
        
        self.apply(weight_init)
    
    def forward(self,
                m,
                beta,
                data: HeteroData,
                scene_enc: Mapping[str, torch.Tensor],
                num_samples: int,
                mm,
                mmscore,
                eval_mask,) -> Dict[str, torch.Tensor]:
        self.num_samples = num_samples
        pos_m = data['agent']['position'][eval_mask][:, self.num_historical_steps - 1, :self.input_dim]
        head_m = data['agent']['heading'][eval_mask][:, self.num_historical_steps - 1]
        head_vector_m = torch.stack([head_m.cos(), head_m.sin()], dim=-1)

        # scene_enc['x_a'] [num_agents, his_steps, 128]
        # x_t [num_agents x his_steps, 128]
        x_t = scene_enc['x_a'][eval_mask].reshape(-1, self.hidden_dim)
        
        # scene_enc['x_pl'] [num_pls, his_steps, 128]
        # x_pl [num_pls x num_samples, 128] take the encoding for the current time step
        x_pl = scene_enc['x_pl'][:, self.num_historical_steps - 1].repeat(self.num_samples, 1)
        
        # x_a [num_agents x num_samples, 128]
        x_a = scene_enc['x_a'][eval_mask][:, -1].repeat(self.num_samples, 1)
        
        # mask_src [num_agents, his_steps] 
        mask_src = data['agent']['valid_mask'][eval_mask][:, :self.num_historical_steps].contiguous()
        # only counts for the recent 20 steps
        mask_src[:, :self.num_historical_steps - self.num_t2m_steps] = False
        
        # test whether there exists last position at 110.
        # mask_dst [num_agents, num_samples]
        mask_dst = data['agent']['predict_mask'][eval_mask].any(dim=-1, keepdim=True).repeat(1, self.num_samples)
        
        pos_t = data['agent']['position'][eval_mask][:, :self.num_historical_steps, :self.input_dim].reshape(-1, self.input_dim)
        head_t = data['agent']['heading'][eval_mask][:, :self.num_historical_steps].reshape(-1)
        edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst[:, -1:].unsqueeze(1))
        rel_pos_t2m = pos_t[edge_index_t2m[0]] - pos_m[edge_index_t2m[1]]
        rel_head_t2m = wrap_angle(head_t[edge_index_t2m[0]] - head_m[edge_index_t2m[1]])
        r_t2m = torch.stack(
            [torch.norm(rel_pos_t2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_t2m[1]], nbr_vector=rel_pos_t2m[:, :2]),
             rel_head_t2m,
             (edge_index_t2m[0] % self.num_historical_steps) - self.num_historical_steps + 1], dim=-1)
        r_t2m = self.r_t2m_emb(continuous_inputs=r_t2m, categorical_embs=None)
        edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst.unsqueeze(1))
        r_t2m = r_t2m.repeat_interleave(repeats=self.num_samples, dim=0)

        pos_pl = data['map_polygon']['position'][:, :self.input_dim]
        orient_pl = data['map_polygon']['orientation']
        edge_index_pl2m = radius(
            x=pos_m[:, :2],
            y=pos_pl[:, :2],
            r=self.pl2m_radius,
            batch_x=data['agent']['batch'][eval_mask] if isinstance(data, Batch) else None,
            batch_y=data['map_polygon']['batch'] if isinstance(data, Batch) else None,
            max_num_neighbors=300)
        edge_index_pl2m = edge_index_pl2m[:, mask_dst[edge_index_pl2m[1], 0]]
        rel_pos_pl2m = pos_pl[edge_index_pl2m[0]] - pos_m[edge_index_pl2m[1]]
        rel_orient_pl2m = wrap_angle(orient_pl[edge_index_pl2m[0]] - head_m[edge_index_pl2m[1]])
        r_pl2m = torch.stack(
            [torch.norm(rel_pos_pl2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_pl2m[1]], nbr_vector=rel_pos_pl2m[:, :2]),
             rel_orient_pl2m], dim=-1)
        r_pl2m = self.r_pl2m_emb(continuous_inputs=r_pl2m, categorical_embs=None)
        edge_index_pl2m = torch.cat([edge_index_pl2m + i * edge_index_pl2m.new_tensor(
            [[data['map_polygon']['num_nodes']], [pos_m.size(0)]]) for i in range(self.num_samples)], dim=1)
        r_pl2m = r_pl2m.repeat(self.num_samples, 1)

        edge_index_a2m = radius_graph(
            x=pos_m[:, :2],
            r=self.a2m_radius,
            batch=data['agent']['batch'][eval_mask] if isinstance(data, Batch) else None,
            loop=False,
            max_num_neighbors=300)
        edge_index_a2m = edge_index_a2m[:, mask_src[:, -1][edge_index_a2m[0]] & mask_dst[edge_index_a2m[1], 0]]
        rel_pos_a2m = pos_m[edge_index_a2m[0]] - pos_m[edge_index_a2m[1]]
        rel_head_a2m = wrap_angle(head_m[edge_index_a2m[0]] - head_m[edge_index_a2m[1]])
        r_a2m = torch.stack(
            [torch.norm(rel_pos_a2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_a2m[1]], nbr_vector=rel_pos_a2m[:, :2]),
             rel_head_a2m], dim=-1)
        r_a2m = self.r_a2m_emb(continuous_inputs=r_a2m, categorical_embs=None)
        edge_index_a2m = torch.cat(
            [edge_index_a2m + i * edge_index_a2m.new_tensor([pos_m.size(0)]) for i in
             range(self.num_samples)], dim=1)
        r_a2m = r_a2m.repeat(self.num_samples, 1)

        beta_emb = self.noise_emb(beta)
        
        # mm [num_agents, 6, dim]
        # mmscore [num_agents, 6]
        mmscore = mmscore
        num_modes = mmscore.size(1)
        mmscore_emb = self.score_emb(mmscore.reshape(-1,1))
        
        # mm_emb [num_agents, 6, dim]
        mm = self.proj_in_mm(mm)
        # ms_emb = mm.reshape(-1, self.hidden_dim) + mmscore_emb
        # flat_ms_emb = ms_emb.unsqueeze(1).repeat(1,self.num_samples,1)
        # flat_ms_emb = flat_ms_emb.reshape(-1, self.hidden_dim)
        num_agents = mmscore.size(0)
        ms_emb = (mm.reshape(-1, self.hidden_dim) + mmscore_emb).view(num_agents, num_modes,-1)
        flat_ms_emb = ms_emb.unsqueeze(1).repeat(1,self.num_samples,1,1)
        flat_ms_emb = flat_ms_emb.reshape(-1, self.hidden_dim)

        
        # print(flat_mm_emb.size(), m.reshape(-1, self.hidden_dim).size(),mmscore_emb.size(), num_modes)
        total_num = flat_ms_emb.size(0)
        edge_index_ms2m = torch.stack([torch.arange(total_num),torch.arange(total_num)//num_modes]).long().to(m.device
                                                                                                              )
        # print(edge_index_ms2m)
        # exit()
        
        # m [num_agents, num_samples, dim]
        # print(m.size())
        m = self.proj_in(m)
        m = m.reshape(-1, self.hidden_dim)
        m_sum = m
        for i in range(self.num_layers):
            m = m + beta_emb
            m = self.t2m_propose_attn_layers[i]((x_t, m), r_t2m, edge_index_t2m)
            # [num_samples, num_agents, dim]
            m = m.reshape(-1, self.num_samples, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            m = self.pl2m_propose_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
            m = self.a2m_propose_attn_layers[i]((x_a, m), r_a2m, edge_index_a2m)
            m = self.m2m_propose_attn_layer[i]((m, m), r_a2m, edge_index_a2m)
            m = m.reshape(self.num_samples, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)   
            
            m = self.ms2m_propose_attn_layer[i]((flat_ms_emb, m), None, edge_index_ms2m)
            m_sum += m
        
        # out1 = self.linear1(torch.cat([m,x_a,beta_emb],dim=-1))
        # out1 = self.mlp1(out1)
        
        # out2 = m + out1
        # out2 = self.linear2(torch.cat([out2,x_a,beta_emb],dim=-1))
        # out2 = self.mlp2(out2)

        # out = m + self.to_out(m+out1+out2)
        
        # out = m_sum
        
        
        
        out = self.to_out(m_sum)
        
        # out = self.to_out(m_sum)
        out = out.reshape(-1, self.num_samples, self.hidden_dim)
        
        return self.proj_out(out)
        
class MyMLP(torch.nn.Module):
    def __init__(self, d_in = 128, d_h = 128, d_out = 128, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.linear_in = torch.nn.Linear(d_in, d_h)
        self.ac = act_layer()
        self.linear_out = torch.nn.Linear(d_h, d_out)
        self.drop = nn.Dropout(drop)

    def forward(self, x):          
        out = self.linear_in(x)
        out = self.ac(out)
        out = self.drop(out)
        out = self.linear_out(out)
        return out

class VarianceSchedule(nn.Module):

    def __init__(self, num_steps, mode='linear',beta_1=1e-4, beta_T=5e-2,cosine_s=8e-3):
        super().__init__()
        assert mode in ('linear', 'cosine')
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == 'cosine':
            timesteps = (
            torch.arange(num_steps + 1) / num_steps + cosine_s
            )
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding
        
        alphas = 1 - betas
        
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()
        # print('alpha_bars',alpha_bars)
        # print('beta / sqrt 1 - alpha_bars',betas/torch.sqrt(1-alpha_bars))
        # print('1/ sqrt 1 - alpha_bars',1/torch.sqrt(1-alpha_bars))
        # print(' 1 - alpha_bars',(1-alpha_bars[:-1])/(1-alpha_bars[1:]))
        # exit()
        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)
        
        # kt
        sqrt_alpha_bars = torch.sqrt(alpha_bars)
        # kt = sqrt_alpha_bars * (1 - sqrt_alpha_bars) # quadratic-shift
        # kt = torch.linspace(0, 0, steps=num_steps+1)
        kt = 1 - sqrt_alpha_bars # shifted diffusion
        # print('kt', kt)
        # exit()
        # kt[0] = 0
        self.register_buffer('kt', kt)
        
        inv_sqrt_alpha = 1 / torch.sqrt(alphas)
        co_g = betas / torch.sqrt(1-alpha_bars)
        co_st = torch.sqrt(alphas[1:]) * (1-alpha_bars[:-1])/(1-alpha_bars[1:])
        co_st = torch.cat([torch.tensor([0]),co_st])
        co_z = torch.sqrt((1-alpha_bars[:-1])/(1-alpha_bars[1:])*betas[1:])
        co_z = torch.cat([torch.tensor([0]),co_z])
        self.register_buffer('inv_sqrt_alpha', inv_sqrt_alpha)
        self.register_buffer('co_g', co_g)
        self.register_buffer('co_st', co_st)
        self.register_buffer('co_z', co_z)
        

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas
    
class SkipMLP(torch.nn.Module):
    def __init__(self, d_model = 128, act_layer=nn.GELU):
        super().__init__()
        self.linear = torch.nn.Linear(d_model, d_model)
        self.ac = act_layer()
        self.norm1 = nn.LayerNorm([d_model])
        self.norm2 = nn.LayerNorm([d_model])

    def forward(self, x):          
        out = x + self.ac(self.linear(x))
        out = self.norm2(x + self.norm1(out))
        return out
