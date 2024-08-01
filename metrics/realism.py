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
from typing import Optional

import torch
from torchmetrics import Metric

from metrics.utils import topk
from metrics.utils import valid_filter


class KinematicFeasibleRate(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(KinematicFeasibleRate, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

    def update(self,
               batch_agent_idx: torch.Tensor,
               pred: torch.Tensor,
               max_acc: torch.Tensor,
               max_lateral_acc: torch.Tensor,
               max_curvature: torch.Tensor,
               ) -> None:
        
        num_samples = pred.size(1)
        
        # max acc
        space = 10
        pred = pred[:,:,::space,:2]
        pred_vel = torch.norm(pred[:,:,1:,:2] - pred[:,:,:-1,:2], dim=-1)
        pred_acc = torch.abs(pred_vel[:,:,1:]- pred_vel[:,:,:-1])
        max_pred_acc = pred_acc.max(-1)[0]
        max_vel_succ = (max_pred_acc < max_acc)

        # minimal steering curvature
        pred_curvature = []
        pred_lateral_acc = []
        for i in range(1,pred.size(2)-1):
            p1 = pred[:,:,i-1,:2]
            p2 = pred[:,:,i,:2]
            p3 = pred[:,:,i+1,:2]
            area2 = (p2[:,:,0]-p1[:,:,0])*(p3[:,:,1]-p1[:,:,1]) - (p2[:,:,1]-p1[:,:,1])*(p3[:,:,0]-p1[:,:,0])
            l1 = torch.norm(p1-p2, dim=-1)
            l2 = torch.norm(p1-p3, dim=-1)
            l3 = torch.norm(p3-p2, dim=-1)
            mask = (l1 > 3) & (l2 > 3) & (l3 > 3)
            tc = torch.abs(2*area2/(l1*l2*l3))
            pred_curvature.append((tc * mask).unsqueeze(-1))
            pred_lateral_acc.append((pred_vel[:,:,i-1]**2).unsqueeze(-1) * pred_curvature[-1])
            
            
        pred_curvature = torch.cat(pred_curvature, dim=-1)
        max_pred_curvature = pred_curvature.max(-1)[0]
        max_curvature_succ = (max_pred_curvature < max_curvature)
        
        pred_lateral_acc = torch.cat(pred_lateral_acc, dim=-1)
        max_pred_lateral_acc = pred_lateral_acc.max(-1)[0]
        max_lateral_acc_succ = (max_pred_lateral_acc < max_lateral_acc)
        
        
        succ = max_vel_succ & max_curvature_succ & max_lateral_acc_succ
        
        num_scenes = batch_agent_idx[-1].item()+1
        num_agents_per_scene = batch_agent_idx.new_tensor([(batch_agent_idx == i).sum() for i in range(num_scenes)])

        for i in range(num_scenes):
            start_id = torch.sum(num_agents_per_scene[:i])
            end_id = torch.sum(num_agents_per_scene[:i+1])
            scene_succ = succ[start_id:end_id].sum(dim=0)
            scene_succ = scene_succ == (end_id - start_id)
            self.sum += scene_succ.sum() / num_samples
        self.count += num_scenes

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
    
 