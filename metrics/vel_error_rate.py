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


class meanVelRate(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(meanVelRate, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

    def update(self,
               batch_agent_idx: torch.Tensor,
               pred: torch.Tensor,
               mean_vel: torch.Tensor,
               threshold: torch.Tensor,
               ) -> None:
        
        pred_mean_vel = (pred[:,:,1:,:2] - pred[:,:,:-1,:2]).mean(-2)
        mean_vel = mean_vel.unsqueeze(1)
        succ = (pred_mean_vel > (mean_vel - threshold)) & (pred_mean_vel < (mean_vel + threshold))
        succ = (succ.sum(-1)) == 2
        num_samples = pred.size(1)
        
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
    
    
class maxVelRate(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(maxVelRate, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

    def update(self,
               batch_agent_idx: torch.Tensor,
               pred: torch.Tensor,
               max_vel: torch.Tensor,
               threshold: torch.Tensor,
               ) -> None:
        
        pred_max_vel_squared = ((pred[:,:,1:,:2] - pred[:,:,:-1,:2])**2).max(-2)[0]
        max_vel = max_vel.unsqueeze(1)
        succ = pred_max_vel_squared < (max_vel + threshold)**2
        succ = (succ.sum(-1)) == 2
        num_samples = pred.size(1)
        
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


class targetVelmeanError(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(targetVelmeanError, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

    def update(self,
               batch_agent_idx: torch.Tensor,
               pred: torch.Tensor,
               target_vel: torch.Tensor,
               threshold: torch.Tensor,
               ) -> None:
        
        # space = 10
        # sparse_pred = pred[:,:,::space,:2]
        # pred_vel = torch.norm(sparse_pred[:,:,1:,:2] - sparse_pred[:,:,:-1,:2], dim=-1)
        # target_vel = target_vel.unsqueeze(1).unsqueeze(1)
        # # succ = (pred_vel > (target_vel - threshold)) & (pred_vel < (target_vel + threshold))
        # # error = torch.abs(pred_vel-target_vel) * (1-succ)
        # error = torch.abs(pred_vel-target_vel) 
        # error = error.mean(-1)
        
        space = 10
        sparse_pred = pred[:,:,::space,:2]
        pred_vel = torch.norm(sparse_pred[:,:,1:,:] - sparse_pred[:,:,:-1,:], dim=-1)[:,:,-1]
        target_vel = target_vel.unsqueeze(1)
        error = torch.abs(pred_vel-target_vel) 
        error = error.mean(-1)
        
        num_scenes = batch_agent_idx[-1].item()+1
        num_agents_per_scene = batch_agent_idx.new_tensor([(batch_agent_idx == i).sum() for i in range(num_scenes)])

        for i in range(num_scenes):
            start_id = torch.sum(num_agents_per_scene[:i])
            end_id = torch.sum(num_agents_per_scene[:i+1])
            scene_error = error[start_id:end_id].mean()
            self.sum += scene_error
        self.count += num_scenes

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
    
class targetVelminError(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(targetVelminError, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

    def update(self,
               batch_agent_idx: torch.Tensor,
               pred: torch.Tensor,
               target_vel: torch.Tensor,
               threshold: torch.Tensor,
               ) -> None:
        
        # space = 10
        # sparse_pred = pred[:,:,::space,:2]
        # pred_vel = torch.norm(sparse_pred[:,:,1:,:2] - sparse_pred[:,:,:-1,:2], dim=-1)
        # target_vel = target_vel.unsqueeze(1).unsqueeze(1)
        # # succ = (pred_vel > (target_vel - threshold)) & (pred_vel < (target_vel + threshold))
        # # error = torch.abs(pred_vel-target_vel) * (1-succ)
        # error = torch.abs(pred_vel-target_vel) 
        # error = error.mean(-1)
        
        
        space = 10
        sparse_pred = pred[:,:,::space,:2]
        pred_vel = torch.norm(sparse_pred[:,:,1:,:2] - sparse_pred[:,:,:-1,:2], dim=-1)[:,:,-1]
        target_vel = target_vel.unsqueeze(1)
        error = torch.abs(pred_vel-target_vel) 
        error = error.mean(-1)
        
        num_scenes = batch_agent_idx[-1].item()+1
        num_agents_per_scene = batch_agent_idx.new_tensor([(batch_agent_idx == i).sum() for i in range(num_scenes)])

        for i in range(num_scenes):
            start_id = torch.sum(num_agents_per_scene[:i])
            end_id = torch.sum(num_agents_per_scene[:i+1])
            scene_error = error[start_id:end_id].min(-1)[0]
            scene_error = scene_error.mean()
            self.sum += scene_error
        self.count += num_scenes

    def compute(self) -> torch.Tensor:
        return self.sum / self.count