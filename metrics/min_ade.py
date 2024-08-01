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


class minADE(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(minADE, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True,
               min_criterion: str = 'FDE') -> None:
        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        pred_topk, _ = topk(self.max_guesses, pred, prob)
        if min_criterion == 'FDE':
            inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=self.device)).argmax(dim=-1)
            inds_best = torch.norm(
                pred_topk[torch.arange(pred.size(0)), :, inds_last] -
                target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2), p=2, dim=-1).argmin(dim=-1)
            self.sum += ((torch.norm(pred_topk[torch.arange(pred.size(0)), inds_best] - target, p=2, dim=-1) *
                          valid_mask).sum(dim=-1) / valid_mask.sum(dim=-1)).sum()
        elif min_criterion == 'ADE':
            self.sum += ((torch.norm(pred_topk - target.unsqueeze(1), p=2, dim=-1) *
                          valid_mask.unsqueeze(1)).sum(dim=-1).min(dim=-1)[0] / valid_mask.sum(dim=-1)).sum()
        else:
            raise ValueError('{} is not a valid criterion'.format(min_criterion))
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
    
    

class BestMinJADE(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(BestMinJADE, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

    def update(self,
               batch_agent_idx: torch.Tensor,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True,
               min_criterion: str = 'ADE') -> None:
        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        pred_topk, _ = topk(self.max_guesses, pred, prob)
        if min_criterion == 'FDE':
            inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=self.device)).argmax(dim=-1)
            inds_best = torch.norm(
                pred_topk[torch.arange(pred.size(0)), :, inds_last] -
                target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2), p=2, dim=-1).argmin(dim=-1)
            ade = ((torch.norm(pred_topk[torch.arange(pred.size(0)), inds_best] - target, p=2, dim=-1) *
                          valid_mask).sum(dim=-1) / valid_mask.sum(dim=-1))
        elif min_criterion == 'ADE':
            ade = ((torch.norm(pred_topk - target.unsqueeze(1), p=2, dim=-1) *
                          valid_mask.unsqueeze(1)).sum(dim=-1).min(dim=-1)[0] / valid_mask.sum(dim=-1))
        else:
            raise ValueError('{} is not a valid criterion'.format(min_criterion))
        
        temp_ade = {}
        for i in range(pred.size(0)):
            sc_id = batch_agent_idx[i].item()
            if sc_id not in temp_ade:
                temp_ade[sc_id] = [ade[i]]
            else:
                temp_ade[sc_id].append(ade[i])
                
        for key in temp_ade.keys():
            self.sum += torch.stack(temp_ade[key]).mean()
            
        self.count += len(temp_ade)
        
        # print(self.sum / self.count, self.count)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count


class minJADE(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(minJADE, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

    def update(self,
               batch_agent_idx: torch.Tensor,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True,
               min_criterion: str = 'FDE') -> None:
        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        # pred_topk, _ = topk(self.max_guesses, pred, prob)
        pred_topk = pred
        # pred_topk [num_agents, num_modes, 60, 2]
        # valid_mask [num_agents, 60]
        # ade [num_agents, num_modes]
        ade = ((torch.norm(pred_topk - target.unsqueeze(1), p=2, dim=-1) * 
                valid_mask.unsqueeze(1)).sum(dim=-1) / valid_mask.sum(dim=-1).unsqueeze(1))
        
        
        temp_ade = {}
        for i in range(pred.size(0)):
            sc_id = batch_agent_idx[i].item()
            if sc_id not in temp_ade:
                temp_ade[sc_id] = [ade[i]]
            else:
                temp_ade[sc_id].append(ade[i])
                
        
        for key in temp_ade.keys():
            temp_ade[key] = torch.stack(temp_ade[key]).mean(dim=0).min()
            
        for key in temp_ade.keys():
            self.sum += temp_ade[key]
            
        self.count += len(temp_ade)
        
        # print(self.sum / self.count, self.count)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
    
    
class meanJADE(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(meanJADE, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

    def update(self,
               batch_agent_idx: torch.Tensor,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True,
               min_criterion: str = 'FDE') -> None:
        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        # pred_topk, _ = topk(self.max_guesses, pred, prob)
        pred_topk = pred
        # pred_topk [num_agents, num_modes, 60, 2]
        # valid_mask [num_agents, 60]
        # ade [num_agents, num_modes]
        ade = ((torch.norm(pred_topk - target.unsqueeze(1), p=2, dim=-1) * 
                valid_mask.unsqueeze(1)).sum(dim=-1) / valid_mask.sum(dim=-1).unsqueeze(1))
        
        
        temp_ade = {}
        for i in range(pred.size(0)):
            sc_id = batch_agent_idx[i].item()
            if sc_id not in temp_ade:
                temp_ade[sc_id] = [ade[i]]
            else:
                temp_ade[sc_id].append(ade[i])
                
        
        for key in temp_ade.keys():
            temp_ade[key] = torch.stack(temp_ade[key]).mean(dim=0).mean()
            
        for key in temp_ade.keys():
            self.sum += temp_ade[key]
            
        self.count += len(temp_ade)
        
        # print(self.sum / self.count, self.count)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
    
    
    
    



class minJLDE(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(minJLDE, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

    def update(self,
               task: str,
               batch_agent_idx: torch.Tensor,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True,
               min_criterion: str = 'FDE') -> None:
        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        # pred_topk, _ = topk(self.max_guesses, pred, prob)
        pred_topk = pred
        # pred_topk [num_agents, num_modes, 60, 2]
        # valid_mask [num_agents, 60]
        # ade [num_agents, num_modes]
        if 'goal_at3s' in task:
            index = 30
        elif 'goal_at5s' in task:
            index = 50
        else:
            index = 59
           
        ade = torch.norm(pred_topk[:,:,:index+1,:].unsqueeze(3) - target.unsqueeze(1).unsqueeze(2), dim=-1).min(-1)[0].mean(-1)
        
        
        temp_ade = {}
        for i in range(pred.size(0)):
            sc_id = batch_agent_idx[i].item()
            if sc_id not in temp_ade:
                temp_ade[sc_id] = [ade[i]]
            else:
                temp_ade[sc_id].append(ade[i])
                
        
        for key in temp_ade.keys():
            temp_ade[key] = torch.stack(temp_ade[key]).mean(dim=0).min()
            
        for key in temp_ade.keys():
            self.sum += temp_ade[key]
            
        self.count += len(temp_ade)
        
        # print(self.sum / self.count, self.count)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
    
    
class meanJLDE(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(meanJLDE, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

    def update(self,
               task: str,
               batch_agent_idx: torch.Tensor,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True,
               min_criterion: str = 'FDE') -> None:
        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        # pred_topk, _ = topk(self.max_guesses, pred, prob)
        pred_topk = pred
        # pred_topk [num_agents, num_modes, 60, 2]
        # valid_mask [num_agents, 60]
        # ade [num_agents, num_modes]
        if 'goal_at3s' in task:
            index = 30
        elif 'goal_at5s' in task:
            index = 50
        else:
            index = 59
           
        ade = torch.norm(pred_topk[:,:,:index+1,:].unsqueeze(3) - target.unsqueeze(1).unsqueeze(2), dim=-1).min(-1)[0].mean(-1)
        
        temp_ade = {}
        for i in range(pred.size(0)):
            sc_id = batch_agent_idx[i].item()
            if sc_id not in temp_ade:
                temp_ade[sc_id] = [ade[i]]
            else:
                temp_ade[sc_id].append(ade[i])
                
        
        for key in temp_ade.keys():
            temp_ade[key] = torch.stack(temp_ade[key]).mean(dim=0).mean()
            
        for key in temp_ade.keys():
            self.sum += temp_ade[key]
            
        self.count += len(temp_ade)
        
        # print(self.sum / self.count, self.count)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
    
    
    
class minJFDEG(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(minJFDEG, self).__init__(**kwargs)
        self.max_guesses = max_guesses
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
               task: str,
               batch_agent_idx: torch.Tensor,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True) -> None:

        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        # pred_topk, _ = topk(self.max_guesses, pred, prob)
        pred_topk = pred
        # print(pred_topk.size())     
        if 'goal_at3s' in task:
            index = 30
        elif 'goal_at5s' in task:
            index = 50
        else:
            index = 59
           
        fde = torch.norm(pred_topk[:,:,index,:] - target.unsqueeze(1), dim=-1)

        temp_fde = {}
        for i in range(pred.size(0)):
            sc_id = batch_agent_idx[i].item()
            if sc_id not in temp_fde:
                temp_fde[sc_id] = [fde[i]]
            else:
                temp_fde[sc_id].append(fde[i])
                
        for key in temp_fde.keys():
            # print(torch.stack(temp_fde[key]).mean(dim = 0).min())
            self.sum += torch.stack(temp_fde[key]).mean(dim = 0).min()
            
        self.count += len(temp_fde)
        # print(self.sum / self.count, self.count)
        
    def compute(self) -> torch.Tensor:
        return self.sum / self.count
    
    
class meanJFDEG(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(meanJFDEG, self).__init__(**kwargs)
        self.max_guesses = max_guesses
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
               task: str,
               batch_agent_idx: torch.Tensor,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True) -> None:

        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        # pred_topk, _ = topk(self.max_guesses, pred, prob)
        pred_topk = pred
        # print(pred_topk.size())
        if 'goal_at3s' in task:
            index = 30
        elif 'goal_at5s' in task:
            index = 50
        else:
            index = 59
           
        fde = torch.norm(pred_topk[:,:,index,:] - target.unsqueeze(1), dim=-1)

        temp_fde = {}
        for i in range(pred.size(0)):
            sc_id = batch_agent_idx[i].item()
            if sc_id not in temp_fde:
                temp_fde[sc_id] = [fde[i]]
            else:
                temp_fde[sc_id].append(fde[i])
                
        for key in temp_fde.keys():
            self.sum += torch.stack(temp_fde[key]).mean(dim = 0).mean()
            
        self.count += len(temp_fde)
        # print(self.sum / self.count, self.count)
        
    def compute(self) -> torch.Tensor:
        return self.sum / self.count