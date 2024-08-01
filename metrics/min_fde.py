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


class minFDE(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(minFDE, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True) -> None:
        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        pred_topk, _ = topk(self.max_guesses, pred, prob)
        inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=self.device)).argmax(dim=-1)
        self.sum += torch.norm(pred_topk[torch.arange(pred.size(0)), :, inds_last] -
                               target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2),
                               p=2, dim=-1).min(dim=-1)[0].sum()
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
    
    



class BestMinJFDE(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(BestMinJFDE, self).__init__(**kwargs)
        self.max_guesses = max_guesses
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
               batch_agent_idx: torch.Tensor,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True) -> None:

        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        pred_topk, _ = topk(self.max_guesses, pred, prob)
        inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=self.device)).argmax(dim=-1)
        fde = torch.norm(pred_topk[torch.arange(pred.size(0)), :, inds_last] -
                               target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2),
                               p=2, dim=-1).min(dim=-1)[0]

        temp_fde = {}
        for i in range(pred.size(0)):
            sc_id = batch_agent_idx[i].item()
            if sc_id not in temp_fde:
                temp_fde[sc_id] = [fde[i]]
            else:
                temp_fde[sc_id].append(fde[i])
                
        for key in temp_fde.keys():
            self.sum += torch.stack(temp_fde[key]).mean()
            
        self.count += len(temp_fde)
        # print(self.sum / self.count, self.count)
        
    def compute(self) -> torch.Tensor:
        return self.sum / self.count
    
    
    
class minJFDE(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(minJFDE, self).__init__(**kwargs)
        self.max_guesses = max_guesses
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
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
        inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=self.device)).argmax(dim=-1)
        fde = torch.norm(pred_topk[torch.arange(pred.size(0)), :, inds_last] -
                               target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2),
                               p=2, dim=-1)

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
    
    
class meanJFDE(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(meanJFDE, self).__init__(**kwargs)
        self.max_guesses = max_guesses
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
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
        inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=self.device)).argmax(dim=-1)
        fde = torch.norm(pred_topk[torch.arange(pred.size(0)), :, inds_last] -
                               target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2),
                               p=2, dim=-1)

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