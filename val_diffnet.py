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
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,4'

from argparse import ArgumentParser

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from datasets import ArgoverseV2Dataset
from predictors import QCNet, DiffNet
from transforms import TargetBuilder



if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4) 
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=str, default="4,")
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--sampling', choices=['ddpm','ddim'],default='ddpm')
    parser.add_argument('--sampling_stride', type = int, default = 20)
    parser.add_argument('--num_eval_samples', type = int, default = 6)
    parser.add_argument('--eval_mode_error_2', type = int, default = 1)
    
    parser.add_argument('--ex_opm', type=int, default=0)
    parser.add_argument('--std_state', choices=['est', 'one'],default = 'est')
    parser.add_argument('--cluster', choices=['normal', 'traj'],default = 'traj')
    parser.add_argument('--cluster_max_thre', type = float,default = 2.5)
    parser.add_argument('--cluster_mean_thre', type = float,default = 2.5)
    
    parser.add_argument('--guid_sampling', choices=['no_guid', 'guid'],default = 'no_guid')
    parser.add_argument('--guid_task', type=str,default = 'none')
    parser.add_argument('--guid_method', choices=['none', 'ECM', 'ECMR'],default = 'none')
    parser.add_argument('--guid_plot',choices=['no_plot', 'plot'],default = 'no_plot')
    parser.add_argument('--std_reg',type = float, default=0.1)
    parser.add_argument('--path_pca_V_k', type = str,default = 'none')
    
    parser.add_argument('--network_mode', choices=['val', 'test'],default = 'val')
    parser.add_argument('--submission_file_name', type=str, default='submission')
    
    parser.add_argument('--cond_norm', type = int, default = 0)
    
    parser.add_argument('--cost_param_costl', type = float, default = 1.0)
    parser.add_argument('--cost_param_threl', type = float, default = 1.0)
    
    args = parser.parse_args()

    model = {
        'DiffNet': DiffNet,
    }['DiffNet'].load_from_checkpoint(checkpoint_path=args.ckpt_path)
    
    model.add_extra_param(args)
    
    
    model.sampling = args.sampling
    model.sampling_stride = args.sampling_stride
    model.check_param()
    model.num_eval_samples = args.num_eval_samples
    model.eval_mode_error_2 = args.eval_mode_error_2
    val_dataset = {
        'argoverse_v2': ArgoverseV2Dataset,
    }[model.dataset](root=args.root, split=args.network_mode,
                     transform=TargetBuilder(model.num_historical_steps, model.num_future_steps))
    dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)
    
    # for data in dataloader:
    #     model.validation_step(data.to('cuda:4'),1)
    #     break
    
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, strategy='ddp')
    if args.network_mode == 'val':
        a = trainer.validate(model, dataloader)

        import csv
        file_name = 'exps/'+args.guid_task+'_s'+str(model.num_eval_samples)+'_'+args.guid_method + '_costl_'+str(args.cost_param_costl)+'_threl_'+str(args.cost_param_threl)+'.csv'
        if not os.path.exists(file_name):
            a = a[0]
            data = list(a.values())
            column_names = list(a.keys())
            row_data = dict(zip(column_names, data))
            
            with open(file_name, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=column_names)
                
                writer.writeheader()
                
                writer.writerow(row_data)

            print(f'Data has been written to {file_name}')
            
        import csv
        file_name = 'exps/efficiency_'+args.guid_task+'_'+args.guid_method + '_costl_'+str(args.cost_param_costl)+'_threl_'+str(args.cost_param_threl)+'.csv'
        if not os.path.exists(file_name):
            column_names = ['time per step','GPU memory']
            row_data = dict(zip(column_names, [np.mean(model.joint_diffusion.infer_time_per_step),np.mean(model.joint_diffusion.GPU_incre_memory)]))
            
            with open(file_name, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=column_names)
                
                writer.writeheader()
                
                writer.writerow(row_data)

            print(f'Data has been written to {file_name}')

        
    elif args.network_mode == 'test':
        model.submission_file_name = args.submission_file_name
        trainer.test(model, dataloader)

    # print(model.targetVelminError.compute())
    # print('KinematicConfortRate',model.KinematicConfortRate.compute())

    # print('KinematicFeasibleRate',self.KinematicFeasibleRate.compute())
    # print('targetVelmeanError',self.targetVelmeanError.compute())
    # import csv

    # # Open the file in write mode
    # with open('example.csv', mode='a', newline='') as file:
    #     writer = csv.writer(file)
        
    #     # Write the header
    #     writer.writerow(['Name', 'Age', 'City'])
        
        
    #     # Write multiple rows
    #     rows = [
    #         ['Alice', 29, 'New York'],
    #         ['Bob', 25, 'Los Angeles'],
    #         ['Charlie', 35, 'Chicago']
    #     ]
    #     writer.writerows(rows)