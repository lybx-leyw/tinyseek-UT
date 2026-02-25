from model.tinyseek import shared_TinySeek as TinySeek
from tools import Vocab

import time
import torch
import random
import numpy as np
import json
import math
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim

class SharedPreTrainingDataset(Dataset):
    def __init__(self,text_tensor,pad_idx,n_layers):
        self.input_ids = text_tensor
        self.pad_idx = pad_idx
        self.n_layers = n_layers
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self,idx):
        input_seq = self.input_ids[idx]
        labels = torch.roll(input_seq,shifts=-1,dims=-1)
        labels[-1] = self.pad_idx      
        
        return input_seq,labels

def count_non_shared_parameters(model):
    seen_tensors = set()
    total_non_shared = 0
    
    for _, param in model.named_parameters():
        if param.requires_grad:
            # 检查这个参数张量是否已经被计算过（用于检测共享参数）
            if id(param) not in seen_tensors:
                seen_tensors.add(id(param))
                total_non_shared += param.numel()
    
    return total_non_shared

def shared_pre_train(
        config,vocab_data_path,vocab_trg_path,
        json_data_path,max_len,batch_size,
        max_epochs=1,num_workers=4,accumulation_steps=8,
        warmup_index=471,keep_temp_index=1300,
        sava_frq=10,last_index=0,n_of_samples=3000,
        n_of_samplings=5,print_frq=20,
        conlude_epoch=4,init_lr=5e-4,
        seed=42,prefetch_factor=2,NO_loss_p=False
        ): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    vocab = Vocab(vocab_data_path,vocab_trg_path,config['vocab_size']) 
    model = TinySeek(config,NO_loss_p=NO_loss_p).to(config['device'])
    # 加载上一次训练的模型参数
    try:
        try:
            model = torch.load(f"out\\Shared_TinySeek_Pre{last_index}_1.pkl").to(config['device'])
        except:
            model.load_state_dict(torch.load(f"out\\Shared_TinySeek_Pre{last_index}_1.pkl"))
        print(f"成功加载模型参数：Shared_TinySeek_Pre{last_index}_1.pkl")
    except FileNotFoundError:
        print(f"未找到模型参数：Shared_TinySeek_Pre{last_index}_1.pkl，开始新的训练")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总共享参数量: {total_params:,}")
    non_shared_params = count_non_shared_parameters(model)
    print(f"\n总非共享参数量: {non_shared_params:,}")
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    if config['device'] == 'cuda':
        scaler = torch.amp.GradScaler("cuda")

    # 随机抽取n条数据进行经验回放训练
    tran_idx = list(range(1,1413000))
    random.shuffle(tran_idx)
    for input_number in range(n_of_samplings):
        all_ids = []
        # 采样n_of_samples条数据进行训练
        if input_number*n_of_samples >= 1413000:
            print("已完成所有数据的采样训练")
            break
        # 跳过已经训练过的数据
        if input_number*n_of_samples <= last_index:
            continue
        random_indices = set(tran_idx[input_number*n_of_samples:(input_number+1)*n_of_samples])
        with open(json_data_path, "r", encoding='utf-8') as data_file:
            for line_index, line in enumerate(data_file, 1):
                if line_index not in random_indices:
                    continue
                data = json.loads(line)
                conversations = data['text']
                ids = vocab.encode(conversations,max_len=max_len)
                all_ids.append(ids)
            all_data = torch.cat(all_ids,dim=0)
            shared_train_epoch(
                config=config,
                data=all_data,
                batch_size=batch_size,
                max_epochs=max_epochs,
                num_workers=num_workers,
                model=model,
                optimizer=optimizer,
                scheduler=True,
                number=input_number*n_of_samples,
                accumulation_steps=accumulation_steps,
                scaler=scaler if config['device'] == 'cuda' else None,
                warmup_index=warmup_index,
                keep_index=keep_temp_index,
                save_frq=sava_frq,
                print_frq=print_frq,
                conlude_epoch=conlude_epoch,
                n_of_samples=n_of_samples,
                n_of_sampling=n_of_samplings,
                prefetch_factor=prefetch_factor
            )
            all_ids = []
    torch.save(model.state_dict(),f"out\\Shared_TinySeek_Pre_final.pkl")

def shared_train_epoch(config,data,batch_size,max_epochs,
                num_workers,model,optimizer,scheduler,
                number,accumulation_steps=8,scaler=None,
                warmup_index=471,keep_index=1300,save_frq=10,print_frq=20,
                conlude_epoch=4,n_of_samples=300,n_of_sampling=4710,
                prefetch_factor=2):
    if number//n_of_samples <= keep_index:
        current_index = number//n_of_samples+1
        warmup_factor = min(1.0, current_index/warmup_index)
        for param_group in optimizer.param_groups:
            param_group['lr'] = 5e-4*warmup_factor
        scheduler = False
    optimizer.zero_grad()
    device = config['device']
    dataset = SharedPreTrainingDataset(data,pad_idx=config['pad_idx'],n_layers=config['n_layer'])
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True, 
        drop_last=True,  
        pin_memory=True,       
        prefetch_factor=prefetch_factor,    
        num_workers=num_workers,
        persistent_workers=True
    )
    # 开始训练
    for epoch in range(max_epochs):
        timer_epoch = time.perf_counter()
        for batch_idx,(batch_src,batch_trg) in enumerate(train_dataloader):
            def init_train(batch_idx,batch_src,batch_trg):
                timer = time.perf_counter()

                batch_src = batch_src.to(device)
                batch_trg = batch_trg.to(device)

                model.train()
                if scaler is not None:
                    with torch.amp.autocast('cuda',dtype=torch.float16):  
                        cross_loss_1,cross_loss_2,Lexp = model(batch_src,batch_trg)
                        loss = cross_loss_1 + cross_loss_2 + config['alpha']*Lexp                
                        scaled_loss = loss / accumulation_steps
                        scaler.scale(scaled_loss).backward(retain_graph=True) 
                        
                    if (batch_idx + 1) % accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        
                        # 学习率调度
                        if scheduler is True:
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = max(param_group['lr']*0.9995,1e-7)
                        else:
                            current_index = number//n_of_samples+1
                            warmup_factor = min(1.0, current_index/warmup_index)
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = 5e-4*warmup_factor
                else:
                    cross_loss_1,cross_loss_2,Lexp = model(batch_src,batch_trg)
                    loss = cross_loss_1 + cross_loss_2 + config['alpha']*Lexp                 
                    scaled_loss = loss / accumulation_steps  
                    scaled_loss.backward()
                    
                    if (batch_idx + 1) % accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        if scheduler is True:
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = max(param_group['lr']*0.9995,1e-7)
                        else:
                            current_index = number//n_of_samples+1
                            warmup_factor = min(1.0, current_index/warmup_index)
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = 5e-4*warmup_factor

                current_lr = optimizer.param_groups[0]['lr']

                timer_end = time.perf_counter()
                time_batch = timer_end-timer
                last_batch_time = time_batch*(len(train_dataloader)-batch_idx-1)
                last_index_time = time_batch*len(train_dataloader)*(n_of_sampling-number//n_of_samples-1)
                last_time = last_batch_time+last_index_time

                if (batch_idx+1) % print_frq == 0:   
                    log_line = (f'Epoch {epoch+1:03d}/{max_epochs} | '
                                f'Batch {batch_idx+1:04d}/{len(train_dataloader)} | '
                                f'Index {number//n_of_samples:04d}/{n_of_sampling} | '
                                f'Loss: {loss:8.4f} | '
                                f'Loss_p: {cross_loss_1:8.2f} | '
                                f'PPx: {math.exp(cross_loss_2):8.2f} | '
                                f'Lexp: {Lexp:8.2f} | '
                                f'LR: {current_lr:8.4e} | '
                                f'Device: {device} | '
                                f'Batch_t: {time_batch:5.2f}s | '
                                f'Remaining: {last_time/3600:5.2f}h')
                    print(log_line)
                    with open("log.txt","a",encoding='utf-8') as log:
                        log.write(f'{log_line}\n')
                
            init_train(batch_idx,batch_src,batch_trg)
        
        timer_epoch_end = time.perf_counter()
        epoch_time = timer_epoch_end-timer_epoch
        last_index_time = epoch_time*(n_of_sampling-number//n_of_samples-1)
        if number//n_of_samples % conlude_epoch == 0:
            print(f"| Estimated Remaining:{last_index_time/3600:.2f} hours | ")
            with open("log.txt","a",encoding='utf-8') as log:
                log.write(f"| Estimated Remaining:{last_index_time/3600:.2f} hours | \n")
        if number//n_of_samples % save_frq == 0:
            torch.save(model.state_dict(),f"out\\Shared_TinySeek_Pre{number}_{epoch+1}.pkl")