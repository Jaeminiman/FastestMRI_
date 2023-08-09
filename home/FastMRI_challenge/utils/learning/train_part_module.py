import shutil
import numpy as np
import torch
import torch.nn as nn
import time
import requests
from tqdm import tqdm
from pathlib import Path
import copy
import pickle
import cv2

from collections import defaultdict
from utils.data.load_data import create_data_loaders_Module
from utils.model.accConvertModule import AccConvertModule

import os
# 임의로 device 넣음
def train_epoch(args, epoch, model, data_loader, optimizer, device):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.
    loss_396 = 0.
    loss_392 = 0.
    cnt_396 = 0
    cnt_392 = 0    
    torch.cuda.empty_cache()
    loss = torch.zeros((1,)).to(device)
    for iter, data in enumerate(data_loader):
        mask_8x, target, kspace_input, kspace_target,maximum, mask_len = data             
        
        mask_8x = mask_8x.cuda(non_blocking=True)
        kspace_input = kspace_input.cuda(non_blocking=True)
        kspace_target = kspace_target.cuda(non_blocking=True)
        
        kspace_output = model(kspace_input, mask_8x)        
        
        temp_loss = torch.norm(kspace_target - kspace_output, p = 2) # l2 norm
        loss += temp_loss
        if(mask_len == 396):
            loss_396 += temp_loss.item()
            cnt_396 += 1
        elif(mask_len == 392):
            loss_392 += temp_loss.item()
            cnt_392 += 1
        
        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item()/2:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()

        if((iter%2 == 0) or (iter == len_loader-1)):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            del loss
            loss = torch.zeros((1,)).to(device)        
        

    loss_396 = loss_396 / cnt_396
    loss_392 = loss_392 / cnt_392
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch, loss_396, loss_392


def validate(args, model, data_loader, device):
    model.eval()
    start = time.perf_counter()

    val_loss = 0
    num_subjects = len(data_loader)
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask_8x, target, kspace_input, kspace_target,maximum, mask_len = data        
        
            mask_8x = mask_8x.cuda(non_blocking=True)
            kspace_input = kspace_input.cuda(non_blocking=True)
            kspace_target = kspace_target.cuda(non_blocking=True)
            
            
            kspace_output = model(kspace_input, mask_8x)        
        
            val_loss += torch.norm(kspace_target - kspace_output, p = 2).item() # l2 norm            
    
    return val_loss, num_subjects, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best, name = ""):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / f'model{name}.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / f'model{name}.pt', exp_dir / f'best_model{name}.pt')



        
def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    torch.cuda.set_device(device)    
    print('Current cuda device: ', torch.cuda.current_device())

    model = AccConvertModule(num_cascades=args.cascade, 
                   chans=args.chans, 
                   sens_chans=args.sens_chans)
    
    model.to(device=device)
    
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    best_val_loss = 1.
    start_epoch = 0

    
    train_loader = create_data_loaders_Module(data_path = args.data_path_train, args = args, shuffle=True)
    val_loader = create_data_loaders_Module(data_path = args.data_path_val, args = args)
    
    val_loss_log = np.empty((0, 2))
    now = time.localtime()
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... acc change Module ...............')

        # 임의로 device, GPU amount 넣음         
        train_loss, train_time, loss_396, loss_392 = train_epoch(args, epoch, model, train_loader, optimizer, device)
        
        val_loss, num_subjects, val_time = validate(args, model, val_loader, device)
        
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = os.path.join(args.val_loss_dir, "val_loss_log")
        np.save(file_path, val_loss_log)
        print(f"loss file saved! {file_path}")

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)
        loss_396 = torch.tensor(loss_396).cuda(non_blocking=True)
        loss_392 = torch.tensor(loss_392).cuda(non_blocking=True)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best, f"_{now.tm_mday}_{now.tm_hour}_{now.tm_min}")
        with open(f'/workspace/result_epoch:{epoch}.txt', 'w') as file:
            file.write(
                f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} Loss_396 : {loss_396:.4g} Loss_392 : {loss_392:.4g}'
                f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
            )
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} Loss_396 : {loss_396:.4g} Loss_392 : {loss_392:.4g}'
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        
        

        
        if epoch == 2:
            save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best, f"_epoch3_{now.tm_mday}_{now.tm_hour}_{now.tm_min}")

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()            
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )
            # 변수를 저장할 딕셔너리 생성 (예시)
            data_to_save = {
                'args': args,
                'best_val_loss': best_val_loss 
            }

            # 변수를 파일에 저장
            with open(f'/workspace/args_cascade:{args.cascade}_chans:{args.chans}_sen_chans:{args.sens_chans}_epoch:{epoch}.pkl', 'wb') as file:
                pickle.dump(data_to_save, file)
