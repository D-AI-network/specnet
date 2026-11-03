
import argparse
import csv
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import cfg
from data import (MilanWindowDataset, denormalize, load_milan, normalize,
                  set_seed, split_train_val, train_only_min_max)
from model import SPECNet


def frequency_loss(y, prediction):
    err = (torch.fft.rfft2(y.permute(0,3,1,2))
           - torch.fft.rfft2(prediction.permute(0,3,1,2))).abs()
    h,w=y.shape[1:3]
    rows=torch.linspace(-1,1,h,device=y.device)
    cols=torch.linspace(-1,1,w,device=y.device)
    radius=torch.sqrt(rows[:,None]**2+cols[None,:]**2)[:,:w//2+1]
    weight=torch.exp(-3*radius)
    return (err*weight[None,None,:,:]).mean()


def pinball(y,pred,tau):
    delta=y-pred
    return torch.maximum(tau*delta,(tau-1)*delta).mean()


def compute_loss(y, pred, tv):
    huber=nn.functional.huber_loss(pred,y,delta=cfg.LOSS_DELTA)
    freq=frequency_loss(y,pred)
    quantile=sum(pinball(y,pred,t) for t in (0.2,0.5,0.8))/3
    # Keep all four terms: identical weights to the paper's stated objective.
    return huber + cfg.FREQ_LOSS_W*freq + cfg.PINBALL_W*quantile + cfg.TV_W*tv


class Metrics:
    def __init__(self,channels):
        self.channels=channels
        self.squared=torch.zeros(channels,dtype=torch.float64)
        self.absolute=torch.zeros(channels,dtype=torch.float64)
        self.points=0

    def add(self,actual,prediction):
        error=(prediction-actual).detach().to(torch.float64).cpu()
        self.squared+=(error*error).sum(dim=(0,1,2))
        self.absolute+=error.abs().sum(dim=(0,1,2))
        self.points+=error.shape[0]*error.shape[1]*error.shape[2]

    def result(self):
        if self.points==0:
            raise ValueError('No observations were evaluated')
        mse=self.squared/self.points
        mae=self.absolute/self.points
        return {
            'rmse':math.sqrt(float(self.squared.sum()/(self.points*self.channels))),
            'mae':float(self.absolute.sum()/(self.points*self.channels)),
            'per_channel_rmse':mse.sqrt().tolist(),
            'per_channel_mae':mae.tolist(),
            'evaluated_steps':self.points,
        }


def to_device(batch,device):
    return batch.to(device,non_blocking=(device.type=='cuda'))


def optimize_epoch(model,loader,minimum,maximum,optimizer,device,micro_batch):
    model.train()
    weighted_loss=0.0
    num_examples=0
    for x,y in loader:
        batch_n=len(x)
        optimizer.zero_grad(set_to_none=True)
        batch_loss=0.0
        for xx,yy in zip(x.split(micro_batch),y.split(micro_batch)):
            xx=to_device(xx,device)
            yy=to_device(yy,device)
            normalized_x=normalize(xx,minimum,maximum)
            normalized_y=normalize(yy,minimum[:,0],maximum[:,0])
            pred,tv=model(normalized_x,raw_input=xx)
            loss=compute_loss(normalized_y,pred,tv)
            # Accumulate a mathematically correct global mean for the batch.
            (loss*(len(xx)/batch_n)).backward()
            batch_loss+=float(loss.detach())*len(xx)/batch_n
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        weighted_loss+=batch_loss*batch_n
        num_examples+=batch_n
    return weighted_loss/max(num_examples,1)


@torch.no_grad()
def evaluate(model,loader,minimum,maximum,device,collect=False):
    model.eval()
    totals=Metrics(model.channels)
    pred_list=[]
    true_list=[]
    weighted_loss=0.0
    n_examples=0
    for x,y in loader:
        xx=to_device(x,device)
        yy=to_device(y,device)
        xn=normalize(xx,minimum,maximum)
        yn=normalize(yy,minimum[:,0],maximum[:,0])
        yp,tv=model(xn,raw_input=xx)
        loss=compute_loss(yn,yp,tv)
        raw_pred=denormalize(yp,minimum[:,0],maximum[:,0])
        totals.add(yy,raw_pred)
        weighted_loss+=float(loss)*len(x)
        n_examples+=len(x)
        if collect:
            pred_list.append(raw_pred.cpu().numpy())
            true_list.append(yy.cpu().numpy())
    return weighted_loss/max(n_examples,1),totals.result(),(pred_list,true_list)


def run(train_npy,test_npy,save_dir,micro_batch=2):
    if micro_batch<1 or micro_batch>cfg.BATCH:
        raise ValueError('micro_batch must be in [1, 32]')
    set_seed(cfg.SEED)
    out=Path(save_dir)
    out.mkdir(parents=True,exist_ok=True)
    train_full,test=load_milan(train_npy,test_npy,cfg.CHANNELS)
    train,val=split_train_val(train_full,cfg.P_IN)
    print(f'Milan one step: train={len(train)}, val={len(val)}, test={len(test)} timestamps; P={cfg.P_IN}')
    print(f'Input shape={train_full.shape[1:]}, channels={cfg.CHANNELS}, actual FFT bins={cfg.P_IN//2+1}')
    if len(test)!=168:
        print('WARNING: paper describes a 168-hour (7-day) Milan holdout; supplied test has',len(test),'observations')
    if train_full.shape[0]+test.shape[0]!=1464:
        print('WARNING: paper reports 1464 total hourly Milan timestamps; confirm source/preprocessing.')
    tr_ds=MilanWindowDataset(train,cfg.P_IN,cfg.CHANNELS)
    val_ds=MilanWindowDataset(val,cfg.P_IN,cfg.CHANNELS)
    te_ds=MilanWindowDataset(test,cfg.P_IN,cfg.CHANNELS)
    loader_kw={'num_workers':cfg.NUM_WORKERS,'pin_memory':cfg.DEVICE.startswith('cuda')}
    tr_dl=DataLoader(tr_ds,batch_size=cfg.BATCH,shuffle=True,**loader_kw)
    # Validation/test use micro-batches to reduce memory use; metrics cover all examples.
    val_dl=DataLoader(val_ds,batch_size=micro_batch,shuffle=False,**loader_kw)
    te_dl=DataLoader(te_ds,batch_size=micro_batch,shuffle=False,**loader_kw)
    minimum,maximum=train_only_min_max(train,cfg.CHANNELS)
    device=torch.device(cfg.DEVICE)
    minimum=minimum.to(device)
    maximum=maximum.to(device)
    model=SPECNet(cfg.H,cfg.W,len(cfg.CHANNELS),cfg.P_IN).to(device)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}; device={device}; effective batch={cfg.BATCH}, micro={micro_batch}')
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.LR,weight_decay=cfg.WD)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg.EPOCHS)
    best=float('inf'); best_epoch=0; stale=0
    history=[]
    ckpt=out/'specnet_milan_best.pt'
    tic=time.perf_counter()
    for epoch in range(1,cfg.EPOCHS+1):
        e0=time.perf_counter()
        train_loss=optimize_epoch(model,tr_dl,minimum,maximum,opt,device,micro_batch)
        val_loss,metric,_=evaluate(model,val_dl,minimum,maximum,device)
        scheduler.step()
        row={'epoch':epoch,'train_loss':train_loss,'val_loss':val_loss,
             'val_rmse':metric['rmse'],'val_mae':metric['mae'],
             'lr':opt.param_groups[0]['lr'],'epoch_sec':time.perf_counter()-e0}
        history.append(row)
        if metric['rmse']<best-1e-10:
            best=metric['rmse'];best_epoch=epoch;stale=0
            torch.save({'model':model.state_dict(),'config':asdict(cfg),
                        'minimum':minimum.cpu(),'maximum':maximum.cpu(),
                        'epoch':epoch,'best_val_rmse':best},ckpt)
        else:
            stale+=1
        print(f"epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f} "
              f"valRMSE={metric['rmse']:.4f} best={best:.4f} patience={stale}/{cfg.EARLY_STOP_PATIENCE}",flush=True)
        if stale>=cfg.EARLY_STOP_PATIENCE:
            print('Early stopping triggered')
            break
    with (out/'history.csv').open('w',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=list(history[0]))
        writer.writeheader();writer.writerows(history)
    # No test-set peeking before checkpoint selection.
    saved=torch.load(ckpt,map_location=device,weights_only=True)
    model.load_state_dict(saved['model'])
    test_loss,test_metrics,predictions=evaluate(model,te_dl,minimum,maximum,device,collect=True)
    np.save(out/'test_predictions.npy',np.concatenate(predictions[0],axis=0))
    np.save(out/'test_targets.npy',np.concatenate(predictions[1],axis=0))
    summary={'best_epoch':best_epoch,'best_val_rmse':best,'epochs_ran':len(history),
             'train_time_sec':time.perf_counter()-tic,'test_loss':test_loss,
             'test_metrics':test_metrics,'config':asdict(cfg),'micro_batch':micro_batch,
             'split':{'train_observations':len(train),'val_observations':len(val),
                      'test_observations':len(test),
                      'train_windows':len(tr_ds),'val_windows':len(val_ds),'test_windows':len(te_ds)}}
    (out/'results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({'best_epoch':best_epoch,'test_metrics':test_metrics},indent=2))
    return summary


def main():
    parser=argparse.ArgumentParser(description='Paper-aligned SPEC-Net Milan one-step training')
    parser.add_argument('--train',required=True,help='Preprocessed Milan data before last 7 days, .npy')
    parser.add_argument('--test',required=True,help='Held-out last 7 days, .npy')
    parser.add_argument('--save-dir',default=cfg.SAVE_DIR)
    parser.add_argument('--micro-batch',type=int,default=2,
                        help='Gradient accumulation micro-batch; effective batch remains 32')
    args=parser.parse_args()
    run(args.train,args.test,args.save_dir,args.micro_batch)


if __name__=='__main__':
    main()
