import argparse, json, hashlib, random, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, mean_absolute_error, confusion_matrix
from joblib import dump
from importlib.metadata import version

def score(y,p):
    relaxed=np.where(np.abs(p-y)<=1,y,p)
    strict=f1_score(y,p,labels=np.arange(1,7),average='macro',zero_division=0)
    tolerant=f1_score(y,relaxed,labels=np.arange(1,7),average='macro',zero_division=0)
    assert tolerant+1e-12>=strict
    return {'Macro F1':float(strict),'Relaxed Macro F1':float(tolerant),'MAE':float(mean_absolute_error(y,p))}

def preprocessing():
    return StandardScaler()

def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

class MLP(nn.Module):
    def __init__(self,n_features):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(n_features,256),nn.ReLU(),nn.Dropout(.5),
            nn.Linear(256,128),nn.ReLU(),nn.Dropout(.5),nn.Linear(128,6))
    def forward(self,x):return self.net(x) # logits; CrossEntropyLoss applies log-softmax

def make_training(x,y,device,seed):
    seed_everything(seed)
    model=MLP(x.shape[1]).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,betas=(.9,.999),eps=1e-8)
    xt=torch.as_tensor(x,dtype=torch.float32,device=device)
    yt=torch.as_tensor(y-1,dtype=torch.long,device=device)
    return model,optimizer,xt,yt

def epoch_train(model,optimizer,x,y,batch_size):
    model.train();total=0.
    order=torch.randperm(len(y),device=x.device)
    for start in range(0,len(y),batch_size):
        idx=order[start:start+batch_size]
        optimizer.zero_grad(set_to_none=True)
        loss=nn.functional.cross_entropy(model(x[idx]),y[idx])
        loss.backward();optimizer.step();total+=loss.item()*len(idx)
    return total/len(y)

def choose_epochs(x,y,outer_train,seed,device,max_epochs,patience,batch_size):
    inner_train,inner_val=train_test_split(outer_train,test_size=.2,stratify=y[outer_train],random_state=seed)
    transform=preprocessing().fit(x.iloc[inner_train],y[inner_train])
    a=transform.transform(x.iloc[inner_train]);b=transform.transform(x.iloc[inner_val])
    model,optim,xt,yt=make_training(a,y[inner_train],device,seed)
    xv=torch.as_tensor(b,dtype=torch.float32,device=device)
    yv=torch.as_tensor(y[inner_val]-1,dtype=torch.long,device=device)
    best=float('inf');best_epoch=0;wait=0;history=[]
    for epoch in range(1,max_epochs+1):
        train_loss=epoch_train(model,optim,xt,yt,batch_size)
        model.eval()
        with torch.no_grad():val_loss=nn.functional.cross_entropy(model(xv),yv).item()
        if not np.isfinite(val_loss):raise RuntimeError('Нечисловой validation loss')
        history.append({'epoch':epoch,'train_loss':train_loss,'validation_loss':val_loss})
        if val_loss<best:
            best=val_loss;best_epoch=epoch;wait=0
        else:wait+=1
        if wait>=patience:break
    return best_epoch,history,inner_train,inner_val
