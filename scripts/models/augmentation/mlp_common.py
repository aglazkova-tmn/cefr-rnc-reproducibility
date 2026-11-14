import random
import numpy as np
import torch
from torch import nn
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
        assert torch.isfinite(loss), 'Nonfinite cross entropy'
        loss.backward();optimizer.step();total+=loss.item()*len(idx)
    return total/len(y)
