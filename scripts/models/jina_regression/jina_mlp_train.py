"""Jina + 159 Textometr features: MLP regression, stratified CV5.
Uses existing test_features.csv and cached classification-adapter embeddings.
No Jina fine-tuning; no feature selection. Independent of mlp_common.py.
Run: python jina_mlp_train.py --features /path/to/feature_directory --mode cv
Re-running trains all folds again; classification results are not overwritten.
"""
import argparse,json,hashlib,os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from joblib import dump
from importlib.metadata import version
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
            nn.Linear(256,128),nn.ReLU(),nn.Dropout(.5),nn.Linear(128,1))
    def forward(self,x):return self.net(x).squeeze(-1) # scalar level, linear output

def make_training(x,y,device,seed):
    seed_everything(seed)
    model=MLP(x.shape[1]).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,betas=(.9,.999),eps=1e-8)
    xt=torch.as_tensor(x,dtype=torch.float32,device=device)
    yt=torch.as_tensor(y,dtype=torch.float32,device=device)
    return model,optimizer,xt,yt

def epoch_train(model,optimizer,x,y,batch_size):
    model.train();total=0.
    order=torch.randperm(len(y),device=x.device)
    for start in range(0,len(y),batch_size):
        idx=order[start:start+batch_size]
        optimizer.zero_grad(set_to_none=True)
        loss=nn.functional.mse_loss(model(x[idx]),y[idx])
        if not torch.isfinite(loss): raise RuntimeError("Non-finite training MSE")
        loss.backward();optimizer.step();total+=loss.item()*len(idx)
    return total/len(y)

def choose_epochs(x,y,outer_train,seed,device,max_epochs,patience,batch_size):
    inner_train,inner_val=train_test_split(outer_train,test_size=.2,stratify=y[outer_train],random_state=seed)
    transform=preprocessing().fit(x.iloc[inner_train],y[inner_train])
    a=transform.transform(x.iloc[inner_train]);b=transform.transform(x.iloc[inner_val])
    model,optim,xt,yt=make_training(a,y[inner_train],device,seed)
    xv=torch.as_tensor(b,dtype=torch.float32,device=device)
    yv=torch.as_tensor(y[inner_val],dtype=torch.float32,device=device)
    best=float('inf');best_epoch=0;wait=0;history=[]
    for epoch in range(1,max_epochs+1):
        train_loss=epoch_train(model,optim,xt,yt,batch_size)
        model.eval()
        with torch.no_grad():val_loss=nn.functional.mse_loss(model(xv),yv).item()
        if not np.isfinite(val_loss):raise RuntimeError('Нечисловой validation loss')
        history.append({'epoch':epoch,'train_loss':train_loss,'validation_loss':val_loss})
        if val_loss<best:
            best=val_loss;best_epoch=epoch;wait=0
        else:wait+=1
        if wait>=patience:break
    return best_epoch,history,inner_train,inner_val


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run(args):
    root=Path(args.features).resolve();embedding_dir=Path(json.loads((root/'latest_embeddings.json').read_text())['directory'])
    em=json.loads((embedding_dir/'manifest.json').read_text());assert em['status']=='complete'
    assert em['model']=='jinaai/jina-embeddings-v3' and em['task']=='classification' and em['dimensions']==1024
    frames={s:pd.read_csv(root/f'{s}_features.csv') for s in ['test']}
    features=(root/'input_features.txt').read_text(encoding='utf-8').splitlines()
    assert len(features)==159
    matrices={}
    for split,f in frames.items():
        assert sha(root/f'{split}_features.csv')==em['input_files'][split]
        assert sha(embedding_dir/f'{split}_embeddings.npy')==em['embedding_files'][split]
        e=np.load(embedding_dir/f'{split}_embeddings.npy',allow_pickle=False)
        assert f.row_id.tolist()==list(range(len(f))) and e.shape==(len(f),1024)
        matrices[split]=np.concatenate([f[features].to_numpy(dtype=np.float32),e],axis=1)
        assert np.isfinite(matrices[split]).all() and matrices[split].shape[1]==1183
    assert len(frames['test'])==1618
    names=features+[f'jina_{i:04d}' for i in range(1024)]
    x=pd.DataFrame(matrices['test'],columns=names)
    original_target=frames['test'].target.to_numpy()
    y=original_target.astype(int);offset=0
    assert np.array_equal(original_target,y), 'Expected integer median_rating_round'
    splits=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(x,y))
    assert np.isin(y,np.arange(1,7)).all()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(min(4,os.cpu_count() or 1))
    config={'mode':args.mode,'seed':42,'features':159,'embeddings':1024,'input_dim':1183,
        'preprocessing':'StandardScaler on concatenated 1183 columns, fit on training rows only; no feature selection',
        'embedding_manifest_sha256':sha(embedding_dir/'manifest.json'),'epochs':args.max_epochs,'patience':args.patience,'batch_size':args.batch_size,
        'early_stopping':'20% stratified inner validation, MSE; fresh refit on full training partition for best_epoch',
        'architecture':[1183,256,128,1],'activation':'ReLU','dropout':.5,'output':'linear scalar','loss':'MSE','target':'median_rating_round','rounding':'clip(floor(raw+0.5),1,6)','optimizer':{'name':'Adam','lr':.001,'betas':[.9,.999],'eps':1e-8},
        'versions':{p:version(p) for p in ['numpy','pandas','scikit-learn','torch']},'device':str(device),
        'script_sha256':sha(__file__)}
    key=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:20]
    out=root/'jina_mlp_regression'/args.mode/key;out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'input_columns.json').write_text(json.dumps(names,ensure_ascii=False),encoding='utf-8')
    pred=np.full(1618,-1);raw=np.full(1618,np.nan);assigned=np.full(1618,-1);fold_metrics=[];indices=[]
    for fold,(train,test) in enumerate(splits,1):
        assert not set(train)&set(test)
        best,history,inner_train,inner_val=choose_epochs(x,y,train,42+fold,device,args.max_epochs,args.patience,args.batch_size)
        assert not set(test)&set(inner_train) and not set(test)&set(inner_val)
        scaler=preprocessing().fit(x.iloc[train],y[train]);a=scaler.transform(x.iloc[train]);b=scaler.transform(x.iloc[test])
        model,optim,xt,yt=make_training(a,y[train],device,1042+fold);refit=[]
        for epoch in range(1,best+1):refit.append({'epoch':epoch,'train_loss':epoch_train(model,optim,xt,yt,args.batch_size)})
        model.eval()
        with torch.inference_mode():p=model(torch.as_tensor(b,dtype=torch.float32,device=device)).cpu().numpy()
        rows=test-offset;assert (assigned[rows]==-1).all()
        assert np.isfinite(p).all()
        raw[rows]=p;pred[rows]=np.clip(np.floor(p+.5),1,6).astype(int);assigned[rows]=fold
        record={'fold':fold,'n_train':len(train),'n_test':len(test),'best_epoch':best,**score(y[test],pred[rows])};fold_metrics.append(record)
        indices.append({'fold':fold,'train_indices':train.tolist(),'test_indices':test.tolist(),'inner_train':inner_train.tolist(),'inner_val':inner_val.tolist()})
        folder=out/f'fold_{fold}';folder.mkdir(exist_ok=True)
        dump(scaler,folder/'scaler.joblib')
        torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'input_dim':1183,'output_dim':1,'task':'regression','best_epoch':best},folder/'mlp.pt')
        pd.DataFrame(history).to_csv(folder/'early_stopping.csv',index=False);pd.DataFrame(refit).to_csv(folder/'refit.csv',index=False)
        pd.DataFrame(fold_metrics).to_csv(out/'fold_metrics.csv',index=False)
        pd.DataFrame({'row_id':rows,'target':y[test],'raw_prediction':p,'prediction':pred[rows]}).to_csv(folder/'predictions.csv',index=False)
        print(args.mode,record,flush=True)
    assert (assigned>0).all() and np.isfinite(raw).all()
    truth=frames['test'].target.to_numpy(dtype=int)
    summary={'experiment':args.mode,'n_test':1618,**{'OOF '+k:v for k,v in score(truth,pred).items()}}
    table=pd.DataFrame(fold_metrics)
    if args.mode=='cv':
        for metric in ['Macro F1','Relaxed Macro F1','MAE']:
            summary[metric+' mean']=float(table[metric].mean());summary[metric+' std']=float(table[metric].std(ddof=1))
    table.to_csv(out/'fold_metrics.csv',index=False)
    results=frames['test'][['row_id','text','target']].copy();results['fold']=assigned;results['prediction']=pred
    results['prediction_relaxed']=np.where(np.abs(pred-truth)<=1,truth,pred)
    results['raw_prediction']=raw
    results.to_csv(out/'predictions_1618.csv',index=False,encoding='utf-8-sig')
    (out/'split_indices.json').write_text(json.dumps({'index_space':'concatenated train then test' if args.mode=='external' else 'expert dataset','splits':indices}),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame([summary]).to_csv(out/'summary.csv',index=False,encoding='utf-8-sig')
    (root/f'latest_{args.mode}_regression.json').write_text(json.dumps({'directory':str(out)}),encoding='utf-8')
    print('ГОТОВО',summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--mode',choices=['cv'],default='cv')
    p.add_argument('--max-epochs',type=int,default=100);p.add_argument('--patience',type=int,default=20);p.add_argument('--batch-size',type=int,default=32)
    a=p.parse_args();assert a.max_epochs>0 and a.patience>0 and a.batch_size>0;run(a)
