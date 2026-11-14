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
from sklearn.metrics import f1_score, mean_absolute_error, confusion_matrix, precision_recall_fscore_support
from joblib import dump
from importlib.metadata import version

def score(y,p):
    relaxed=np.where(np.abs(p-y)<=1,y,p)
    strict=f1_score(y,p,labels=np.arange(1,7),average='macro',zero_division=0)
    tolerant=f1_score(y,relaxed,labels=np.arange(1,7),average='macro',zero_division=0)
    assert tolerant+1e-12>=strict
    return {'Macro F1':float(strict),'Relaxed Macro F1':float(tolerant),'MAE':float(mean_absolute_error(y,p))}

def class_weights(y):
    y=np.asarray(y)
    assert np.isin(y,np.arange(1,7)).all()
    counts=np.bincount(y.astype(int),minlength=7)[1:]
    assert (counts>0).all(), 'Every level must occur in the training partition'
    weights=len(y)/(6*counts)
    assert np.isclose(weights[y.astype(int)-1].mean(),1.)
    return weights

def per_class(y,p):
    precision,recall,f1,support=precision_recall_fscore_support(y,p,labels=np.arange(1,7),zero_division=0)
    relaxed=np.where(np.abs(p-y)<=1,y,p)
    relaxed_f1=precision_recall_fscore_support(y,relaxed,labels=np.arange(1,7),zero_division=0)[2]
    return pd.DataFrame({'level':['A1','A2','B1','B2','C1','C2'],
        'precision':precision,'recall':recall,'F1':f1,'Relaxed F1':relaxed_f1,
        'MAE_by_true_level':[float(np.mean(np.abs(p[y==c]-y[y==c]))) if (y==c).any() else np.nan for c in range(1,7)],
        'support':support,'predicted_count':[int((p==c).sum()) for c in range(1,7)]})

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

def epoch_train(model,optimizer,x,y,batch_size,weights):
    model.train();total=0.;weight_total=0.
    order=torch.randperm(len(y),device=x.device)
    for start in range(0,len(y),batch_size):
        idx=order[start:start+batch_size]
        optimizer.zero_grad(set_to_none=True)
        loss=nn.functional.cross_entropy(model(x[idx]),y[idx],weight=weights)
        if not torch.isfinite(loss):raise RuntimeError("Non-finite training loss")
        loss.backward();optimizer.step()
        batch_weight=weights[y[idx]].sum().item()
        total+=loss.item()*batch_weight;weight_total+=batch_weight
    return total/weight_total

def choose_epochs(x,y,outer_train,seed,device,max_epochs,patience,batch_size):
    inner_train,inner_val=train_test_split(outer_train,test_size=.2,stratify=y[outer_train],random_state=seed)
    transform=preprocessing().fit(x.iloc[inner_train],y[inner_train])
    a=transform.transform(x.iloc[inner_train]);b=transform.transform(x.iloc[inner_val])
    model,optim,xt,yt=make_training(a,y[inner_train],device,seed)
    xv=torch.as_tensor(b,dtype=torch.float32,device=device)
    yv=torch.as_tensor(y[inner_val]-1,dtype=torch.long,device=device)
    weights=torch.as_tensor(class_weights(y[inner_train]),dtype=torch.float32,device=device)
    best=float('inf');best_epoch=0;wait=0;history=[]
    for epoch in range(1,max_epochs+1):
        train_loss=epoch_train(model,optim,xt,yt,batch_size,weights)
        model.eval()
        with torch.no_grad():val_loss=nn.functional.cross_entropy(model(xv),yv,weight=weights).item()
        if not np.isfinite(val_loss):raise RuntimeError('Нечисловой validation loss')
        history.append({'epoch':epoch,'train_loss':train_loss,'validation_loss':val_loss})
        if val_loss<best:
            best=val_loss;best_epoch=epoch;wait=0
        else:wait+=1
        if wait>=patience:break
    return best_epoch,history,inner_train,inner_val

def run(args):
    feature_dir=Path(args.features).resolve()
    path=feature_dir/'features_1618.csv'
    manifest=json.loads((feature_dir/'features_manifest.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['feature_table_sha256']
    frame=pd.read_csv(path);features=manifest['features']
    assert len(frame)==1618 and len(features)==159 and frame.row_id.tolist()==list(range(1618))
    x=frame[features];y=frame.target.to_numpy(dtype=int)
    assert np.isfinite(x.to_numpy()).all() and np.isin(y,np.arange(1,7)).all()
    assert np.array_equal(frame.target.to_numpy(),y)
    config={'folds':5,'seed':42,'target':'median_rating_round','features_sha256':manifest['feature_table_sha256'],
        'loss':'CrossEntropyLoss(weight=N_train/(6*n_train_class), reduction=mean)',
        'validation_weights':'from inner train only','input_features':features,
        'max_epochs':args.max_epochs,'patience':args.patience,'batch_size':args.batch_size,'validation_fraction':.2,
        'MLP':'all 159 features, StandardScaler only; 256/ReLU/Dropout(.5),128/ReLU/Dropout(.5),6/Softmax',
        'Adam':{'lr':1e-3,'betas':[.9,.999],'eps':1e-8},'early_stopping':'inner validation weighted cross-entropy; refit full outer train for best_epoch',
        'std':'sample standard deviation, ddof=1','runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'versions':{p:version(p) for p in ['numpy','pandas','scikit-learn','torch']}}
    config['device']=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    run_id=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
    output_root=Path(args.output).resolve() if args.output else feature_dir/'mlp159_weighted_classifier_results'
    out=output_root/run_id;out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('MLP device:',device,flush=True)
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    folds=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(x,y))
    assignments=np.full(len(y),-1);oof={m:np.full(len(y),-1) for m in ['MLP']}
    raw=np.full(len(y),np.nan);probs=np.full((len(y),6),np.nan);records=[]
    split_records=[];class_records=[]
    for fold,(train,test) in enumerate(folds,1):
        assert not set(train)&set(test)
        fold_dir=out/f'fold_{fold}';fold_dir.mkdir(exist_ok=True)
        # Fit scaling on the outer training fold; keep all features.
        transform=preprocessing().fit(x.iloc[train],y[train])
        a=transform.transform(x.iloc[train]);b=transform.transform(x.iloc[test])
        selected=list(features)
        best,history,inner_train,inner_val=choose_epochs(x,y,train,42+fold,device,args.max_epochs,args.patience,args.batch_size)
        assert not set(test)&set(inner_train) and not set(test)&set(inner_val)
        model,optim,xt,yt=make_training(a,y[train],device,1042+fold)
        weights=torch.as_tensor(class_weights(y[train]),dtype=torch.float32,device=device)
        weight_info={'classes':[1,2,3,4,5,6],
            'inner_train_counts':np.bincount(y[inner_train],minlength=7)[1:].tolist(),
            'inner_train_weights':class_weights(y[inner_train]).tolist(),
            'outer_train_counts':np.bincount(y[train],minlength=7)[1:].tolist(),
            'outer_train_weights':class_weights(y[train]).tolist()}
        (fold_dir/'class_weights.json').write_text(json.dumps(weight_info,indent=2),encoding='utf8')
        refit_history=[]
        for epoch in range(1,best+1):
            refit_history.append({'epoch':epoch,'train_loss':epoch_train(model,optim,xt,yt,args.batch_size,weights)})
        model.eval()
        with torch.no_grad():
            probabilities=torch.softmax(model(torch.as_tensor(b,dtype=torch.float32,device=device)),dim=1).cpu().numpy()
        probs[test]=probabilities;oof['MLP'][test]=probabilities.argmax(axis=1)+1
        assert np.allclose(probabilities.sum(axis=1),1,atol=1e-5)
        assignments[test]=fold
        class_records.append(per_class(y[test],oof['MLP'][test]).assign(fold=fold))
        for name in oof:
            records.append({'model':name,'fold':fold,'n_train':len(train),'n_test':len(test),'selected_features':len(selected),
                'best_epoch':best if name=='MLP' else None,**score(y[test],oof[name][test])})
        split_records.append({'fold':fold,'train_rows':train.tolist(),'test_rows':test.tolist(),
            'mlp_inner_train_rows':inner_train.tolist(),'mlp_inner_validation_rows':inner_val.tolist()})
        dump(transform,fold_dir/'preprocessing.joblib')
        torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'n_features':len(selected),
            'classes':[1,2,3,4,5,6],'best_epoch':best},fold_dir/'mlp.pt')
        (fold_dir/'selected_features.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding='utf-8')
        pd.DataFrame(history).to_csv(fold_dir/'mlp_early_stopping.csv',index=False)
        pd.DataFrame(refit_history).to_csv(fold_dir/'mlp_refit.csv',index=False)
        pd.DataFrame(records).to_csv(out/'fold_metrics.csv',index=False)
        print(f'Фолд {fold}/5; использовано {len(selected)} признаков; MLP best_epoch={best}',flush=True)
        print(pd.DataFrame(records[-1:]).to_string(index=False),flush=True)
    assert (assignments>0).all() and all((p>=1).all() for p in oof.values())
    assert np.isfinite(probs).all()
    metrics=pd.DataFrame(records);summary=[]
    for name in oof:
        row={'model':name,**{f'OOF {k}':v for k,v in score(y,oof[name]).items()}}
        for metric in ['Macro F1','Relaxed Macro F1','MAE']:
            values=metrics.loc[metrics.model==name,metric]
            row[metric+' mean']=float(values.mean());row[metric+' std']=float(values.std(ddof=1))
        summary.append(row)
        pd.DataFrame(confusion_matrix(y,oof[name],labels=np.arange(1,7)),index=range(1,7),columns=range(1,7)).to_csv(out/f'{name}_confusion_matrix.csv')
    pd.DataFrame(summary).to_csv(out/'metrics_summary.csv',index=False,encoding='utf-8-sig')
    result=frame[['row_id','text','target']].copy();result['fold']=assignments
    for name in oof:
        result[name+'_prediction']=oof[name]
        result[name+'_relaxed']=np.where(np.abs(oof[name]-y)<=1,y,oof[name])
    for c in range(6):result[f'MLP_probability_{c+1}']=probs[:,c]
    result.to_csv(out/'oof_predictions.csv',index=False,encoding='utf-8-sig')
    (out/'fold_indices.json').write_text(json.dumps(split_records),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    class_table=per_class(y,oof['MLP'])
    class_table.to_csv(out/'per_class_oof.csv',index=False,encoding='utf-8-sig')
    pd.concat(class_records,ignore_index=True).to_csv(out/'per_class_by_fold.csv',index=False,encoding='utf-8-sig')
    print('OOF quality by class:\n'+class_table.to_string(index=False),flush=True)
    (output_root/'latest_weighted_classifier.json').write_text(json.dumps({'directory':str(out)}),encoding='utf-8')
    print('\nГОТОВО\n'+pd.DataFrame(summary).to_string(index=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--features',required=True)
    parser.add_argument('--output',default='')
    parser.add_argument('--max-epochs',type=int,default=100)
    parser.add_argument('--patience',type=int,default=20)
    parser.add_argument('--batch-size',type=int,default=32)
    args=parser.parse_args()
    assert args.max_epochs>0 and args.patience>0 and args.batch_size>0
    run(args)
