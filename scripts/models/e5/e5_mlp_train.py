import argparse,json,hashlib,os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from joblib import dump
from importlib.metadata import version
from mlp_common import preprocessing,choose_epochs,make_training,epoch_train,score

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run(args):
    root=Path(args.features).resolve();embedding_dir=Path(json.loads((root/'latest_embeddings.json').read_text())['directory'])
    em=json.loads((embedding_dir/'manifest.json').read_text());assert em['status']=='complete'
    frames={s:pd.read_csv(root/f'{s}_features.csv') for s in ['train','test']}
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
    assert len(frames['train'])==7322 and len(frames['test'])==1618
    assert not {' '.join(t.lower().split()) for t in frames['train'].text}&{' '.join(t.lower().split()) for t in frames['test'].text}
    names=features+[f'e5_{i:04d}' for i in range(1024)]
    if args.mode=='external':
        x=pd.DataFrame(np.concatenate([matrices['train'],matrices['test']]),columns=names)
        y=np.concatenate([frames['train'].target,frames['test'].target]).astype(int)
        n=len(frames['train']);splits=[(np.arange(n),np.arange(n,len(y)))];offset=n
    else:
        x=pd.DataFrame(matrices['test'],columns=names);y=frames['test'].target.to_numpy(dtype=int);offset=0
        splits=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(x,y))
    assert np.isin(y,np.arange(1,7)).all()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(min(4,os.cpu_count() or 1))
    config={'mode':args.mode,'seed':42,'features':159,'embeddings':1024,'input_dim':1183,
        'preprocessing':'StandardScaler on concatenated 1183 columns, fit on training rows only; no feature selection',
        'embedding_manifest_sha256':sha(embedding_dir/'manifest.json'),'epochs':args.max_epochs,'patience':args.patience,'batch_size':args.batch_size,
        'early_stopping':'20% stratified inner validation, cross-entropy; fresh refit on full training partition for best_epoch',
        'architecture':[1183,256,128,6],'activation':'ReLU','dropout':.5,'output':'Softmax','optimizer':{'name':'Adam','lr':.001,'betas':[.9,.999],'eps':1e-8},
        'versions':{p:version(p) for p in ['numpy','pandas','scikit-learn','torch']},'device':str(device),
        'script_sha256':sha(__file__),'common_sha256':sha(Path(__file__).with_name('mlp_common.py'))}
    key=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:20]
    out=root/'mlp_experiments'/args.mode/key;out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'input_columns.json').write_text(json.dumps(names,ensure_ascii=False),encoding='utf-8')
    pred=np.full(1618,-1);prob=np.full((1618,6),np.nan);assigned=np.full(1618,-1);fold_metrics=[];indices=[]
    for fold,(train,test) in enumerate(splits,1):
        assert not set(train)&set(test)
        best,history,inner_train,inner_val=choose_epochs(x,y,train,42+fold,device,args.max_epochs,args.patience,args.batch_size)
        assert not set(test)&set(inner_train) and not set(test)&set(inner_val)
        scaler=preprocessing().fit(x.iloc[train],y[train]);a=scaler.transform(x.iloc[train]);b=scaler.transform(x.iloc[test])
        model,optim,xt,yt=make_training(a,y[train],device,1042+fold);refit=[]
        for epoch in range(1,best+1):refit.append({'epoch':epoch,'train_loss':epoch_train(model,optim,xt,yt,args.batch_size)})
        model.eval()
        with torch.inference_mode():p=torch.softmax(model(torch.as_tensor(b,dtype=torch.float32,device=device)),dim=1).cpu().numpy()
        rows=test-offset;assert (assigned[rows]==-1).all()
        pred[rows]=p.argmax(1)+1;prob[rows]=p;assigned[rows]=fold
        record={'fold':fold,'n_train':len(train),'n_test':len(test),'best_epoch':best,**score(y[test],pred[rows])};fold_metrics.append(record)
        indices.append({'fold':fold,'train_indices':train.tolist(),'test_indices':test.tolist(),'inner_train':inner_train.tolist(),'inner_val':inner_val.tolist()})
        folder=out/f'fold_{fold}';folder.mkdir(exist_ok=True)
        dump(scaler,folder/'scaler.joblib')
        torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'input_dim':1183,'classes':list(range(1,7)),'best_epoch':best},folder/'mlp.pt')
        pd.DataFrame(history).to_csv(folder/'early_stopping.csv',index=False);pd.DataFrame(refit).to_csv(folder/'refit.csv',index=False)
        print(args.mode,record,flush=True)
    assert (assigned>0).all() and np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-5)
    truth=frames['test'].target.to_numpy(dtype=int)
    summary={'experiment':args.mode,'n_test':1618,**score(truth,pred)}
    table=pd.DataFrame(fold_metrics)
    if args.mode=='cv':
        for metric in ['Macro F1','Relaxed Macro F1','MAE']:
            summary[metric+' mean']=float(table[metric].mean());summary[metric+' std']=float(table[metric].std(ddof=1))
    table.to_csv(out/'fold_metrics.csv',index=False)
    results=frames['test'][['row_id','text','target']].copy();results['fold']=assigned;results['prediction']=pred
    results['prediction_relaxed']=np.where(np.abs(pred-truth)<=1,truth,pred)
    for i in range(6):results[f'probability_{i+1}']=prob[:,i]
    results.to_csv(out/'predictions_1618.csv',index=False,encoding='utf-8-sig')
    (out/'split_indices.json').write_text(json.dumps({'index_space':'concatenated train then test' if args.mode=='external' else 'expert dataset','splits':indices}),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame([summary]).to_csv(out/'summary.csv',index=False,encoding='utf-8-sig')
    (root/f'latest_{args.mode}.json').write_text(json.dumps({'directory':str(out)}),encoding='utf-8')
    print('ГОТОВО',summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--mode',choices=['external','cv'],required=True)
    p.add_argument('--max-epochs',type=int,default=100);p.add_argument('--patience',type=int,default=20);p.add_argument('--batch-size',type=int,default=32)
    a=p.parse_args();assert a.max_epochs>0 and a.patience>0 and a.batch_size>0;run(a)
