import argparse, gc, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from huggingface_hub import HfApi
from sklearn.model_selection import StratifiedKFold
from mlp_all import score, seed_everything
from importlib.metadata import version

def run(args):
    root=Path(args.features)
    frame=pd.read_csv(root/'features_1618.csv')
    y=frame.target.to_numpy(dtype=int)
    assert len(y)==1618 and np.array_equal(y,frame.target) and np.isin(y,range(1,7)).all()
    assert frame.row_id.tolist()==list(range(1618))
    device=torch.device('cuda')
    assert torch.cuda.is_available(), 'Enable GPU in Colab'
    name='ai-forever/ruBert-base'
    revision=args.revision or HfApi().model_info(name).sha
    config=dict(model=name,revision=revision,epochs=5,lr=2e-5,batch_size=args.batch_size,
        max_length=256,weight_decay=.01,seed=42,folds=5,loss='MSE',output_units=1,
        optimizer='AdamW; betas=(0.9,0.999), eps=1e-8; no decay on bias/LayerNorm',
        scheduler='linear, warmup=0',gradient_clip=1.,mixed_precision=False,
        feature_hash=hashlib.sha256((root/'features_1618.csv').read_bytes()).hexdigest(),
        runner_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        versions={p:version(p) for p in ['torch','transformers','numpy','scikit-learn']})
    out=root/'rubert_regression'/hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
    out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf8')
    tokenizer=AutoTokenizer.from_pretrained(name,revision=revision)
    enc=tokenizer(frame.text.tolist(),padding='max_length',truncation=True,max_length=256,return_tensors='pt')
    keys=list(enc.keys()); raw=np.full(len(y),np.nan); assignments=np.zeros(len(y),int); records=[]; splits=[]
    for fold,(train,test) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(frame,y),1):
        fd=out/f'fold_{fold}';fd.mkdir(exist_ok=True)
        splits.append(dict(fold=fold,train_rows=train.tolist(),test_rows=test.tolist()))
        cache=fd/'predictions.csv'
        if cache.exists():
            saved=pd.read_csv(cache)
            assert saved.row_id.tolist()==test.tolist() and np.isfinite(saved.raw).all()
            raw[test]=saved.raw.to_numpy()
            print(f'Fold {fold}: saved predictions loaded',flush=True)
        else:
            seed_everything(42)
            model=AutoModelForSequenceClassification.from_pretrained(name,revision=revision,
                num_labels=1,problem_type='regression').to(device)
            no_decay=lambda n: n.endswith('bias') or 'LayerNorm.weight' in n
            groups=[{'params':[p for n,p in model.named_parameters() if not no_decay(n)],'weight_decay':.01},
                    {'params':[p for n,p in model.named_parameters() if no_decay(n)],'weight_decay':0.}]
            optimizer=torch.optim.AdamW(groups,lr=2e-5,betas=(.9,.999),eps=1e-8)
            dataset=TensorDataset(*[enc[k][train] for k in keys],torch.tensor(y[train],dtype=torch.float32))
            loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True)
            scheduler=get_linear_schedule_with_warmup(optimizer,0,5*len(loader))
            history=[]
            for epoch in range(1,6):
                model.train();total=0.
                for batch in loader:
                    inputs={k:v.to(device) for k,v in zip(keys,batch[:-1])}
                    optimizer.zero_grad(set_to_none=True)
                    loss=model(**inputs,labels=batch[-1].to(device)).loss
                    if not torch.isfinite(loss):raise RuntimeError('Non-finite MSE')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    optimizer.step();scheduler.step();total+=loss.item()*len(batch[-1])
                history.append(dict(epoch=epoch,train_MSE=total/len(train)))
                print(f'Fold {fold}/5 epoch {epoch}/5 MSE={total/len(train):.5f}',flush=True)
            model.eval()
            with torch.no_grad():
                for start in range(0,len(test),args.batch_size):
                    ids=test[start:start+args.batch_size]
                    raw[ids]=model(**{k:enc[k][ids].to(device) for k in keys}).logits.reshape(-1).cpu().numpy()
            assert np.isfinite(raw[test]).all()
            pd.DataFrame(history).to_csv(fd/'training_history.csv',index=False)
            temporary=fd/'predictions.tmp.csv'
            pd.DataFrame({'row_id':test,'raw':raw[test]}).to_csv(temporary,index=False)
            temporary.replace(cache)
            del model,optimizer,scheduler,loader,dataset,loss
            gc.collect();torch.cuda.empty_cache()
        assignments[test]=fold
        pred=np.clip(np.floor(raw[test]+.5),1,6).astype(int)
        records.append(dict(model='RuBERT regression',fold=fold,**score(y[test],pred)))
        pd.DataFrame(records).to_csv(out/'fold_metrics.csv',index=False)
    assert np.isfinite(raw).all() and (assignments>0).all()
    pred=np.clip(np.floor(raw+.5),1,6).astype(int)
    result=frame[['row_id','text','target']].copy()
    result['fold']=assignments;result['raw_prediction']=raw;result['prediction']=pred
    result['relaxed_prediction']=np.where(abs(pred-y)<=1,y,pred)
    result.to_csv(out/'oof_predictions.csv',index=False,encoding='utf-8-sig')
    row={'model':'RuBERT regression',**{'OOF '+k:v for k,v in score(y,pred).items()}}
    metrics=pd.DataFrame(records)
    for k in score(y,pred):row[k+' mean']=float(metrics[k].mean());row[k+' std']=float(metrics[k].std(ddof=1))
    (out/'summary.json').write_text(json.dumps([row],indent=2),encoding='utf8')
    (out/'fold_indices.json').write_text(json.dumps(splits),encoding='utf8')
    (root/'latest_regression.json').write_text(json.dumps({'directory':str(out)}),encoding='utf8')
    print(row,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--revision',default=None,help='Optional recorded model commit; omitted preserves original online resolution')
    args=p.parse_args();assert args.batch_size>0;run(args)
