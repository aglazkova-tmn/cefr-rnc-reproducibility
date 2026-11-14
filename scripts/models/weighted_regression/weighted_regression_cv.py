"""MLP159 / RuBERT regression: identical CV5, MSE vs level-weighted MSE."""
import argparse,gc,hashlib,json,random,os
from pathlib import Path
from importlib.metadata import version
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,mean_absolute_error,precision_recall_fscore_support,confusion_matrix
from joblib import dump

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,obj):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');tmp.replace(p)
def seed(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def weights(y,weighted):
    y=np.asarray(y);assert np.isin(y,range(1,7)).all()
    counts=np.bincount(y.astype(int),minlength=7)[1:]
    assert (counts>0).all(),'Missing level in training partition'
    w=len(y)/(6*counts) if weighted else np.ones(6)
    assert np.isclose(w[y.astype(int)-1].mean(),1)
    return w
def loss_fn(pred,y,w):
    # Mean weighted squared error over examples, NOT divided by batch weight sum.
    assert pred.shape==y.shape
    return (w[y.long()-1]*(pred-y).square()).mean()
def release():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
def score(y,p):
    relaxed=np.where(abs(y-p)<=1,y,p)
    f=lambda v:float(f1_score(y,v,labels=range(1,7),average='macro',zero_division=0))
    a,b=f(p),f(relaxed);assert b+1e-12>=a
    return {'Macro F1':a,'Relaxed Macro F1':b,'MAE':float(mean_absolute_error(y,p))}
class MLP(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(159,256),nn.ReLU(),nn.Dropout(.5),nn.Linear(256,128),nn.ReLU(),nn.Dropout(.5),nn.Linear(128,1))
    def forward(self,x):return self.net(x).squeeze(-1)
def init_mlp(x,y,device,s):
    seed(s);m=MLP().to(device);opt=torch.optim.Adam(m.parameters(),lr=.001,betas=(.9,.999),eps=1e-8)
    return m,opt,torch.as_tensor(x,dtype=torch.float32,device=device),torch.as_tensor(y,dtype=torch.float32,device=device)
def epoch_mlp(m,opt,x,y,w,batch):
    m.train();total=0.
    order=torch.randperm(len(y),device=x.device)
    for start in range(0,len(y),batch):
        ids=order[start:start+batch];opt.zero_grad(set_to_none=True)
        loss=loss_fn(m(x[ids]),y[ids],w)
        if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
        loss.backward();opt.step();total+=loss.item()*len(ids)
    return total/len(y)
def fit_mlp(x,y,tr,te,fold,fd,args,device):
    it,iv=train_test_split(tr,test_size=.2,stratify=y[tr],random_state=42+fold)
    assert not set(te)&(set(it)|set(iv))
    inner_w=weights(y[it],args.loss=='weighted');outer_w=weights(y[tr],args.loss=='weighted')
    js(fd/'weights.json',{'inner_train_counts':np.bincount(y[it],minlength=7)[1:].tolist(),
        'inner_train_weights':inner_w.tolist(),'outer_train_counts':np.bincount(y[tr],minlength=7)[1:].tolist(),'outer_train_weights':outer_w.tolist()})
    js(fd/'inner_split.json',{'inner_train':it.tolist(),'inner_validation':iv.tolist()})
    scaler=StandardScaler().fit(x[it]);a=scaler.transform(x[it]);b=scaler.transform(x[iv])
    m,opt,xt,yt=init_mlp(a,y[it],device,42+fold)
    xv=torch.as_tensor(b,dtype=torch.float32,device=device);yv=torch.as_tensor(y[iv],dtype=torch.float32,device=device)
    w=torch.as_tensor(inner_w,dtype=torch.float32,device=device)
    best=float('inf');best_epoch=0;wait=0;history=[]
    for epoch in range(1,args.max_epochs+1):
        train_loss=epoch_mlp(m,opt,xt,yt,w,args.batch_size);m.eval()
        with torch.inference_mode():val=loss_fn(m(xv),yv,w).item()
        if not np.isfinite(val):raise RuntimeError('Non-finite validation loss')
        history.append({'epoch':epoch,'train_objective':train_loss,'validation_objective':val})
        if val<best:best=val;best_epoch=epoch;wait=0
        else:wait+=1
        if wait>=args.patience:break
    pd.DataFrame(history).to_csv(fd/'early_stopping.csv',index=False)
    del m,opt,xt,yt,xv,yv;release()
    scaler=StandardScaler().fit(x[tr]);a=scaler.transform(x[tr]);b=scaler.transform(x[te])
    m,opt,xt,yt=init_mlp(a,y[tr],device,1042+fold)
    w=torch.as_tensor(outer_w,dtype=torch.float32,device=device);history=[]
    for epoch in range(1,best_epoch+1):history.append({'epoch':epoch,'train_objective':epoch_mlp(m,opt,xt,yt,w,args.batch_size)})
    m.eval()
    with torch.inference_mode():raw=m(torch.as_tensor(b,dtype=torch.float32,device=device)).cpu().numpy()
    pd.DataFrame(history).to_csv(fd/'refit.csv',index=False);dump(scaler,fd/'scaler.joblib')
    torch.save({'state_dict':{k:v.detach().cpu() for k,v in m.state_dict().items()},'input_dim':159,'best_epoch':best_epoch},fd/'mlp.pt')
    js(fd/'training_info.json',{'best_epoch':best_epoch})
    print('MLP fold',fold,'best epoch',best_epoch,flush=True)
    return raw
def fit_rubert(enc,y,tr,te,fd,args,device,revision):
    from transformers import AutoModelForSequenceClassification,get_linear_schedule_with_warmup
    from torch.utils.data import DataLoader,TensorDataset
    seed(42);cw=weights(y[tr],args.loss=='weighted')
    js(fd/'weights.json',{'outer_train_counts':np.bincount(y[tr],minlength=7)[1:].tolist(),'outer_train_weights':cw.tolist()})
    w=torch.as_tensor(cw,dtype=torch.float32,device=device)
    m=AutoModelForSequenceClassification.from_pretrained('ai-forever/ruBert-base',revision=revision,num_labels=1,problem_type='regression').to(device)
    no_decay=lambda n:n.endswith('bias') or 'LayerNorm.weight' in n
    groups=[{'params':[p for n,p in m.named_parameters() if not no_decay(n)],'weight_decay':.01},
            {'params':[p for n,p in m.named_parameters() if no_decay(n)],'weight_decay':0.}]
    opt=torch.optim.AdamW(groups,lr=2e-5,betas=(.9,.999),eps=1e-8)
    keys=list(enc);loader=DataLoader(TensorDataset(*[enc[k][tr] for k in keys],torch.tensor(y[tr],dtype=torch.float32)),batch_size=args.batch_size,shuffle=True)
    scheduler=get_linear_schedule_with_warmup(opt,0,args.bert_epochs*len(loader));history=[]
    for epoch in range(1,args.bert_epochs+1):
        m.train();total=0.
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            pred=m(**{k:v.to(device) for k,v in zip(keys,batch[:-1])}).logits.reshape(-1)
            truth=batch[-1].to(device);loss=loss_fn(pred,truth,w)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite RuBERT loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();scheduler.step()
            total+=loss.item()*len(truth)
        history.append({'epoch':epoch,'train_objective':total/len(tr)})
        print(fd.name,'RuBERT epoch',epoch,'loss',total/len(tr),flush=True)
    pd.DataFrame(history).to_csv(fd/'training_history.csv',index=False)
    m.eval();raw=[]
    with torch.inference_mode():
        for start in range(0,len(te),args.batch_size):
            ids=te[start:start+args.batch_size]
            raw.extend(m(**{k:enc[k][ids].to(device) for k in keys}).logits.reshape(-1).cpu().numpy().tolist())
    if args.save_bert_weights:m.save_pretrained(fd/'rubert')
    return np.asarray(raw)

def run(args):
    root=Path(args.features).resolve();path=root/'features_1618.csv'
    manifest=json.loads((root/'features_manifest.json').read_text(encoding='utf8'))
    assert sha(path)==manifest['feature_table_sha256']
    frame=pd.read_csv(path);features=manifest['features'];x=frame[features].to_numpy(dtype=np.float32);y=frame.target.to_numpy(dtype=int)
    assert len(frame)==1618 and len(features)==159 and frame.row_id.tolist()==list(range(1618))
    assert np.array_equal(y,frame.target) and np.isin(y,range(1,7)).all() and np.isfinite(x).all()
    assert torch.cuda.is_available(),'Enable GPU in Colab'
    device=torch.device('cuda');torch.set_num_threads(min(4,os.cpu_count() or 1))
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    revision=None
    if args.model=='rubert':
        from huggingface_hub import HfApi
        rev_file=output/'rubert_revision.json'
        if rev_file.exists():revision=json.loads(rev_file.read_text())['revision']
        else:revision=HfApi().model_info('ai-forever/ruBert-base').sha;js(rev_file,{'revision':revision})
    config={'model':args.model,'loss':args.loss,'formula':'mean(w_true * (raw - true)^2)',
        'weight_rule':'N_train / (6 * n_train_level)' if args.loss=='weighted' else 'all ones',
        'features':features,'feature_hash':sha(path),'seed':42,'folds':5,'revision':revision,
        'batch_size':args.batch_size,'max_epochs':args.max_epochs,'patience':args.patience,'bert_epochs':args.bert_epochs,
        'mlp':{'layers':[159,256,128,1],'dropout':.5,'activation':'ReLU','optimizer':'Adam','lr':.001,
               'betas':[.9,.999],'eps':1e-8,'weight_decay':0,'validation_fraction':.2,'inner_seed':'42+fold','refit_seed':'1042+fold',
               'selection':'validation objective uses weights from inner train; fresh refit full outer train'},
        'rubert':{'lr':2e-5,'max_length':256,'weight_decay':.01,'no_decay':'bias and LayerNorm.weight','optimizer':'AdamW',
                  'betas':[.9,.999],'eps':1e-8,'scheduler':'linear, warmup=0','gradient_clip':1.,'mixed_precision':False},
        'save_bert_weights':args.save_bert_weights,'script_hash':sha(__file__),
        'versions':{p:version(p) for p in ['numpy','pandas','scikit-learn','torch','transformers']}}
    key=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
    tag=args.model+'_'+args.loss;out=output/tag/key;out.mkdir(parents=True,exist_ok=True);js(out/'config.json',config)
    enc=None
    if args.model=='rubert':
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained('ai-forever/ruBert-base',revision=revision)
        enc=tok(frame.text.tolist(),padding='max_length',truncation=True,max_length=256,return_tensors='pt')
        if args.save_bert_weights:tok.save_pretrained(out/'tokenizer')
    raw=np.full(len(y),np.nan);assigned=np.zeros(len(y),int);records=[];splits=[];class_records=[]
    for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(x,y),1):
        fd=out/f'fold_{fold}';fd.mkdir(exist_ok=True);cache=fd/'predictions.csv'
        splits.append({'fold':fold,'train_rows':tr.tolist(),'test_rows':te.tolist()})
        assert not set(tr)&set(te)
        if cache.exists():
            saved=pd.read_csv(cache);assert saved.row_id.tolist()==te.tolist() and np.array_equal(saved.target,y[te])
            raw[te]=saved.raw_prediction.to_numpy();print(tag,'fold',fold,'loaded',flush=True)
        else:
            raw[te]=fit_mlp(x,y,tr,te,fold,fd,args,device) if args.model=='mlp' else fit_rubert(enc,y,tr,te,fd,args,device,revision)
            assert np.isfinite(raw[te]).all()
            temp=fd/'predictions.tmp.csv';pd.DataFrame({'row_id':te,'target':y[te],'raw_prediction':raw[te]}).to_csv(temp,index=False);temp.replace(cache)
            release()
        assert np.isfinite(raw[te]).all();assigned[te]=fold;p=np.clip(np.floor(raw[te]+.5),1,6).astype(int)
        records.append({'model':tag,'fold':fold,**score(y[te],p)})
        precision,recall,f1,support=precision_recall_fscore_support(y[te],p,labels=range(1,7),zero_division=0)
        for i,label in enumerate(['A1','A2','B1','B2','C1','C2']):class_records.append({'model':tag,'fold':fold,'level':label,'precision':precision[i],'recall':recall[i],'F1':f1[i],'support':int(support[i])})
        pd.DataFrame(records).to_csv(out/'fold_metrics.csv',index=False);print(records[-1],flush=True)
    assert np.isfinite(raw).all() and (assigned>0).all();pred=np.clip(np.floor(raw+.5),1,6).astype(int)
    result=frame[['row_id','text','target']].copy();result['fold']=assigned;result['raw_prediction']=raw;result['prediction']=pred;result['relaxed_prediction']=np.where(abs(y-pred)<=1,y,pred)
    result.to_csv(out/'oof_predictions.csv',index=False,encoding='utf-8-sig')
    summary={'model':tag,**{'OOF '+k:v for k,v in score(y,pred).items()}}
    df=pd.DataFrame(records)
    for k in score(y,pred):summary[k+' mean']=float(df[k].mean());summary[k+' std']=float(df[k].std(ddof=1))
    precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=range(1,7),zero_division=0)
    pd.DataFrame({'level':['A1','A2','B1','B2','C1','C2'],'precision':precision,'recall':recall,'F1':f1,'support':support}).to_csv(out/'per_class.csv',index=False)
    pd.DataFrame(class_records).to_csv(out/'per_class_by_fold.csv',index=False)
    pd.DataFrame(confusion_matrix(y,pred,labels=range(1,7)),index=range(1,7),columns=range(1,7)).to_csv(out/'confusion_matrix.csv')
    js(out/'fold_indices.json',splits);js(out/'summary.json',summary);pd.DataFrame([summary]).to_csv(out/'summary.csv',index=False)
    js(output/f'latest_{tag}.json',{'directory':str(out)});print('COMPLETE',tag,summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model',choices=['mlp','rubert'],required=True);p.add_argument('--loss',choices=['plain','weighted'],required=True)
    p.add_argument('--batch-size',type=int,required=True);p.add_argument('--max-epochs',type=int,default=100);p.add_argument('--patience',type=int,default=20)
    p.add_argument('--bert-epochs',type=int,default=5);p.add_argument('--save-bert-weights',action='store_true')
    a=p.parse_args();assert min(a.batch_size,a.max_epochs,a.patience,a.bert_epochs)>0;run(a)
