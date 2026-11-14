"""Four unweighted augmented experiments; fixed original CV splits and full-data inference."""
import os
os.environ['USE_TF']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse,gc,hashlib,json
from pathlib import Path
from importlib.metadata import version
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset,DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,mean_absolute_error,precision_recall_fscore_support,confusion_matrix
from joblib import dump,load
from transformers import AutoTokenizer,AutoModelForSequenceClassification,get_linear_schedule_with_warmup
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mlp_common import MLP,seed_everything,make_training,epoch_train
from final_models_io import read_new_texts,rounded_levels,save_predictions

LEVELS=['A1','A2','B1','B2','C1','C2']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp')
    t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');t.replace(p)
def release():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
def scores(y,p):
    relaxed=np.where(abs(y-p)<=1,y,p)
    return {'Macro F1':float(f1_score(y,p,labels=range(1,7),average='macro',zero_division=0)),
            'Relaxed Macro F1':float(f1_score(y,relaxed,labels=range(1,7),average='macro',zero_division=0)),
            'MAE':float(mean_absolute_error(y,p))}
def report(out,name,y,p,fold):
    records=[{'model':name,'fold':f,**scores(y[fold==f],p[fold==f])} for f in range(1,6)]
    summary={'model':name,**{'OOF '+k:v for k,v in scores(y,p).items()}}
    for metric in scores(y,p):
        values=[r[metric] for r in records];summary[metric+' mean']=float(np.mean(values));summary[metric+' std']=float(np.std(values,ddof=1))
    pr,re,f1,sup=precision_recall_fscore_support(y,p,labels=range(1,7),zero_division=0)
    pd.DataFrame({'level':LEVELS,'precision':pr,'recall':re,'F1':f1,'support':sup}).to_csv(out/f'{name}_per_class.csv',index=False)
    cm=confusion_matrix(y,p,labels=range(1,7));pd.DataFrame(cm,index=LEVELS,columns=LEVELS).to_csv(out/f'{name}_confusion_matrix.csv')
    fig,ax=plt.subplots(figsize=(8,8));ax.imshow(cm,cmap='viridis',vmin=0)
    for i in range(6):
        for j in range(6):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='#440154' if cm[i,j]>cm.max()/2 else '#fde725',fontsize=14)
    ax.set(xticks=range(6),yticks=range(6),xticklabels=LEVELS,yticklabels=LEVELS,xlabel='Predicted CEFR level',ylabel='Gold CEFR level',title=name.replace('_',' '))
    fig.tight_layout();fig.savefig(out/f'{name}_confusion_matrix.png',dpi=200);plt.close(fig)
    return summary,records
def table(directory,frame,features):
    directory=Path(directory);p=directory/'features.csv'
    if not p.exists():p=directory/'features_1618.csv'
    m=json.loads((directory/'features_manifest.json').read_text(encoding='utf8'))
    assert m['features']==features and m['feature_table_sha256']==sha(p)
    t=pd.read_csv(p,keep_default_na=False)
    assert t.row_id.tolist()==list(range(len(frame))) and t.text.tolist()==frame.text.tolist()
    x=t[features].to_numpy(dtype=np.float32);assert np.isfinite(x).all()
    return x
def mlp_probs(model,x,device):
    model.eval();result=[]
    with torch.inference_mode():
        for start in range(0,len(x),256):result.append(torch.softmax(model(torch.as_tensor(x[start:start+256],dtype=torch.float32,device=device)),1).cpu().numpy())
    return np.concatenate(result)
def select_epochs(x,y,extra_x,extra_y,it,iv,seed,args,device,out):
    train_x=np.concatenate([x[it],extra_x]);train_y=np.concatenate([y[it],extra_y])
    scaler=StandardScaler().fit(train_x)
    model,opt,xt,yt=make_training(scaler.transform(train_x),train_y,device,seed)
    xv=torch.as_tensor(scaler.transform(x[iv]),dtype=torch.float32,device=device)
    yv=torch.as_tensor(y[iv]-1,dtype=torch.long,device=device)
    best=float('inf');best_epoch=0;wait=0;history=[]
    for epoch in range(1,args.mlp_epochs+1):
        loss=epoch_train(model,opt,xt,yt,args.mlp_batch);model.eval()
        with torch.no_grad():val=torch.nn.functional.cross_entropy(model(xv),yv).item()
        assert np.isfinite(loss) and np.isfinite(val)
        history.append({'epoch':epoch,'train_CE':loss,'validation_CE':val})
        if val<best:best=val;best_epoch=epoch;wait=0
        else:wait+=1
        if epoch==1 or epoch%10==0:print(f'MLP validation epoch {epoch}: CE={val:.5f}',flush=True)
        if wait>=args.patience:break
    pd.DataFrame(history).to_csv(out/'early_stopping.csv',index=False)
    del model,opt,xt,yt,xv,yv;release();return best_epoch
def fit_mlp(x,y,epochs,seed,args,device,out):
    out.mkdir(parents=True,exist_ok=True);scaler=StandardScaler().fit(x)
    model,opt,xt,yt=make_training(scaler.transform(x),y,device,seed);history=[]
    for e in range(1,epochs+1):
        loss=epoch_train(model,opt,xt,yt,args.mlp_batch);assert np.isfinite(loss)
        history.append({'epoch':e,'train_CE':loss})
        if e==1 or e%10==0 or e==epochs:print(f'MLP refit {e}/{epochs}: CE={loss:.5f}',flush=True)
    dump(scaler,out/'scaler.joblib')
    torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'input_dim':159},out/'mlp.pt')
    pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
    return model,scaler
def fit_rubert(texts,y,revision,args,device,out):
    out.mkdir(parents=True,exist_ok=True);seed_everything(42)
    tokenizer=AutoTokenizer.from_pretrained('ai-forever/ruBert-base',revision=revision)
    enc=tokenizer(list(texts),padding='max_length',truncation=True,max_length=256,return_tensors='pt');keys=list(enc)
    model=AutoModelForSequenceClassification.from_pretrained('ai-forever/ruBert-base',revision=revision,num_labels=1,problem_type='regression').to(device)
    no_decay=lambda n:n.endswith('bias') or 'LayerNorm.weight' in n
    groups=[{'params':[p for n,p in model.named_parameters() if not no_decay(n)],'weight_decay':.01},
            {'params':[p for n,p in model.named_parameters() if no_decay(n)],'weight_decay':0.}]
    opt=torch.optim.AdamW(groups,lr=2e-5,betas=(.9,.999),eps=1e-8)
    loader=DataLoader(TensorDataset(*[enc[k] for k in keys],torch.tensor(y,dtype=torch.float32)),batch_size=args.rubert_batch,shuffle=True)
    sched=get_linear_schedule_with_warmup(opt,0,args.rubert_epochs*len(loader));history=[]
    for e in range(1,args.rubert_epochs+1):
        model.train();total=0.
        for batch in loader:
            opt.zero_grad(set_to_none=True);inputs={k:v.to(device) for k,v in zip(keys,batch[:-1])}
            loss=model(**inputs,labels=batch[-1].to(device)).loss
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite MSE')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();sched.step();total+=loss.item()*len(batch[-1])
        history.append({'epoch':e,'train_MSE':total/len(y)});print(f'RuBERT epoch {e}/{args.rubert_epochs}: {total/len(y):.5f}',flush=True)
    model.save_pretrained(out/'rubert');tokenizer.save_pretrained(out/'rubert')
    pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
    return model,tokenizer
def rubert_raw(model,tokenizer,texts,batch_size,device):
    model.eval();result=[]
    with torch.inference_mode():
        for start in range(0,len(texts),batch_size):
            enc=tokenizer(list(texts[start:start+batch_size]),padding='max_length',truncation=True,max_length=256,return_tensors='pt')
            result.extend(model(**{k:v.to(device) for k,v in enc.items()}).logits.reshape(-1).cpu().numpy().tolist())
            if start%1024==0:print(f'RuBERT prediction {min(start+batch_size,len(texts))}/{len(texts)}',flush=True)
    result=np.array(result);assert np.isfinite(result).all();return result
def main(args):
    bundle=Path(args.bundle);features=(bundle/'project/config_features.txt').read_text(encoding='utf8').splitlines()
    base=read_new_texts(bundle/'datasets/test.csv');y=base.median_rating_round.to_numpy(dtype=int)
    assert len(base)==1618 and len(features)==159 and np.isin(y,range(1,7)).all()
    device=torch.device('cuda');assert torch.cuda.is_available(),'Enable GPU in Colab'
    root=Path(args.results);root.mkdir(parents=True,exist_ok=True)
    paths=json.loads(Path(args.feature_paths).read_text(encoding='utf8'))
    x=table(paths['base'],base,features)
    splits=json.loads((bundle/'datasets/fold_indices.json').read_text(encoding='utf8'))
    assignments=np.zeros(len(y),int)
    for sp in splits:
        tr,te=sp['train_rows'],sp['test_rows'];assert set(tr).isdisjoint(te) and set(tr)|set(te)==set(range(1618))
        assert not assignments[te].any();assignments[te]=sp['fold']
        it,iv=sp['mlp_inner_train_rows'],sp['mlp_inner_validation_rows'];assert set(it).isdisjoint(iv) and set(it)|set(iv)==set(tr)
    assert (assignments>0).all()
    revision=json.loads((bundle/'datasets/rubert_revision.json').read_text())['revision']
    config={'seed':42,'splits_sha256':sha(bundle/'datasets/fold_indices.json'),'bundle_sha256':sha(bundle/'bundle_manifest.json'),
            'runner_sha256':sha(__file__),'revision':revision,'mlp_epochs':args.mlp_epochs,'patience':args.patience,
            'mlp_batch':args.mlp_batch,'rubert_epochs':args.rubert_epochs,'rubert_batch':args.rubert_batch,
            'feature_manifests':{k:sha(Path(v)/'features_manifest.json') for k,v in paths.items()},
            'versions':{p:version(p) for p in ['torch','transformers','numpy','scikit-learn']},
            'protocol':'Augmentation in training only; original 1618 OOF; inner validation original only; unweighted CE/MSE; final MLP 100 epochs by default',
            'MLP':'159-256-128-6, ReLU, dropout .5, Adam lr .001, betas .9/.999, eps 1e-8; StandardScaler fit on augmented training partition',
            'RuBERT':'max_length256, AdamW2e-5, decay .01 excluding bias/LayerNorm, linear scheduler, warmup0, clip1, full precision',
            'rounding':'floor(x+.5), clip1..6; ensemble=mean(MLP expected level, RuBERT raw)'}
    run=root/hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:20];run.mkdir(exist_ok=True);js(run/'config.json',config);js(root/'latest_run.json',{'directory':str(run)})
    all_summaries=[];all_folds=[]
    school=read_new_texts(bundle/'datasets/school.csv');sx=table(paths['school'],school,features)
    assert len(school)==10000  # Private first-sentence assertion removed for publication.
    from augmentation_data import norm
    base_keys={norm(t) for t in base.text}
    expected_extra=json.loads((bundle/'datasets/augmentation_manifest.json').read_text(encoding='utf8'))['n_deepseek_A1']
    result_columns={}
    for source in ['deepseek','textbooks']:
        extra=read_new_texts(bundle/f'datasets/A1_{source}.csv');ey=extra.target.to_numpy(dtype=int);assert (ey==1).all()
        assert len(extra)==expected_extra and len({norm(t) for t in extra.text})==len(extra)
        assert not base_keys & {norm(t) for t in extra.text}
        ax=table(paths[source],extra,features);augx=np.concatenate([x,ax]);augy=np.concatenate([y,ey]);texts=base.text.tolist()+extra.text.tolist()
        out=run/source;out.mkdir(exist_ok=True);prob=np.full((len(y),6),np.nan);raw=np.full(len(y),np.nan)
        for sp in splits:
            fold=sp['fold'];tr=np.array(sp['train_rows']);te=np.array(sp['test_rows']);fd=out/f'fold_{fold}';fd.mkdir(exist_ok=True)
            train=np.concatenate([tr,np.arange(len(y),len(augy))])
            js(fd/'indices.json',{**sp,'augmentation_source':source,'augmentation_rows':len(extra)})
            mp=fd/'mlp_predictions.npz'
            if mp.exists():
                with np.load(mp) as cache:assert np.array_equal(cache['row_id'],te);prob[te]=cache['probabilities']
            else:
                epochs=select_epochs(x,y,ax,ey,sp['mlp_inner_train_rows'],sp['mlp_inner_validation_rows'],42+fold,args,device,fd)
                model,scaler=fit_mlp(augx[train],augy[train],epochs,1042+fold,args,device,fd/'mlp')
                prob[te]=mlp_probs(model,scaler.transform(x[te]),device);js(fd/'best_epoch.json',{'best_epoch':epochs})
                with (fd/'mlp_predictions.tmp').open('wb') as f:np.savez(f,row_id=te,probabilities=prob[te])
                (fd/'mlp_predictions.tmp').replace(mp);del model,scaler;release()
            rp=fd/'rubert_predictions.npz'
            if rp.exists():
                with np.load(rp) as cache:assert np.array_equal(cache['row_id'],te);raw[te]=cache['raw']
            else:
                model,tokenizer=fit_rubert([texts[i] for i in train],augy[train],revision,args,device,fd/'rubert')
                raw[te]=rubert_raw(model,tokenizer,base.text.iloc[te].tolist(),args.rubert_batch,device)
                with (fd/'rubert_predictions.tmp').open('wb') as f:np.savez(f,row_id=te,raw=raw[te])
                (fd/'rubert_predictions.tmp').replace(rp);del model,tokenizer;release()
            print(f'{source}: fold {fold}/5 complete',flush=True)
        assert np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-6) and np.isfinite(raw).all()
        expected=prob@np.arange(1,7);preds={'MLP':prob.argmax(1)+1,'RuBERT':rounded_levels(raw),'ensemble':rounded_levels((expected+raw)/2)}
        oof=pd.DataFrame({'row_id':range(len(y)),'text':base.text,'target':y,'fold':assignments,'rubert_raw':raw,'mlp_expected_level':expected,'ensemble_raw':(expected+raw)/2})
        for k in range(6):oof[f'mlp_probability_{k+1}']=prob[:,k]
        for name,p in preds.items():
            oof[name+'_prediction']=p;s,f=report(out,name+'_'+source,y,p,assignments);all_summaries.append(s);all_folds.extend(f)
        oof.to_csv(out/'oof_predictions.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(all_summaries).to_csv(run/'metrics_summary.csv',index=False);pd.DataFrame(all_folds).to_csv(run/'fold_metrics.csv',index=False)
        final=out/'full_models';final.mkdir(exist_ok=True)
        if not (final/'mlp_complete.json').exists():
            model,scaler=fit_mlp(augx,augy,args.mlp_epochs,42,args,device,final/'mlp')
            js(final/'mlp_complete.json',{'n_train':len(augy),'epochs':args.mlp_epochs})
        else:
            saved=torch.load(final/'mlp/mlp.pt',map_location='cpu',weights_only=True);model=MLP(159);model.load_state_dict(saved['state_dict']);model=model.to(device)
            scaler=load(final/'mlp/scaler.joblib')
        pp=mlp_probs(model,scaler.transform(sx),device);del model,scaler;release()
        if not (final/'rubert_complete.json').exists():
            model,tokenizer=fit_rubert(texts,augy,revision,args,device,final)
            js(final/'rubert_complete.json',{'n_train':len(augy),'epochs':args.rubert_epochs})
        else:
            tokenizer=AutoTokenizer.from_pretrained(final/'rubert',local_files_only=True)
            model=AutoModelForSequenceClassification.from_pretrained(final/'rubert',local_files_only=True).to(device)
        rr=rubert_raw(model,tokenizer,school.text.tolist(),args.rubert_batch,device);del model,tokenizer;release()
        ee=pp@np.arange(1,7);en=(ee+rr)/2
        for k in range(6):result_columns[f'{source}_mlp_probability_{k+1}']=pp[:,k]
        for k,v in {'mlp_class':pp.argmax(1)+1,'mlp_expected_level':ee,'rubert_raw':rr,'rubert_rounded':rounded_levels(rr),'ensemble_raw':en,'ensemble_rounded':rounded_levels(en)}.items():result_columns[source+'_'+k]=v
        from augmentation_data import norm
        seen={norm(t) for t in texts};result_columns[source+'_seen_in_training']=np.array([int(norm(t) in seen) for t in school.text])
        save_predictions(school,{k:v for k,v in result_columns.items() if k.startswith(source+'_')},run/f'school_{source}_predictions.csv')
    result=save_predictions(school,result_columns,run/'school_10000_two_augmented_ensembles.csv')
    for c in result:
        if c.endswith('_rounded'):assert result[c].isin(range(1,7)).all()
    print(pd.DataFrame(all_summaries).to_string(index=False),flush=True);print('RESULTS:',run,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--results',required=True);p.add_argument('--feature-paths',required=True)
    p.add_argument('--mlp-epochs',type=int,default=100);p.add_argument('--patience',type=int,default=20);p.add_argument('--mlp-batch',type=int,default=32)
    p.add_argument('--rubert-epochs',type=int,default=5);p.add_argument('--rubert-batch',type=int,default=16)
    args=p.parse_args();assert min(args.mlp_epochs,args.patience,args.mlp_batch,args.rubert_epochs,args.rubert_batch)>0;main(args)
