import argparse, hashlib, json, re, warnings
from pathlib import Path
from functools import lru_cache
from importlib.metadata import version
import numpy as np
import pandas as pd
import pymorphy3
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score, mean_absolute_error, confusion_matrix, classification_report
from joblib import dump

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(data,out):
    data=Path(data);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((data/'bundle_manifest.json').read_text(encoding='utf8'))
    for n,h in manifest['files'].items():assert sha(data/n)==h,n
    train=pd.read_csv(data/'train.csv');test=pd.read_csv(data/'test.csv')
    assert len(train)==7322 and len(test)==1618
    train_text=train.fragment.tolist();test_text=test.text.tolist()
    yt=train['textbook-assigned cefr level'].to_numpy();y=test.median_rating_round.to_numpy()
    assert np.isin(yt,range(1,7)).all() and np.isin(y,range(1,7)).all()
    yt=yt.astype(int);y=y.astype(int)
    normalize=lambda t:' '.join(t.lower().split())
    overlap=set(map(normalize,train_text))&set(map(normalize,test_text))
    assert not overlap,'Exact normalized train/test overlap'
    levels=['A1','A2','B1','B2','C1']
    vocabs={l:{s.strip().lower() for s in (data/f'new_vocab_{l.lower()}.txt').read_text(encoding='utf8').splitlines() if s.strip()} for l in levels}
    assert all(vocabs.values())
    morph=pymorphy3.MorphAnalyzer()
    @lru_cache(maxsize=200000)
    def lemma(w):return morph.parse(w)[0].normal_form
    columns=[l+'_tokens' for l in levels]
    matrices={}
    for split,texts in [('train',train_text),('test',test_text)]:
        rows=[]
        for i,text in enumerate(texts):
            tokens=[lemma(w) for w in re.findall(r'[а-яё]+',text.lower())]
            if not tokens:raise ValueError(f'{split} row {i}: no Cyrillic words')
            rows.append({'row_id':i,'text':text,**{l+'_tokens':round(sum(w not in vocabs[l] for w in tokens)/len(tokens)*100,2) for l in levels}})
            if (i+1)%1000==0:print(f'{split}: {i+1}/{len(texts)}',flush=True)
        frame=pd.DataFrame(rows);frame.to_csv(out/f'{split}_features.csv',index=False,encoding='utf-8-sig')
        matrices[split]=frame[columns].to_numpy()
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=42))
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(matrices['train'],yt)
    pred=model.predict(matrices['test']);probs=model.predict_proba(matrices['test'])
    relaxed=np.where(abs(pred-y)<=1,y,pred)
    metrics={'experiment':'logreg_vocab_external','protocol':'external_train_expert_test',
        'n_train':len(train),'n_test':len(test),
        'Macro F1':float(f1_score(y,pred,labels=range(1,7),average='macro',zero_division=0)),
        'Relaxed Macro F1':float(f1_score(y,relaxed,labels=range(1,7),average='macro',zero_division=0)),
        'MAE':float(mean_absolute_error(y,pred))}
    assert metrics['Relaxed Macro F1']>=metrics['Macro F1']-1e-12
    result=pd.DataFrame({'row_id':range(len(y)),'text':test_text,'y_true':y,'prediction':pred,'prediction_relaxed':relaxed})
    for i,c in enumerate(model.classes_):result[f'probability_{c}']=probs[:,i]
    result.to_csv(out/'predictions_1618.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([metrics]).to_csv(out/'metrics.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf8')
    pd.DataFrame(confusion_matrix(y,pred,labels=range(1,7)),index=range(1,7),columns=range(1,7)).to_csv(out/'confusion_matrix.csv')
    pd.DataFrame(classification_report(y,pred,labels=range(1,7),target_names=['A1','A2','B1','B2','C1','C2'],output_dict=True,zero_division=0)).T.to_csv(out/'classification_report.csv')
    dump(model,out/'logreg_pipeline.joblib')
    config={'data_hashes':manifest['files'],'features':columns,'text_processing':'Cyrillic regex [а-яё]+, lowercase, first pymorphy3 parse normal_form; all POS',
        'feature_values':'token OOV percentage, round to 2 decimal places; independent vocabularies, no unions',
        'train_target':'textbook-assigned cefr level','test_target':'median_rating_round',
        'versions':{p:version(p) for p in ['numpy','pandas','scikit-learn','pymorphy3','pymorphy3-dicts-ru']},
        'estimator':model[-1].get_params(),'script_sha256':sha(__file__),'overlap_normalized':len(overlap),
        'n_iter':model[-1].n_iter_.tolist()}
    (out/'manifest.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(metrics,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.data,a.output)
