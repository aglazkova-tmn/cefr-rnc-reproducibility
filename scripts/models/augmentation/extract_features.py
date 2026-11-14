
import argparse, os, sys, json, hashlib, subprocess, copy, time
from pathlib import Path
from importlib.metadata import version
from functools import lru_cache

parser=argparse.ArgumentParser()
parser.add_argument('--bundle',required=True)
parser.add_argument('--results',required=True)
parser.add_argument('--mystem',required=True)
parser.add_argument('--batch-size',type=int,default=16)
parser.add_argument('--csv',default='')
parser.add_argument('--sep',default=',')
args=parser.parse_args()
bundle=Path(args.bundle).resolve()
project=bundle/'project'
os.environ['STANZA_RESOURCES_DIR']=str(bundle/'nlp_models/stanza')
os.environ['HF_HOME']=str(bundle/'nlp_models/huggingface')
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np
import pandas as pd
import torch
import stanza
import pymystem3
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, mean_absolute_error, classification_report, confusion_matrix
from joblib import dump

assert 1 <= args.batch_size <= 64, 'batch-size должен быть от 1 до 64'
if not torch.cuda.is_available():
    raise RuntimeError('GPU недоступен. Colab: Среда выполнения → Сменить среду выполнения → GPU, затем запустите ячейки заново.')
torch.set_num_threads(min(4,os.cpu_count() or 1))
print('GPU:',torch.cuda.get_device_name(0),flush=True)
sys.path.insert(0,str(project))
os.chdir(project)
from app.analyzer_foreign import AnalyzerForeign
from app.analyzer_syntax_features import AnalyzerSyntaxFeatures
from app.connectors.connector_analyser import ConnectorAnalyzer
from app.analyzer_morpheme import AnalyzerMorpheme

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic_json(path,value):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    tmp.replace(path)

from final_models_io import read_new_texts
if args.csv:
    input_path=Path(args.csv).resolve()
    test=read_new_texts(input_path,args.sep)[['text']].copy()
    # Placeholder for the existing checkpoint schema, never used as a training label.
    test['target']=0
else:
    input_path=bundle/'datasets/test.csv'
    test=pd.read_csv(input_path,keep_default_na=False)
    test['target']=test['median_rating_round']
frames={'test':test}
features=(project/'config_features.txt').read_text(encoding='utf-8').splitlines()
assert len(features)==len(set(features))==159
assert len(test)>0
if not args.csv: assert len(test)==1618
for frame in frames.values():
    assert frame.text.map(lambda t:isinstance(t,str) and bool(t.strip())).all()
    if not args.csv: assert np.isin(frame.target,np.arange(1,7)).all()
versions={p:version(p) for p in ['numpy','pandas','scikit-learn','stanza','torch','spacy','ru_core_news_lg','lexicalrichness','pymorphy3','pymystem3']}
identity={
    'bundle_manifest':sha(bundle/'bundle_manifest.json'),
    'runner':sha(__file__),
    'datasets':{'test':sha(input_path)}, 'csv_separator':args.sep, 'new_unlabeled_csv':bool(args.csv),
    'versions':versions,
    'features':features,
    'device':torch.cuda.get_device_name(0),
}
run_id=hashlib.sha256(json.dumps(identity,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:20]
out=Path(args.results).resolve()/run_id
cache=out/'feature_checkpoints'
cache.mkdir(parents=True,exist_ok=True)
manifest={**identity,'run_id':run_id,'n_texts':len(test),
    'target':'none (placeholder zero)' if args.csv else 'median_rating_round',
    'pipeline':'159 raw features; no scaling or feature selection',
    'test_used_for_fit':False,'rounding':'floor(raw+0.5), clip [1,6]',
    'missing_feature_policy':'fill with zero as in original run_analyzer.py',
    'gpu':'Stanza; spaCy and MyStem use CPU',
    'status':'extracting_features'}
atomic_json(out/'manifest.json',manifest)
atomic_json(Path(args.results)/'latest_run.json',{'run_id':run_id,'directory':str(out)})
print('Результаты и контрольные точки:',out,flush=True)
print('Текстов:',len(test),'Признаков:',len(features),flush=True)

CHUNK_SIZE=32 # fixed checkpoint boundaries; inference batch size can be changed on resume
def checkpoint_path(split,start):return cache/f'{split}_{start:06d}.json'
pending=any(not checkpoint_path(split,i).exists() for split,frame in frames.items() for i in range(0,len(frame),CHUNK_SIZE))
if pending:
    ms=pymystem3.Mystem(mystem_bin=args.mystem,entire_input=False,disambiguation=True)
    foreign=AnalyzerForeign(ms,use_model=False)
    clean=[foreign._AnalyzerForeign__clean_text(t) for f in frames.values() for t in f.text]
    lines=list(dict.fromkeys(line for text in clean for line in text.splitlines() if line))
    print('Пакетный морфологический разбор MyStem...',flush=True)
    process=subprocess.run([args.mystem,*ms._mystemargs],input=('\n'.join(lines)+'\n').encode(),capture_output=True,check=True)
    stream=process.stdout.decode('utf-8'); decoder=json.JSONDecoder(); parsed=[]; pos=0
    while pos<len(stream):
        if stream[pos].isspace():pos+=1;continue
        obj,pos=decoder.raw_decode(stream,pos);parsed.append(obj)
    assert len(parsed)==len(lines),'Не совпало число строк MyStem'
    analyses=dict(zip(lines,parsed))
    class CachedMyStem:
        def analyze(self,text):return copy.deepcopy([t for line in text.splitlines() if line for t in analyses[line]])
        def lemmatize(self,text):return list(filter(None,map(pymystem3.Mystem._get_lemma,self.analyze(text))))
    foreign.mystem=CachedMyStem()
    syntax=AnalyzerSyntaxFeatures(ms)
    syntax.nlp=lru_cache(maxsize=128)(syntax.nlp)
    connector=ConnectorAnalyzer()
    raw_stanza=connector.nlp
    assert str(raw_stanza.device).startswith('cuda'), f'Stanza использует {raw_stanza.device} вместо GPU'
    print('Stanza работает на:',raw_stanza.device,flush=True)
    stanza_cache={}
    from razdel import sentenize
    def prepare_stanza(texts):
     global stanza_cache
     from types import SimpleNamespace
     sentences={text:[s.text for s in sentenize(text)] for text in texts}
     inputs=list(dict.fromkeys(t for text in texts for t in [text,*sentences[text]]))
     tokenized={t:raw_stanza(t,processors='tokenize') for t in inputs}
     signature=lambda doc:[[w.text for w in s.words] for s in doc.sentences]
     reusable={text for text in texts if sentences[text]!=[text] and signature(tokenized[text])==[words for s in sentences[text] for words in signature(tokenized[s])]}
     required=list(dict.fromkeys(t for text in texts for t in [*sentences[text],*([] if text in reusable else [text])]))
     docs=raw_stanza([tokenized[t] for t in required],processors='pos,lemma,depparse')
     for doc in docs:
      for sentence in doc.sentences:
       for token in sentence.tokens:
        frozen=token.to_dict()
        token.to_dict=lambda frozen=frozen:[dict(item) for item in frozen]
     stanza_cache=dict(zip(required,docs))
     for text in reusable:
      stanza_cache[text]=SimpleNamespace(sentences=[s for part in sentences[text] for s in stanza_cache[part].sentences])
    def cached_stanza(text):
     return stanza_cache[text] if text in stanza_cache else raw_stanza(text)
    connector.nlp=cached_stanza
    morpheme=AnalyzerMorpheme(nlp=connector.nlp)


tables={};issues=[]
for split,frame in frames.items():
    rows=[]; started=time.time()
    for start in range(0,len(frame),CHUNK_SIZE):
        stop=min(start+CHUNK_SIZE,len(frame))
        path=checkpoint_path(split,start)
        if path.exists():
            items=json.loads(path.read_text(encoding='utf-8'))
            assert len(items)==stop-start
            for offset,item in enumerate(items,start):
                assert item['row_id']==offset and item['text']==frame.text.iloc[offset]
                assert item['target']==float(frame.target.iloc[offset])
        else:
            items=[]
            for part in range(start,stop,args.batch_size):
                end=min(part+args.batch_size,stop)
                try:
                    prepare_stanza(frame.text.iloc[part:end].tolist())
                except torch.cuda.OutOfMemoryError:
                    raise RuntimeError('Не хватило памяти GPU. Уменьшите BATCH_SIZE (например, до 4) и повторите ячейку запуска. Готовые блоки будут пропущены.') from None
                for i in range(part,end):
                    text=frame.text.iloc[i]
                    try:
                        lex=foreign.start_foreign(text)
                        values={**lex,**syntax.start_syntax(text),**connector.analyze(text),**morpheme.analyze(text)}
                        missing=[f for f in features if f not in values or pd.isna(values[f])]
                        vals={f:float(values[f]) if f in values and not pd.isna(values[f]) else 0. for f in features}
                        assert np.isfinite(list(vals.values())).all()
                        items.append({'row_id':i,'text':text,'target':float(frame.target.iloc[i]),'features':vals,
                            'missing_features':missing,'foreign_rejected':lex.get('text_ok') is False,
                            'foreign_reason':lex.get('text_error_message')})
                    except Exception:
                        print('Ошибка:',split,'строка',i,flush=True)
                        raise
            atomic_json(path,items)
        for item in items:
            rows.append({'row_id':item['row_id'],'text':item['text'],'target':item['target'],**item['features']})
            if item['missing_features'] or item['foreign_rejected']:
                issues.append({'split':split,'row_id':item['row_id'],'foreign_rejected':item['foreign_rejected'],
                    'missing_features':'|'.join(item['missing_features']),'reason':item['foreign_reason']})
        print(f'{split}: {stop}/{len(frame)}; сохранено; прошло {time.time()-started:.0f} с',flush=True)
    tables[split]=pd.DataFrame(rows)
    assert len(tables[split])==len(frame)
    tables[split].to_csv(out/f'{split}_features.csv',index=False,encoding='utf-8-sig')

feature_path=out/'features.csv'
tables['test'].to_csv(feature_path,index=False,encoding='utf-8-sig')
(out/'input_features.txt').write_text('\n'.join(features)+'\n',encoding='utf-8')
pd.DataFrame(issues,columns=['split','row_id','foreign_rejected','missing_features','reason']).to_csv(out/'feature_fill_audit.csv',index=False,encoding='utf-8-sig')
manifest.update(status='complete',feature_table_sha256=sha(feature_path),features=features,
    feature_fill_rows=len(issues),foreign_rejected_rows=sum(i['foreign_rejected'] for i in issues))
atomic_json(out/'features_manifest.json',manifest)
atomic_json(out/'manifest.json',manifest)
print('Признаки сохранены:',feature_path,flush=True)
if pending:ms.close()
