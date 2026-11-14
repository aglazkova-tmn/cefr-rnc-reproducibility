import sys,os,json,hashlib,time,traceback,subprocess,copy
from pathlib import Path
root=Path(os.environ['CEFR_ORIGINAL_WORK']).resolve()
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import pymystem3
from sklearn.metrics import f1_score,mean_absolute_error,classification_report,confusion_matrix
project=Path(os.environ['CEFR_ORIGINAL_PROJECT']).resolve()
input_csv_path=Path(os.environ['CEFR_INPUT_CSV']).resolve()
mystem_path=Path(os.environ['CEFR_MYSTEM']).resolve()
sys.path.insert(0,str(project));os.chdir(project)
from app.analyzer_foreign import AnalyzerForeign
source=input_csv_path
df=pd.read_csv(source)
y=df.median_rating_round.to_numpy(dtype=float)
assert np.isin(y,np.arange(1,7)).all()
output=root/'outputs'/'Textometr_исходная_модель'
output.mkdir(exist_ok=True)
cache=root/'work'/'textometr_cache';cache.mkdir(exist_ok=True)
mystem=pymystem3.Mystem(mystem_bin=str(mystem_path),entire_input=False,disambiguation=True)
analyzer=AnalyzerForeign(mystem)
original_predict=analyzer.ridge.predict
capture={}
def predict_and_record(x):
 result=original_predict(x)
 capture.update(raw=float(result[0]),features=np.asarray(x).ravel().tolist())
 return result
analyzer.ridge.predict=predict_and_record
# Presentation-only replacement: identical numerical postprocessing; omitted
# keywords, grammar explanations and MWE lists never feed into Ridge.predict.
def numeric_report(prediction):
 value=float(np.asarray(prediction).reshape(-1)[0])
 value=max(.5,min(7.,round(value,1)))
 analyzer.data_about_text.update(level_number=value,level_int=min(6,round(value)),level_for_scale=round(value*1.4,1))
analyzer._AnalyzerForeign__tell_me_about_text=numeric_report
# On Windows pymystem3 restarts its executable for each analyze/lemmatize call.
# Batch the same independent input lines using identical MyStem flags, then
# replay their parsed JSON. Check equivalence against already completed rows.
clean_texts=[analyzer._AnalyzerForeign__clean_text(t) for t in df.text]
lines=list(dict.fromkeys(line for text in clean_texts for line in text.splitlines() if line))
batch_cache=cache/'mystem_batch.json'
if batch_cache.exists():
 analyses=json.loads(batch_cache.read_text(encoding='utf-8'))
 assert list(analyses)==lines
else:
 proc=subprocess.run([str(mystem_path),*mystem._mystemargs],
   input=('\n'.join(lines)+'\n').encode('utf-8'),capture_output=True,check=True,
   creationflags=subprocess.CREATE_NO_WINDOW)
 stream=proc.stdout.decode('utf-8');decoder=json.JSONDecoder();parsed=[]
 while stream.strip():
  stream=stream.lstrip();obj,end=decoder.raw_decode(stream);parsed.append(obj);stream=stream[end:]
 assert len(parsed)==len(lines),(len(parsed),len(lines))
 analyses=dict(zip(lines,parsed))
 batch_cache.write_text(json.dumps(analyses,ensure_ascii=False),encoding='utf-8')
class BatchMystem:
 def analyze(self,text):
  return copy.deepcopy([token for line in text.splitlines() if line for token in analyses[line]])
 def lemmatize(self,text):
  return list(filter(None,map(pymystem3.Mystem._get_lemma,self.analyze(text))))
analyzer.mystem=BatchMystem()
for prior in sorted(cache.glob('[0-9]*.json'))[:15]:
 row=json.loads(prior.read_text(encoding='utf-8'))
 if row['status']!='ok':continue
 capture.clear();result=analyzer.start_foreign(row['text'])
 assert result['level_number']==row['level_number']
 np.testing.assert_allclose(capture['features'],row['model_features'],rtol=0,atol=0)
 assert capture['raw']==row['ridge_raw']
print('Batch MyStem equivalence verified against sequential features/predictions',flush=True)
rows=[]
started=time.time()
for i,text in enumerate(df.text):
 path=cache/f'{i:06d}.json'
 if path.exists():
  row=json.loads(path.read_text(encoding='utf-8'));assert row['text']==text
 else:
  capture.clear()
  row={'row_id':i,'text':text,'y_true':int(y[i])}
  if 'id' in df:row['source_id']=str(df.id.iloc[i])
  try:
   result=analyzer.start_foreign(text)
   if 'level_number' in result:
    row.update(status='ok',level_number=float(result['level_number']))
   else:
    row.update(status='rejected',reason=result.get('text_error_message','no level'))
  except Exception as e:
   row.update(status='error',reason=f'{type(e).__name__}: {e}')
  if 'raw' in capture:
   row['ridge_raw']=capture['raw'];row['model_features']=capture['features']
   if 'level_number' not in row:
    # The original predictor already finished. Keep its result separately;
    # do not masquerade a failed original pipeline as a completed one.
    row['level_number_before_aux_error']=max(.5,min(7.,round(max(0.,capture['raw']),1)))
  path.write_text(json.dumps(row,ensure_ascii=False),encoding='utf-8')
 rows.append(row)
 if (i+1)%50==0 or i==0:
  print(i+1,'/',len(df),'seconds',round(time.time()-started,1),'statuses',pd.Series([r['status'] for r in rows]).value_counts().to_dict(),flush=True)
mystem.close()
table=pd.DataFrame(rows)
print('FINAL',table.status.value_counts().to_dict(),flush=True)
table.drop(columns=['model_features'],errors='ignore').to_csv(output/'raw_results.csv',index=False,encoding='utf-8-sig')
features=pd.DataFrame([r.get('model_features',[None]*72) for r in rows],columns=AnalyzerForeign.COLUMNS_NEEDED_FOREIGN)
features.insert(0,'row_id',range(len(df)));features.to_csv(output/'model_inputs.csv',index=False,encoding='utf-8-sig')
table[table.status!='ok'].drop(columns=['model_features'],errors='ignore').to_csv(output/'processing_issues.csv',index=False,encoding='utf-8-sig')
manifest={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'model_sha256':hashlib.sha256((project/'data/model.joblib').read_bytes()).hexdigest(),
 'target':'median_rating_round','n_total':len(df),'status_counts':table.status.value_counts().to_dict(),'syntax':False,'model_changes':False}
(output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
