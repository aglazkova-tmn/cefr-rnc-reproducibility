import sys,json,os
from pathlib import Path
root=Path(os.environ['CEFR_ORIGINAL_WORK']).resolve()
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,mean_absolute_error,classification_report,confusion_matrix
p=root/'outputs'/'Textometr_исходная_модель'
df=pd.read_csv(p/'raw_results.csv')
assert len(df)==1618
assert set(df.status).issubset({'ok','rejected'}),df.status.value_counts().to_dict()
assert np.isfinite(df.loc[df.status=='ok','level_number']).all()
assert df.loc[df.status=='rejected','level_number'].isna().all()
# Reproduce original run_analyzer.py export fillna(0). Preserve the absent raw
# model prediction separately and mark every placeholder explicitly.
df['is_placeholder']=df.status=='rejected'
df['level_number_export']=df.level_number.fillna(0)
df['prediction']=np.clip(np.floor(df.level_number_export.to_numpy()+.5),1,6).astype(int)
yt=df.y_true.to_numpy(dtype=int);yp=df.prediction.to_numpy()
df['prediction_relaxed']=np.where(np.abs(yp-yt)<=1,yt,yp)
labels=np.arange(1,7)
strict=f1_score(yt,yp,labels=labels,average='macro',zero_division=0)
relaxed=f1_score(yt,df.prediction_relaxed,labels=labels,average='macro',zero_division=0)
assert relaxed>=strict
metrics={'model':'Textometr original Ridge','evaluation':'original_export_fillna0_then_clip','n':len(df),'n_placeholders':int(df.is_placeholder.sum()),'target':'median_rating_round',
 'Macro F1':float(strict),'Relaxed Macro F1':float(relaxed),'MAE':float(mean_absolute_error(yt,yp))}
df.to_csv(p/'predictions_rounded.csv',index=False,encoding='utf-8-sig')
valid=~df.is_placeholder.to_numpy()
valid_metrics={'model':'Textometr original Ridge','evaluation':'valid_model_predictions_only','n':int(valid.sum()),'n_placeholders':0,'target':'median_rating_round',
 'Macro F1':float(f1_score(yt[valid],yp[valid],labels=labels,average='macro',zero_division=0)),
 'Relaxed Macro F1':float(f1_score(yt[valid],df.prediction_relaxed.to_numpy()[valid],labels=labels,average='macro',zero_division=0)),
 'MAE':float(mean_absolute_error(yt[valid],yp[valid]))}
pd.DataFrame([metrics,valid_metrics]).to_csv(p/'metrics.csv',index=False,encoding='utf-8-sig')
(p/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
names=['A1','A2','B1','B2','C1','C2']
pd.DataFrame(classification_report(yt,yp,labels=labels,target_names=names,output_dict=True,zero_division=0)).T.to_csv(p/'classification_report.csv',encoding='utf-8-sig')
pd.DataFrame(confusion_matrix(yt,yp,labels=labels),index=names,columns=names).to_csv(p/'confusion_matrix.csv',encoding='utf-8-sig')
manifest=json.loads((p/'manifest.json').read_text(encoding='utf-8'))
manifest.update(rounding='floor(level_number + 0.5), then clip to [1,6]; level_number uses original rounding to 0.1 and bounds [0.5,7]; refused inputs use export fillna(0) as in original run_analyzer.py',
 preprocessing='original AnalyzerForeign feature extraction; same MyStem flags, batched independent lines; exact match verified on first 15 sequential rows',
 omitted='syntax; presentation-only keyword/grammar/MWE reports after model prediction',
 relaxed='replace prediction with truth if abs(error)<=1; macro F1 fixed labels 1..6',
 sklearn_version='1.3.1',n_splits=None,training='none: use supplied model.joblib')
(p/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(metrics,ensure_ascii=False))
print(json.dumps(valid_metrics,ensure_ascii=False))
print('Predicted distribution',df.prediction.value_counts().sort_index().to_dict())
print('Half-integer outputs',int(((df.level_number%1)==.5).sum()))
