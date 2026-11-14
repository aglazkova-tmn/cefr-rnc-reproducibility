import argparse,json,hashlib,os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer,AutoModel
from huggingface_hub import HfApi
from importlib.metadata import version

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(args):
    root=Path(args.features).resolve()
    frames={s:pd.read_csv(root/f'{s}_features.csv') for s in ['train','test']}
    assert len(frames['train'])==7322 and len(frames['test'])==1618
    assert torch.cuda.is_available(),'Включите GPU'
    revision=HfApi().model_info('jinaai/jina-embeddings-v3',revision=args.revision).sha
    code_repo='jinaai/xlm-roberta-flash-implementation'
    code_revision=HfApi().model_info(code_repo,revision=args.code_revision).sha
    config={'model':'jinaai/jina-embeddings-v3','revision':revision,'prefix':'','task':'classification','code_repo':code_repo,'code_revision':code_revision,'use_flash_attn':False,
        'pooling':'attention-mask mean','normalize':'L2','max_length':args.max_length,'dtype':'float32','dimensions':1024,
        'input_files':{s:sha(root/f'{s}_features.csv') for s in frames},
        'versions':{p:version(p) for p in ['torch','transformers','numpy','einops']}}
    key=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:20]
    out=root/'jina_classification'/key;out.mkdir(parents=True,exist_ok=True)
    (out/'manifest.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    tokenizer=AutoTokenizer.from_pretrained(config['model'],revision=revision)
    model=None
    for split,frame in frames.items():
        folder=out/split;folder.mkdir(exist_ok=True);arrays=[];lengths=[]
        for start in range(0,len(frame),64):
            stop=min(start+64,len(frame));path=folder/f'{start:06d}.npz'
            if path.exists():
                with np.load(path,allow_pickle=False) as checkpoint:
                    values=checkpoint['embeddings'];counts=checkpoint['token_lengths']
                    assert np.array_equal(checkpoint['row_ids'],np.arange(start,stop))
            else:
                if model is None:
                    model=AutoModel.from_pretrained(config['model'],revision=revision,
                        trust_remote_code=True,code_revision=code_revision,use_flash_attn=False).float().to('cuda').eval()
                    assert 'classification' in model._adaptation_map, 'В загруженной модели нет адаптера classification'
                    task_id=model._adaptation_map['classification']
                    print('Jina: classification adapter id =',task_id,flush=True)
                    model.requires_grad_(False)
                values=[];counts=[]
                for i in range(start,stop,args.batch_size):
                    texts=frame.text.iloc[i:min(i+args.batch_size,stop)].tolist()
                    counts.extend(len(ids) for ids in tokenizer(texts,add_special_tokens=True,truncation=False)['input_ids'])
                    batch=tokenizer(texts,max_length=args.max_length,padding=True,truncation=True,return_tensors='pt').to('cuda')
                    try:
                        with torch.inference_mode():
                            adapter_mask=torch.full((len(texts),),task_id,dtype=torch.int32,device='cuda')
                            hidden=model(**batch,adapter_mask=adapter_mask).last_hidden_state
                            masked=hidden.masked_fill(~batch['attention_mask'][...,None].bool(),0.)
                            pooled=masked.sum(1)/batch['attention_mask'].sum(1)[...,None]
                            values.append(torch.nn.functional.normalize(pooled,p=2,dim=1).cpu().numpy())
                    except torch.cuda.OutOfMemoryError:
                        raise RuntimeError('Не хватило GPU-памяти Jina. Уменьшите JINA_BATCH_SIZE до 2 или 4 и повторите ячейку; готовые блоки сохранены.') from None
                values=np.concatenate(values);counts=np.asarray(counts,dtype=np.int64)
                tmp=path.with_suffix('.tmp')
                with tmp.open('wb') as f:np.savez_compressed(f,embeddings=values,token_lengths=counts,row_ids=np.arange(start,stop))
                tmp.replace(path)
            assert values.shape==(stop-start,1024) and np.isfinite(values).all()
            assert np.allclose(np.linalg.norm(values,axis=1),1,atol=1e-4)
            arrays.append(values);lengths.extend(counts.tolist())
            print(f'Jina {split}: {stop}/{len(frame)}',flush=True)
        np.save(out/f'{split}_embeddings.npy',np.concatenate(arrays))
        pd.DataFrame({'row_id':frame.row_id,'tokens_with_special_tokens':lengths,'truncated':np.asarray(lengths)>args.max_length}).to_csv(out/f'{split}_token_lengths.csv',index=False)
        print(split,'текстов с усечением:',sum(n>args.max_length for n in lengths),flush=True)
    config['embedding_files']={s:sha(out/f'{s}_embeddings.npy') for s in frames}
    config['status']='complete'
    (out/'manifest.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    (root/'latest_embeddings.json').write_text(json.dumps({'directory':str(out)}),encoding='utf-8')
    print('Эмбеддинги сохранены:',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--revision',default='main')
    p.add_argument('--code-revision',default='main');p.add_argument('--max-length',type=int,default=512)
    args=p.parse_args();assert args.batch_size>0 and 2<=args.max_length<=8192;main(args)
