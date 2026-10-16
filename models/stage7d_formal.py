"""Non-algorithmic audit, selection, and resource helpers for Stage 7D."""
from __future__ import annotations
import hashlib, json, shutil, time
from pathlib import Path
import numpy as np
import pandas as pd
from run_stage7b import rss_bytes, sum_counters

ROOT = Path(__file__).resolve().parents[1]
MODEL_BYTES = 13833 * 4

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def local_reference(config: dict) -> dict[int,float]:
    d=pd.read_csv(ROOT/config['local_validation_reference'])
    if len(d)!=len(config['client_ids']) or set(d.client_id)!=set(config['client_ids']):
        raise RuntimeError('LOCAL VALIDATION REFERENCE HARD FAIL')
    return dict(zip(d.client_id.astype(int),d.local_validation_pinball.astype(float)))

def selection(rows: list[dict], local: dict[int,float]) -> dict:
    p=np.asarray([r['Pinball'] for r in rows],float)
    crossing=float(np.mean([r['quantile_crossing_rate'] for r in rows]))
    rel=np.asarray([(r['Pinball']-local[int(r['client_id'])])/local[int(r['client_id'])] for r in rows])
    metrics={'macro_validation_pinball':float(p.mean()),'quantile_crossing_rate':crossing,
      'p95_validation_pinball':float(np.quantile(p,.95)),'worst_client_relative_degradation_vs_local':float(rel.max())}
    metrics['selection_key']=[metrics[k] for k in ('macro_validation_pinball','quantile_crossing_rate','p95_validation_pinball','worst_client_relative_degradation_vs_local')]
    return metrics

def better(candidate: dict, best: dict|None) -> bool:
    return best is None or (*candidate['selection_key'],candidate['round']) < (*best['selection_key'],best['round'])

def communication(method: str,n: int,p: int,candidate_count: int=0) -> tuple[int,int,int,int]:
    mb=p*4
    if method=='fedfomo': return n*candidate_count*mb,n*mb,n*candidate_count*4,n*candidate_count
    if method in ('pfedgraph','fedamp'): return n*mb,n*mb,n*n*4,n
    if method=='pgfed': return n*mb,n*mb,n*(p*4+8),n
    raise ValueError(method)

def resource_row(method: str,rnd: int,n:int,p:int,started:float,records:list[dict],personalization_seconds:float,candidate_count:int=0)->dict:
    down,up,extra,downloads=communication(method,n,p,candidate_count)
    counters=sum_counters(records)
    return {'method':method,'round':rnd,'server_to_client_bytes':down,'client_to_server_bytes':up,
      'extra_scalar_vector_bytes':extra,'model_downloads':downloads,
      'local_training_seconds':float(sum(x.get('training_seconds',0) for x in records)),
      'extra_personalization_seconds':float(personalization_seconds),'wall_seconds':float(time.perf_counter()-started),
      'rss_bytes':rss_bytes(),**counters}

def provenance(config:dict,runner:Path)->dict:
    paths={'config':ROOT/config['_config_path'],'runner':runner,'initial_checkpoint':ROOT/config['initial_checkpoint'],
      'stage2_dataset_manifest':ROOT/'results/stage2/dataset_manifest.csv','stage2_scaler':ROOT/config['processed_dir']/'scaler.json',
      'stage2_split':ROOT/'results/stage2/split_definition.json','stage2_normalization':ROOT/'results/stage2/feature_set_v1.json'}
    return {k:{'path':str(v),'sha256':sha256(v)} for k,v in paths.items()}

def finalize(out:Path,best:dict,config:dict,runner:Path)->dict:
    selected=out/f"checkpoint_round_{best['round']:03d}.pt"
    if not selected.exists(): raise RuntimeError('SELECTED CHECKPOINT MISSING')
    target=out/'selected_checkpoint.pt';shutil.copy2(selected,target)
    comm=pd.read_csv(out/'communication_report.csv')
    summary={'status':'VALIDATION_COMPLETE','seed':config['seed'],'clients':len(config['client_ids']),'rounds':config['rounds'],
      'selected_round':best['round'],'selection_metrics':{k:v for k,v in best.items() if k not in ('selection_key','round')},
      'selected_checkpoint_sha256':sha256(target),'config_sha256':sha256(ROOT/config['_config_path']),
      'code_sha256':sha256(runner),'data_manifest_sha256':sha256(ROOT/'results/stage2/dataset_manifest.csv'),
      'normalization_sha256':sha256(ROOT/config['processed_dir']/'scaler.json'),'test_read':False,'test_rows_read':0,
      'resources':{'total_runtime_seconds':float(comm.wall_seconds.sum()),'communication_bytes':int((comm.server_to_client_bytes+comm.client_to_server_bytes+comm.extra_scalar_vector_bytes).sum()),
       'peak_rss_bytes':int(comm.rss_bytes.max()),'forward_passes':int(comm.forward_passes.sum()),'backward_passes':int(comm.backward_passes.sum()),'optimizer_steps':int(comm.optimizer_steps.sum())},
      'provenance':provenance(config,runner)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    return summary
