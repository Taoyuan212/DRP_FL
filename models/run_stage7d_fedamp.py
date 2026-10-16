"""FedAMP Stage-7D entry runner; Train/Validation only."""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from local_baseline import make_loader,set_reproducible_seed
from run_stage7b import read_config,initial_state
from stage7b_core import load_pilot_data,clone_state,heur_fedamp_weights,train_state,validate_state,weighted_state
from stage7d_checkpoint import SCHEMA,append_rows,compatibility,config_hash,restore_rng,rng_state,verify_history
from stage7d_formal import local_reference,selection,better,resource_row,finalize,sha256
def fp(x,y,seed,b):q,t=next(iter(make_loader(x,y,b,True,seed)));return hashlib.sha256(q.numpy().tobytes()+t.numpy().tobytes()).hexdigest()
def score(rows):p=np.array([x['Pinball'] for x in rows]);return (float(p.mean()),float(np.quantile(p,.95)),float(p.max()),float(np.mean([x['quantile_crossing_rate'] for x in rows])))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/stage7b.json');ap.add_argument('--output',required=True,type=Path);ap.add_argument('--max-round',required=True,type=int);ap.add_argument('--resume-from',type=Path);a=ap.parse_args()
 cfg=read_config(ROOT/a.config);cfg['_config_path']=str(a.config);set_reproducible_seed(cfg['seed'],cfg['torch_num_threads']);method='fedamp';ids=cfg['client_ids'];n=len(ids);m=cfg['methods'][method];local_ref=local_reference(cfg);out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True);data=load_pilot_data(cfg);init=initial_state(cfg)
 if a.resume_from:
  z=torch.load(a.resume_from,map_location='cpu',weights_only=False);compatibility(z,cfg,method);states=z['states'];best=z['best_so_far'];restore_rng(z['rng_state']);start=z['round']+1;verify_history(out/'round_metrics.csv',z['round'],n)
  if not (out/'communication_report.csv').exists() or len(pd.read_csv(out/'communication_report.csv'))!=z['round']:raise RuntimeError('RESUME OUTPUT HARD FAIL: communication history')
  print(json.dumps({'RESUME MODE':True,'completed_round':z['round'],'next_round':start}),flush=True)
 else:
  if (out/'round_metrics.csv').exists():raise RuntimeError('FRESH OUTPUT HARD FAIL')
  states=[clone_state(init) for _ in ids];best=None;start=1
 for rnd in range(start,a.max_round+1):
  round_started=time.perf_counter();personalization_started=time.perf_counter();w=heur_fedamp_weights(states,m['self_weight'],m['sigma']);centers=[weighted_state(states,w[i]) for i in range(n)];personalization_seconds=time.perf_counter()-personalization_started;local=[];trace=[];training_records=[];prox=m['lambda']/(2*m['alpha'])
  for i,c in enumerate(ids):
   seed=cfg['seed']+rnd*100000+c;trace.append({'round':rnd,'client_id':c,'first_batch_fingerprint':fp(data[c].train_x,data[c].train_y,seed,cfg['batch_size'])});s,record=train_state(centers[i],data[c].train_x,data[c].train_y,cfg,seed,prox_center=centers[i],prox_mu=prox);local.append(s);training_records.append(record)
  states=local;rows=[{'method':method,'round':rnd,'client_id':c,**validate_state(s,data[c],np.asarray(cfg['quantiles']),cfg['batch_size'])} for c,s in zip(ids,states)];cand={'round':rnd,**selection(rows,local_ref)};best=cand if better(cand,best) else best;resource=resource_row(method,rnd,n,cfg['predictor_parameters'],round_started,training_records,personalization_seconds)
  z={'schema':SCHEMA,'method':method,'config_hash':config_hash(cfg),'seed':cfg['seed'],'client_ids':ids,'parameter_count':cfg['predictor_parameters'],'batch_size':cfg['batch_size'],'local_epochs':cfg['local_epochs'],'quantiles':cfg['quantiles'],'lookback':cfg['lookback_steps'],'horizon':cfg['horizon_minutes'],'method_config':m,'round':rnd,'states':states,'method_state':{'attentive_weights':w,'proximal_centers':centers,'proximal_coefficient':prox},'rng_state':rng_state(),'best_so_far':best,'validation_score':cand,'resource_state':resource,'optimizer_state':'NOT_APPLICABLE: per-client Adam recreated each round','code_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'data_hash':sha256(ROOT/'results/stage2/dataset_manifest.csv'),'test_read':False}
  torch.save(z,out/f'checkpoint_round_{rnd:03d}.pt');append_rows(out/'round_metrics.csv',rows);append_rows(out/'training_trace.csv',trace);append_rows(out/'communication_report.csv',[resource]);print(f"fedamp round {rnd} macro={cand['macro_validation_pinball']:.10f}",flush=True)
 finalize(out,best,cfg,Path(__file__))
if __name__=='__main__':main()
