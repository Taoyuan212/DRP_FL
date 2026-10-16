"""PGFed Stage-7D entry runner; Train/Validation only."""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from local_baseline import make_loader,set_reproducible_seed
from run_stage7b import read_config,initial_state
from stage7b_core import load_pilot_data,clone_state,pgfed_train_client,full_gradient_and_loss_minus,train_state,validate_state,weighted_state
from stage7d_checkpoint import SCHEMA,append_rows,compatibility,config_hash,restore_rng,rng_state,verify_history
from stage7d_formal import local_reference,selection,better,resource_row,finalize,sha256
def fp(x,y,seed,b):q,t=next(iter(make_loader(x,y,b,True,seed)));return hashlib.sha256(q.numpy().tobytes()+t.numpy().tobytes()).hexdigest()
def score(rows):p=np.array([x['Pinball'] for x in rows]);return (float(p.mean()),float(np.quantile(p,.95)),float(p.max()),float(np.mean([x['quantile_crossing_rate'] for x in rows])))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/stage7b.json');ap.add_argument('--output',required=True,type=Path);ap.add_argument('--max-round',required=True,type=int);ap.add_argument('--resume-from',type=Path);a=ap.parse_args()
 cfg=read_config(ROOT/a.config);cfg['_config_path']=str(a.config);set_reproducible_seed(cfg['seed'],cfg['torch_num_threads']);method='pgfed';ids=cfg['client_ids'];n=len(ids);m=cfg['methods'][method];local_ref=local_reference(cfg);out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True);data=load_pilot_data(cfg);init=initial_state(cfg)
 if a.resume_from:
  z=torch.load(a.resume_from,map_location='cpu',weights_only=False);compatibility(z,cfg,method);states=z['states'];q=z['method_state'];global_state=q['global_state'];alpha=q['alpha'];prev_grads=q['prev_grads'];prev_loss=q['prev_loss'];optim=q['optimizer_states'];best=z['best_so_far'];restore_rng(z['rng_state']);start=z['round']+1;verify_history(out/'round_metrics.csv',z['round'],n)
  if not (out/'communication_report.csv').exists() or len(pd.read_csv(out/'communication_report.csv'))!=z['round']:raise RuntimeError('RESUME OUTPUT HARD FAIL: communication history')
  print(json.dumps({'RESUME MODE':True,'completed_round':z['round'],'next_round':start}),flush=True)
 else:
  if (out/'round_metrics.csv').exists():raise RuntimeError('FRESH OUTPUT HARD FAIL')
  global_state=clone_state(init);alpha=np.ones((n,n))/n;prev_grads=prev_loss=None;optim=[None]*n;states=[clone_state(init) for _ in ids];best=None;start=1
 for rnd in range(start,a.max_round+1):
  round_started=time.perf_counter();local=[];ng=[];nl=[];trace=[];training_records=[]
  for i,c in enumerate(ids):
   seed=cfg['seed']+rnd*100000+c;trace.append({'round':rnd,'client_id':c,'first_batch_fingerprint':fp(data[c].train_x,data[c].train_y,seed,cfg['batch_size'])})
   s,alpha[i],optim[i],record=pgfed_train_client(global_state,data[c].train_x,data[c].train_y,cfg,seed,m,alpha[i],prev_grads,prev_loss,optim[i]);g,l,grad_record=full_gradient_and_loss_minus(s,data[c].train_x,data[c].train_y,cfg);local.append(s);ng.append(g.numpy());nl.append(l);training_records.extend([record,grad_record])
  personalization_started=time.perf_counter();counts=np.array([len(data[c].train_y) for c in ids],float);global_state=weighted_state(local,counts/counts.sum());states=local;prev_grads=np.stack(ng);prev_loss=np.array(nl);personalization_seconds=time.perf_counter()-personalization_started
  rows=[{'method':method,'round':rnd,'client_id':c,**validate_state(s,data[c],np.asarray(cfg['quantiles']),cfg['batch_size'])} for c,s in zip(ids,states)];cand={'round':rnd,**selection(rows,local_ref)};best=cand if better(cand,best) else best;resource=resource_row(method,rnd,n,cfg['predictor_parameters'],round_started,training_records,personalization_seconds)
  z={'schema':SCHEMA,'method':method,'config_hash':config_hash(cfg),'seed':cfg['seed'],'client_ids':ids,'parameter_count':cfg['predictor_parameters'],'batch_size':cfg['batch_size'],'local_epochs':cfg['local_epochs'],'quantiles':cfg['quantiles'],'lookback':cfg['lookback_steps'],'horizon':cfg['horizon_minutes'],'method_config':m,'round':rnd,'states':states,'method_state':{'global_state':global_state,'alpha':alpha,'prev_grads':prev_grads,'prev_loss':prev_loss,'optimizer_states':optim},'rng_state':rng_state(),'best_so_far':best,'validation_score':cand,'resource_state':resource,'optimizer_state':'PERSISTENT_IN_METHOD_STATE','code_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'data_hash':sha256(ROOT/'results/stage2/dataset_manifest.csv'),'test_read':False}
  torch.save(z,out/f'checkpoint_round_{rnd:03d}.pt');append_rows(out/'round_metrics.csv',rows);append_rows(out/'training_trace.csv',trace);append_rows(out/'communication_report.csv',[resource]);print(f"pgfed round {rnd} macro={cand['macro_validation_pinball']:.10f}",flush=True)
 finalize(out,best,cfg,Path(__file__))
if __name__=='__main__':main()
