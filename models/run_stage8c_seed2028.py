"""Formal Stage-8C Seed-2028 Train/Validation-only runners with strict checkpoints."""
from __future__ import annotations
import argparse,copy,ctypes,gc,hashlib,json,math,os,random,shutil,sys,time
from collections import OrderedDict
from pathlib import Path
import numpy as np,pandas as pd,torch
from numpy.lib.stride_tricks import sliding_window_view

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
from local_baseline import QuantileMLP,forecast_metrics,make_loader,pinball_loss,predict,set_reproducible_seed
from stage7b_core import clone_state,fomo_candidate_indices,fomo_update,pfedgraph_weights,state_loss,state_vector,train_state,weighted_state
from drp_v1 import ema_relation,personalized_vectors,relation_weights,vector_to_state

BASE=ROOT/"results"/"stage8c"/"seed_2028";DATA=ROOT/"data"/"aemo"/"stage8c_seed2028";S8B=ROOT/"results"/"stage8b"
Q=np.arange(.1,1,.1);A=pd.Timestamp("2025-08-13");STEP=pd.Timedelta(minutes=5);PARAMS=13833;BATCH=2048

def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def hash_array(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def hash_states(states):
 h=hashlib.sha256()
 for s in states:
  for k,v in s.items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def rng_state():return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state()}
def restore_rng(z):random.setstate(z["python"]);np.random.set_state(z["numpy"]);torch.set_rng_state(z["torch"])
def rss():
 class PMC(ctypes.Structure):_fields_=[("cb",ctypes.c_ulong),("PageFaultCount",ctypes.c_ulong),("PeakWorkingSetSize",ctypes.c_size_t),("WorkingSetSize",ctypes.c_size_t),("QuotaPeakPagedPoolUsage",ctypes.c_size_t),("QuotaPagedPoolUsage",ctypes.c_size_t),("QuotaPeakNonPagedPoolUsage",ctypes.c_size_t),("QuotaNonPagedPoolUsage",ctypes.c_size_t),("PagefileUsage",ctypes.c_size_t),("PeakPagefileUsage",ctypes.c_size_t)]
 x=PMC();x.cb=ctypes.sizeof(x);ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(x),x.cb);return int(x.WorkingSetSize)
def time_features(origins):
 t=pd.DatetimeIndex(A+origins*STEP);hour=t.hour.to_numpy()+t.minute.to_numpy()/60;days=365+t.is_leap_year.astype(int);doy=t.dayofyear.to_numpy()-1+hour/24
 return np.column_stack([np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),np.sin(2*np.pi*doy/days),np.cos(2*np.pi*doy/days)]).astype(np.float32)

class Client:
 def __init__(self,cid,order):
  self.client_id=cid;self.order=order;d=DATA/cid;self.series=np.load(d/"series_train_validation.npy",mmap_mode="r");self.tr=np.load(d/"train_origins.npy");self.va=np.load(d/"validation_origins.npy");self._val=None;self._inner=None
 def arrays(self,origins):
  x=np.concatenate([sliding_window_view(self.series,72)[origins-72],time_features(origins)],axis=1).astype(np.float32);y=np.asarray(self.series[origins+12],dtype=np.float32);return x,y
 def train(self):return self.arrays(self.tr)
 def val(self):
  if self._val is None:self._val=self.arrays(self.va)
  return self._val
 def fomo_fit(self):
  n=max(1,int(math.ceil(len(self.tr)*.1)));start=self.tr[-n];fit=self.tr[self.tr+12<=start-84];return self.arrays(fit)
 def inner(self):
  if self._inner is None:
   n=max(1,int(math.ceil(len(self.tr)*.1)));self._inner=self.arrays(self.tr[-n:])
  return self._inner

def model_state():return torch.load(BASE/"initial_model_seed_2028.pt",map_location="cpu",weights_only=True)
def predict_state(state,x):m=QuantileMLP();m.load_state_dict(state);return predict(m,x,BATCH)[0]
def metric_row(method,rnd,c,state,data):
 x,y=data.val();p=predict_state(state,x);m=forecast_metrics(y,p,Q,1.0)
 return {"method":method,"round":rnd,"client_id":c.client_id,**m,"prediction_sha256":hash_array(p)}
def aggregate(rows,local_ref):
 p=np.array([r["Pinball"] for r in rows]);rel=np.array([(r["Pinball"]-local_ref[r["client_id"]])/local_ref[r["client_id"]] for r in rows])
 out={"macro_validation_pinball":float(p.mean()),"macro_crps_approx":float(np.mean([r["CRPS_approx"] for r in rows])),"macro_picp80":float(np.mean([r["PICP_80"] for r in rows])),"macro_pinaw80":float(np.mean([r["PINAW_80"] for r in rows])),"macro_q50_mae":float(np.mean([r["MAE"] for r in rows])),"macro_q50_rmse":float(np.mean([r["RMSE"] for r in rows])),"crossing_rate":float(np.mean([r["quantile_crossing_rate"] for r in rows])),"p95_client_pinball":float(np.quantile(p,.95)),"worst_client_pinball":float(p.max()),"worst_client_id":rows[int(np.argmax(p))]["client_id"],"worst_client_relative_degradation_vs_local":float(rel.max())}
 out["selection_key"]=[out[k] for k in ["macro_validation_pinball","crossing_rate","p95_client_pinball","worst_client_relative_degradation_vs_local"]];return out
def better(c,b):return b is None or (*c["selection_key"],c["round"])<(*b["selection_key"],b["round"])
def append(path,rows):
 d=pd.DataFrame(rows);d.to_csv(path,mode="a",header=not path.exists(),index=False)
def first_fp(x,y,seed):
 bx,by=next(iter(make_loader(x,y,BATCH,True,seed)));return hashlib.sha256(bx.numpy().tobytes()+by.numpy().tobytes()).hexdigest()

def run_local(clients,out):
 out.mkdir(parents=True,exist_ok=True);summary=out/"client_metrics.csv";resource=out/"resource.csv";resume=[]
 done=set(pd.read_csv(summary).client_id.astype(str)) if summary.exists() else set();init=model_state();q=torch.tensor(Q,dtype=torch.float32)
 for c in clients:
  if c.client_id in done:resume.append(c.client_id);continue
  started=time.perf_counter();x,y=c.train();vx,vy=c.val();m=QuantileMLP();m.load_state_dict(init);opt=torch.optim.Adam(m.parameters(),lr=.001);best=None;best_state=None;wait=0;fp=bp=steps=0;history=[]
  for epoch in range(1,51):
   m.train();total=seen=0
   for bx,by in make_loader(x,y,BATCH,True,2028+c.order*1000+epoch):
    opt.zero_grad(set_to_none=True);loss=pinball_loss(m(bx),by,q);fp+=1
    if not torch.isfinite(loss):raise FloatingPointError(f"Local NaN {c.client_id}")
    loss.backward();bp+=1;opt.step();steps+=1;total+=float(loss.detach())*len(by);seen+=len(by)
   pred,_=predict(m,vx,BATCH);vm=forecast_metrics(vy,pred,Q,1.0);history.append({"epoch":epoch,"train_pinball":total/seen,"validation_pinball":vm["Pinball"]})
   if best is None or vm["Pinball"]<best-1e-5:best=vm["Pinball"];best_state=clone_state(m.state_dict());best_epoch=epoch;wait=0
   else:wait+=1
   if wait>=6:break
  pred=predict_state(best_state,vx);metrics=forecast_metrics(vy,pred,Q,1.0);ck=out/f"client_{c.client_id}_best.pt";torch.save({"client_id":c.client_id,"best_epoch":best_epoch,"state":best_state,"metrics":metrics,"initialization_hash":hash_states([init]),"test_read":False},ck)
  append(summary,[{"method":"Local","client_id":c.client_id,"best_epoch":best_epoch,"epochs_run":epoch,**metrics,"prediction_sha256":hash_array(pred),"checkpoint_sha256":sha(ck)}]);append(resource,[{"method":"Local","client_id":c.client_id,"wall_seconds":time.perf_counter()-started,"peak_rss_bytes":rss(),"forward_passes":fp,"backward_passes":bp,"optimizer_steps":steps,"communication_bytes":0}]);(out/f"client_{c.client_id}_history.json").write_text(json.dumps(history,indent=2)+"\n")
  print(json.dumps({"method":"Local","client":c.client_id,"best_epoch":best_epoch,"pinball":metrics["Pinball"]}),flush=True);del x,y,vx,vy,m,opt,pred;gc.collect()
 (out/"resume_events.json").write_text(json.dumps({"event":"RESUMED_EXISTING_CLIENTS" if resume else "NO RESUME EVENT","clients":resume},indent=2)+"\n")

def run_federated(method,clients,out,max_round):
 out.mkdir(parents=True,exist_ok=True);metrics_file=out/"client_metrics.csv";round_file=out/"round_metrics.csv";resource_file=out/"resource.csv";init=model_state();ids=[c.client_id for c in clients];n=len(ids);counts=np.array([len(c.tr) for c in clients],float);prior=counts/counts.sum();local_ref=dict(zip(pd.read_csv(BASE/"local"/"client_metrics.csv").client_id.astype(str),pd.read_csv(BASE/"local"/"client_metrics.csv").Pinball))
 complete=[]
 for r in range(1,41):
  if (out/f"round_{r:03d}.complete.json").exists():complete.append(r)
  else:break
 start=(complete[-1]+1) if complete else 1;resume_event="NO RESUME EVENT"
 if start>1:
  z=torch.load(out/f"checkpoint_round_{start-1:03d}.pt",map_location="cpu",weights_only=False);state=z["state"];restore_rng(z["rng_state"]);best=z["best"];resume_event=f"RESUME FROM ROUND {start}"
 else:
  best={}
  if method=="fedavg_drp":state={"anchor":clone_state(init),"ema":None,"drp_states":[clone_state(init) for _ in clients]}
  elif method=="fedprox":state={"anchor":clone_state(init)}
  elif method=="fedfomo":state={"states":[clone_state(init) for _ in clients],"preference":np.zeros((n,n)),"candidate_rng":np.random.default_rng(2028).bit_generator.state}
  else:state={"states":[clone_state(init) for _ in clients],"centers":None}
 for rnd in range(start,max_round+1):
  started=time.perf_counter();training=[];fps=[];local=[];relation_seconds=0;round_aggs=[]
  if method in ["fedavg_drp","fedprox"]:starts=[state["anchor"]]*n
  else:starts=state["states"]
  for i,c in enumerate(clients):
   x,y=c.fomo_fit() if method=="fedfomo" else c.train();seed=2028+rnd*100000+i;fps.append(first_fp(x,y,seed))
   kw={}
   if method=="fedprox":kw={"prox_center":state["anchor"],"prox_mu":.01}
   if method=="pfedgraph" and state["centers"] is not None:kw={"cosine_center":state["centers"][i],"cosine_lambda":.01}
   s,tr=train_state(starts[i],x,y,{"learning_rate":.001,"batch_size":BATCH,"quantiles":Q.tolist()},seed,**kw);local.append(s);training.append(tr);del x,y;gc.collect()
  rows=[];diag={}
  if method=="fedavg_drp":
   anchor_before=state["anchor"];base=state_vector(anchor_before).double().numpy();v=np.stack([state_vector(s).double().numpy() for s in local]);u=v-base;norm=np.linalg.norm(u,axis=1,keepdims=True);un=np.divide(u,norm,out=np.zeros_like(u),where=norm>0);cos=un@un.T
   state["anchor"]=weighted_state(local,prior);t=time.perf_counter();state["ema"]=ema_relation(state["ema"],cos,.5);w=relation_weights(state["ema"],prior,.2);pv=personalized_vectors(v,w);state["drp_states"]=[vector_to_state(x,local[0]) for x in pv];relation_seconds=time.perf_counter()-t
   fed=[metric_row("FedAvg",rnd,c,state["anchor"],c) for c in clients];drp=[metric_row("DRP-FL V1",rnd,c,s,c) for c,s in zip(clients,state["drp_states"])];rows=fed+drp
   diag={"anchor_hash":hash_states([state["anchor"]]),"fedavg_local_update_hash":hash_states(local),"drp_local_update_hash":hash_states(local),"client_order":ids,"first_batch_fingerprints":fps,"relation_sha256":hash_array(state["ema"]),"relation_overhead_seconds":relation_seconds}
   for name,rr in [("FedAvg",fed),("DRP-FL V1",drp)]:
     ag=aggregate(rr,local_ref);ag.update(method=name,round=rnd);best[name]=ag if better(ag,best.get(name)) else best.get(name)
     round_aggs.append(ag)
  elif method=="fedprox":
   state["anchor"]=weighted_state(local,prior);rows=[metric_row("FedProx",rnd,c,state["anchor"],c) for c in clients];ag=aggregate(rows,local_ref);ag.update(method="FedProx",round=rnd);round_aggs.append(ag);best["FedProx"]=ag if better(ag,best.get("FedProx")) else best.get("FedProx");diag={"mu":.01}
  elif method=="fedfomo":
   t=time.perf_counter();gen=np.random.default_rng();gen.bit_generator.state=state["candidate_rng"];new=[];selected=[];eps=max(.3-(rnd-1)*.05,0)
   for i,c in enumerate(clients):
    cand=fomo_candidate_indices(state["preference"][i],i,5,eps,gen);ix,iy=c.inner();ref=state_loss(local[i],ix,iy,torch.tensor(Q),BATCH);losses=[state_loss(local[j],ix,iy,torch.tensor(Q),BATCH) for j in cand];s,raw=fomo_update(local[i],[local[j] for j in cand],ref,losses)
    for j,z in zip(cand,raw):state["preference"][i,j]+=z
    new.append(s);selected.append([ids[j] for j in cand])
   state["states"]=new;state["candidate_rng"]=gen.bit_generator.state;relation_seconds=time.perf_counter()-t;rows=[metric_row("FedFomo",rnd,c,s,c) for c,s in zip(clients,new)];ag=aggregate(rows,local_ref);ag.update(method="FedFomo",round=rnd);round_aggs.append(ag);best["FedFomo"]=ag if better(ag,best.get("FedFomo")) else best.get("FedFomo");diag={"candidate_lists":selected,"epsilon":eps,"preference_hash":hash_array(state["preference"]),"candidate_rng":gen.bit_generator.state}
  else:
   t=time.perf_counter();w,cos=pfedgraph_weights(local,init,prior,.8);state["states"]=[weighted_state(local,w[i]) for i in range(n)];v=np.stack([state_vector(s).double().numpy() for s in local]);norm=np.linalg.norm(v,axis=1);state["centers"]=[torch.tensor(sum(w[i,j]*v[j]/max(norm[j],1e-12) for j in range(n)),dtype=torch.float32) for i in range(n)];relation_seconds=time.perf_counter()-t
   rows=[metric_row("pFedGraph",rnd,c,s,c) for c,s in zip(clients,state["states"])];ag=aggregate(rows,local_ref);ag.update(method="pFedGraph",round=rnd);round_aggs.append(ag);best["pFedGraph"]=ag if better(ag,best.get("pFedGraph")) else best.get("pFedGraph");diag={"min_weight":float(w.min()),"max_weight":float(w.max()),"row_sum_max_error":float(np.max(np.abs(w.sum(1)-1))),"graph_objective_proxy":float(.8*np.sum((w-prior[None,:])**2)-np.sum(cos*w))}
  elapsed=time.perf_counter()-started;forward=sum(x["forward_passes"] for x in training);backward=sum(x["backward_passes"] for x in training);steps=sum(x["optimizer_steps"] for x in training)
  if method=="fedfomo":comm=87*5*PARAMS*4+87*PARAMS*4+87*5*4
  elif method=="pfedgraph":comm=87*PARAMS*4*2+87*87*4
  else:comm=87*PARAMS*4*2
  res={"method":method,"round":rnd,"wall_seconds":elapsed,"local_training_seconds":sum(x["training_seconds"] for x in training),"relation_personalization_seconds":relation_seconds,"communication_bytes":comm,"peak_rss_bytes":rss(),"forward_passes":forward,"backward_passes":backward,"optimizer_steps":steps}
  payload={"schema":"stage8c-seed2028-v1","method":method,"seed":2028,"round":rnd,"client_ids":ids,"state":state,"best":best,"rng_state":rng_state(),"diagnostics":diag,"resource":res,"initialization_hash":hash_states([init]),"manifest_hash":sha(S8B/"external_main_client_manifest.csv"),"data_manifest_hash":sha(DATA/"data_manifest.csv"),"test_read":False,"test_rows_read":0}
  ck=out/f"checkpoint_round_{rnd:03d}.pt";torch.save(payload,ck);append(metrics_file,rows);append(round_file,[{k:v for k,v in x.items() if k!="selection_key"} for x in round_aggs]);append(resource_file,[res]);marker={"round":rnd,"clients_complete":87,"validation_rows":len(rows),"checkpoint_sha256":sha(ck),"resource_row":True,"rng_state":True,"method_state":True,"complete":True};(out/f"round_{rnd:03d}.complete.json").write_text(json.dumps(marker,indent=2)+"\n")
  print(json.dumps({"method":method,"round":rnd,"wall_seconds":elapsed,"best":{k:v["round"] for k,v in best.items()}}),flush=True)
 (out/"resume_events.json").write_text(json.dumps({"event":resume_event},indent=2)+"\n")
 for name,b in best.items():
  src=out/f"checkpoint_round_{b['round']:03d}.pt";shutil.copy2(src,out/f"selected_{name.replace(' ','_').replace('-','_')}.pt")
 (out/"summary.json").write_text(json.dumps({"status":"VALIDATION_COMPLETE","method":method,"rounds":40,"best":best,"test_read":False,"test_rows_read":0},indent=2)+"\n")

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--method",choices=["local","fedavg_drp","fedprox","fedfomo","pfedgraph"],required=True);ap.add_argument("--max-round",type=int,default=40);ap.add_argument("--local-client-orders");ap.add_argument("--local-output");a=ap.parse_args()
 set_reproducible_seed(2028,4);manifest=pd.read_csv(S8B/"external_main_client_manifest.csv").sort_values("client_order");clients=[Client(r.client_id,int(r.client_order)) for r in manifest.itertuples()]
 if a.method=="local":
  if a.local_client_orders:
   wanted={int(x) for x in a.local_client_orders.split(",")};clients=[c for c in clients if c.order in wanted]
  run_local(clients,BASE/(a.local_output or "local"))
 else:
  if not (BASE/"local"/"client_metrics.csv").exists() or len(pd.read_csv(BASE/"local"/"client_metrics.csv"))!=87:raise RuntimeError("LOCAL REFERENCE NOT FROZEN")
  run_federated(a.method,clients,BASE/a.method,a.max_round)
if __name__=="__main__":main()
