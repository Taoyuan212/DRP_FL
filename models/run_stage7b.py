"""Run one Stage-7B method at a time.  No code path loads Official Test."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from local_baseline import QuantileMLP, parameter_count, set_reproducible_seed
from stage7b_core import (
    apple_personalized_state, apple_train_client, clone_state, fomo_candidate_indices,
    fomo_inner_split, fomo_update, full_gradient_and_loss_minus, heur_fedamp_weights,
    load_pilot_data, pgfed_alpha_update, pfedgraph_weights, pinball_kink_audit,
    pgfed_train_client, sha256_file, state_loss, state_vector, train_state, validate_state, vector_state,
    weighted_state,
)

METHODS = ("fedfomo", "apple", "pfedgraph", "fedamp", "pgfed")
REPO_INFO = {
    "fedfomo": ("https://github.com/NVlabs/FedFomo", "external/stage7b/FedFomo", "fe04f6641407bce4fc58ea3fbf8cb314f9af8629"),
    "apple": ("https://github.com/ljaiverson/pFL-APPLE", "external/stage7b/pFL-APPLE", "8078196731b5493ab408fff4fe80f058e2e538f4"),
    "pfedgraph": ("https://github.com/MediaBrain-SJTU/pFedGraph", "external/stage7b/pFedGraph", "4f5f17c45d88745be642f834ffc60c1604f019f4"),
    "pgfed": ("https://github.com/ljaiverson/pgfed", "external/stage7b/PGFed", "b1d70f7db703c80604c7328c36fc2a581e78a289"),
}


def read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("test_read") is not False or config.get("allowed_splits") != ["train", "validation"]:
        raise RuntimeError("Stage-7B test lock is not active")
    return config


def initial_state(config: dict) -> dict[str, torch.Tensor]:
    path = ROOT / config["initial_checkpoint"]
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["state_dict"] if "state_dict" in payload else payload
    model = QuantileMLP(); model.load_state_dict(state)
    if parameter_count(model) != config["predictor_parameters"]:
        raise RuntimeError("Predictor parameter count mismatch")
    return clone_state(state)


def rss_bytes() -> int:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            counters = PROCESS_MEMORY_COUNTERS(); counters.cb = ctypes.sizeof(counters)
            get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
            get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
            get_memory.restype = wintypes.BOOL
            if not get_memory(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                raise OSError("GetProcessMemoryInfo failed")
            return int(counters.PeakWorkingSetSize)
        return 0


def add_validation(rows: list[dict], method: str, rnd: int, states: list[dict], data: dict,
                   config: dict) -> None:
    q = np.asarray(config["quantiles"])
    for cid, state in zip(config["client_ids"], states):
        metrics = validate_state(state, data[cid], q, config["batch_size"])
        rows.append({"method": method, "round": rnd, "client_id": cid, **metrics})


def communication_row(method: str, rnd: int, down: int, up: int, extra: int,
                      downloads: int, training: float, personalization: float, server: float,
                      wall: float, counters: dict, rss: int) -> dict:
    return {"method": method, "round": rnd, "server_to_client_bytes": int(down),
            "client_to_server_bytes": int(up), "extra_scalar_vector_bytes": int(extra),
            "model_downloads": int(downloads), "local_training_seconds": training,
            "extra_personalization_seconds": personalization, "server_aggregation_seconds": server,
            "wall_seconds": wall, "rss_bytes": rss, **counters}


def sum_counters(records: list[dict]) -> dict:
    return {k: int(sum(r.get(k, 0) for r in records)) for k in ("forward_passes", "backward_passes", "optimizer_steps")}


def run_fedfomo(config, data, init, out):
    n = len(config["client_ids"]); method = config["methods"]["fedfomo"]
    states = [clone_state(init) for _ in range(n)]; preference = np.zeros((n, n))
    split_rows = {}; fit_indices = {}; val_indices = {}
    for i, cid in enumerate(config["client_ids"]):
        fit, val = fomo_inner_split(data[cid], method["inner_fraction"], 7)
        fit_indices[i], val_indices[i] = fit, val
        split_rows[str(cid)] = {"train_windows": len(data[cid].train_y), "inner_fit_windows": len(fit),
            "purged_windows": len(data[cid].train_y)-len(fit)-len(val), "inner_validation_windows": len(val),
            "fit_latest_target_time": str(data[cid].train_target_time[fit].max()),
            "inner_validation_earliest_issue_time": str(data[cid].train_issue_time[val].min()),
            "purge_hours": 7}
    (out / "fedfomo_inner_split.json").write_text(json.dumps(split_rows, indent=2), encoding="utf-8")
    metrics=[]; stats=[]; comm=[]; rng=np.random.default_rng(config["seed"]); model_bytes=config["predictor_parameters"]*4
    for rnd in range(1, config["rounds"]+1):
        wall0=time.perf_counter(); peak=rss_bytes(); local=[]; rec=[]
        for i,cid in enumerate(config["client_ids"]):
            d=data[cid]; ix=fit_indices[i]
            s,r=train_state(states[i],d.train_x[ix],d.train_y[ix],config,config["seed"]+rnd*100000+cid)
            local.append(s); rec.append(r); peak=max(peak,rss_bytes())
        p0=time.perf_counter(); new=[]
        epsilon=max(method["epsilon"]-(rnd-1)*method["epsilon_decay"],0.0)
        for i,cid in enumerate(config["client_ids"]):
            candidates=fomo_candidate_indices(preference[i],i,method["candidate_count"],epsilon,rng)
            d=data[cid]; ix=val_indices[i]; q=torch.tensor(config["quantiles"])
            ref_loss=state_loss(local[i],d.train_x[ix],d.train_y[ix],q,config["batch_size"])
            losses=[state_loss(local[j],d.train_x[ix],d.train_y[ix],q,config["batch_size"]) for j in candidates]
            updated,raw=fomo_update(local[i],[local[j] for j in candidates],ref_loss,losses)
            for j,v in zip(candidates,raw): preference[i,j]+=v
            new.append(updated)
            stats.append({"round":rnd,"client_id":cid,"epsilon":epsilon,"candidate_ids":json.dumps([config['client_ids'][j] for j in candidates]),
                "reference_loss":ref_loss,"positive_sources":int((raw>0).sum()),"preference_l1":float(np.abs(preference[i]).sum()),
                "top_preference_client":config["client_ids"][int(np.argmax(preference[i]))]})
        personal=time.perf_counter()-p0; states=new
        add_validation(metrics,"fedfomo",rnd,states,data,config)
        counters=sum_counters(rec); counters["forward_passes"] += n*(method["candidate_count"]+1)
        wall=time.perf_counter()-wall0
        # Candidate id (int64) plus returned scalar weight/loss (float64).
        scalar_bytes=n*method["candidate_count"]*16
        comm.append(communication_row("fedfomo",rnd,n*method["candidate_count"]*model_bytes,n*model_bytes,scalar_bytes,
            n*method["candidate_count"],sum(x["training_seconds"] for x in rec),personal,0,wall,counters,peak))
        print(f"fedfomo round {rnd:02d}/40 validation={np.mean([x['Pinball'] for x in metrics if x['round']==rnd]):.8f}",flush=True)
    return states,metrics,stats,comm,{"preference_persistent":True,"inner_test_rows":0}


def run_apple(config,data,init,out):
    n=len(config["client_ids"]); method=config["methods"]["apple"]
    core=[clone_state(init) for _ in range(n)]; counts=np.array([len(data[c].train_y) for c in config["client_ids"]],float); p0=counts/counts.sum()
    relations=np.tile(p0,(n,1)); metrics=[]; stats=[]; comm=[]; model_bytes=config["predictor_parameters"]*4
    for rnd in range(1,config["rounds"]+1):
        wall0=time.perf_counter(); rec=[]; next_core=[]; next_rel=[]; peak=rss_bytes()
        for i,cid in enumerate(config["client_ids"]):
            s,p,r=apple_train_client(core,relations[i],i,p0,data[cid],config,method,rnd-1)
            next_core.append(s); next_rel.append(p); rec.append(r); peak=max(peak,rss_bytes())
        core=next_core; relations=np.stack(next_rel)
        p0t=time.perf_counter(); states=[apple_personalized_state(core,relations[i]) for i in range(n)]; personal=time.perf_counter()-p0t
        add_validation(metrics,"apple",rnd,states,data,config)
        for i,cid in enumerate(config["client_ids"]):
            absw=np.abs(relations[i]); prob=absw/max(absw.sum(),1e-12)
            stats.append({"round":rnd,"client_id":cid,"self_weight":relations[i,i],"entropy_abs":float(-(prob*np.log(prob+1e-12)).sum()),
                "top_collaborator":config["client_ids"][int(np.argmax(absw))],"weight_min":relations[i].min(),"weight_max":relations[i].max(),
                "one_hot":bool(absw.max()/max(absw.sum(),1e-12)>0.999)})
        wall=time.perf_counter()-wall0; counters=sum_counters(rec)
        comm.append(communication_row("apple",rnd,n*n*model_bytes,n*model_bytes,n*n*4,n*n,
            sum(x["training_seconds"] for x in rec),personal,0,wall,counters,peak))
        print(f"apple round {rnd:02d}/40 validation={np.mean([x['Pinball'] for x in metrics if x['round']==rnd]):.8f}",flush=True)
    return states,metrics,stats,comm,{"relation_one_hot_count_final":int(sum(r["one_hot"] for r in stats if r["round"]==config["rounds"]))}


def run_pfedgraph(config,data,init,out):
    n=len(config["client_ids"]); method=config["methods"]["pfedgraph"]
    states=[clone_state(init) for _ in range(n)]; counts=np.array([len(data[c].train_y) for c in config["client_ids"]],float); prior=counts/counts.sum()
    metrics=[]; stats=[]; comm=[]; model_bytes=config["predictor_parameters"]*4; centers=None
    for rnd in range(1,config["rounds"]+1):
        wall0=time.perf_counter(); rec=[]; local=[]; peak=rss_bytes()
        for i,cid in enumerate(config["client_ids"]):
            cosine_center=None if centers is None else centers[i]
            s,r=train_state(states[i],data[cid].train_x,data[cid].train_y,config,config["seed"]+rnd*100000+cid,
                cosine_center=cosine_center,cosine_lambda=method["local_cosine_lambda"] if rnd>1 else 0)
            local.append(s);rec.append(r);peak=max(peak,rss_bytes())
        p0=time.perf_counter(); w,cos=pfedgraph_weights(local,init,prior,method["graph_lambda"])
        states=[weighted_state(local,w[i]) for i in range(n)]
        vec=np.stack([state_vector(s).double().numpy() for s in local]); norms=np.linalg.norm(vec,axis=1)
        centers=[torch.tensor(sum(w[i,j]*vec[j]/max(norms[j],1e-12) for j in range(n)),dtype=torch.float32) for i in range(n)]
        server=time.perf_counter()-p0
        add_validation(metrics,"pfedgraph",rnd,states,data,config)
        for i,cid in enumerate(config["client_ids"]):
            obj=method["graph_lambda"]*float(np.sum((w[i]-prior)**2))-float(np.dot(cos[i],w[i]))
            initobj=-float(np.dot(cos[i],prior))
            stats.append({"round":rnd,"client_id":cid,"weight_sum":w[i].sum(),"weight_min":w[i].min(),"simplex_error":abs(w[i].sum()-1),
                "objective":obj,"initial_objective":initobj,"objective_improved":obj<=initobj+1e-10,"self_weight":w[i,i]})
        wall=time.perf_counter()-wall0
        comm.append(communication_row("pfedgraph",rnd,n*model_bytes,n*model_bytes,n*n*4,n,
            sum(x["training_seconds"] for x in rec),0,server,wall,sum_counters(rec),peak))
        print(f"pfedgraph round {rnd:02d}/40 validation={np.mean([x['Pinball'] for x in metrics if x['round']==rnd]):.8f}",flush=True)
    return states,metrics,stats,comm,{"max_simplex_error":max(x["simplex_error"] for x in stats),"all_objectives_improved":all(x["objective_improved"] for x in stats)}


def run_fedamp(config,data,init,out):
    n=len(config["client_ids"]); method=config["methods"]["fedamp"]; states=[clone_state(init) for _ in range(n)]
    metrics=[];stats=[];comm=[];model_bytes=config["predictor_parameters"]*4
    for rnd in range(1,config["rounds"]+1):
        wall0=time.perf_counter(); p0=time.perf_counter(); w=heur_fedamp_weights(states,method["self_weight"],method["sigma"])
        centers=[weighted_state(states,w[i]) for i in range(n)]; server=time.perf_counter()-p0; local=[];rec=[];peak=rss_bytes()
        prox_mu=method["lambda"]/(2*method["alpha"])
        for i,cid in enumerate(config["client_ids"]):
            s,r=train_state(centers[i],data[cid].train_x,data[cid].train_y,config,config["seed"]+rnd*100000+cid,
                prox_center=centers[i],prox_mu=prox_mu)
            local.append(s);rec.append(r);peak=max(peak,rss_bytes())
        states=local; add_validation(metrics,"fedamp",rnd,states,data,config)
        for i,cid in enumerate(config["client_ids"]):
            stats.append({"round":rnd,"client_id":cid,"self_weight":w[i,i],"weight_sum":w[i].sum(),"weight_min":w[i].min(),"weight_max":w[i].max(),
                "top_external_client":config["client_ids"][int(np.argmax(np.where(np.arange(n)==i,-np.inf,w[i])))]})
        wall=time.perf_counter()-wall0
        comm.append(communication_row("fedamp",rnd,n*model_bytes,n*model_bytes,n*n*4,n,
            sum(x["training_seconds"] for x in rec),0,server,wall,sum_counters(rec),peak))
        print(f"fedamp round {rnd:02d}/40 validation={np.mean([x['Pinball'] for x in metrics if x['round']==rnd]):.8f}",flush=True)
    return states,metrics,stats,comm,{"variant":"HeurFedAMP","equation_toy_checked":True}


def run_pgfed(config,data,init,out):
    n=len(config["client_ids"]); method=config["methods"]["pgfed"]; global_state=clone_state(init); alpha=np.ones((n,n))/n
    prev_grads=None; prev_loss=None; optimizer_states=[None]*n; metrics=[];stats=[];comm=[];model_bytes=config["predictor_parameters"]*4; states=[clone_state(init) for _ in range(n)]
    for rnd in range(1,config["rounds"]+1):
        wall0=time.perf_counter(); local=[];rec=[];new_grads=[];new_loss=[];peak=rss_bytes()
        for i,cid in enumerate(config["client_ids"]):
            s,alpha[i],optimizer_states[i],r=pgfed_train_client(global_state,data[cid].train_x,data[cid].train_y,config,
                config["seed"]+rnd*100000+cid,method,alpha[i],prev_grads,prev_loss,optimizer_states[i])
            grad,lminus,gr=full_gradient_and_loss_minus(s,data[cid].train_x,data[cid].train_y,config)
            for k in gr:r[k]=r.get(k,0)+gr[k]
            local.append(s);rec.append(r);new_grads.append(grad.numpy());new_loss.append(lminus);peak=max(peak,rss_bytes())
        p0=time.perf_counter(); counts=np.array([len(data[c].train_y) for c in config["client_ids"]],float); global_state=weighted_state(local,counts/counts.sum());server=time.perf_counter()-p0
        states=local;prev_grads=np.stack(new_grads);prev_loss=np.array(new_loss);add_validation(metrics,"pgfed",rnd,states,data,config)
        for i,cid in enumerate(config["client_ids"]):
            stats.append({"round":rnd,"client_id":cid,"alpha_sum":alpha[i].sum(),"alpha_min":alpha[i].min(),"alpha_max":alpha[i].max(),
                "nonlocal_gradient_norm":rec[i]["nonlocal_gradient_norm"]})
        # PGFed uploads one latest gradient/client; server sends one combined
        # gradient and one mean gradient/client, plus the risk scalars.  Source
        # gradients are combined on the server, not all downloaded by targets.
        wall=time.perf_counter()-wall0;extra_bytes=3*n*model_bytes+(n*n+n)*4
        comm.append(communication_row("pgfed",rnd,n*model_bytes,n*model_bytes,extra_bytes,n,
            sum(x["training_seconds"] for x in rec),0,server,wall,sum_counters(rec),peak))
        print(f"pgfed round {rnd:02d}/40 validation={np.mean([x['Pinball'] for x in metrics if x['round']==rnd]):.8f}",flush=True)
    return states,metrics,stats,comm,{"pinball_kink_gradients":pinball_kink_audit(config["quantiles"]),"smoothed_pinball":False}


RUNNERS={"fedfomo":run_fedfomo,"apple":run_apple,"pfedgraph":run_pfedgraph,"fedamp":run_fedamp,"pgfed":run_pgfed}


def write_provenance(method,out,config):
    payload={"method":method,"run_date":time.strftime("%Y-%m-%d"),"python":sys.version,"torch":torch.__version__,
        "platform":platform.platform(),"test_read":False,"allowed_splits":["train","validation"],
        "adaptation_files":["scripts/stage7b_core.py","scripts/run_stage7b.py"]}
    if method in REPO_INFO:
        url,path,commit=REPO_INFO[method];payload.update({"repository_url":url,"snapshot_path":path,"commit":commit,"official_snapshot_modified":False})
    else:
        payload.update({"paper_url":"https://ojs.aaai.org/index.php/AAAI/article/view/16960","official_repository":"not confirmed"})
    (out/"code_provenance.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",default="configs/stage7b.json");parser.add_argument("--method",required=True,choices=METHODS);args=parser.parse_args()
    config=read_config(ROOT/args.config);set_reproducible_seed(config["seed"],config["torch_num_threads"])
    out=ROOT/config["output_root"]/args.method;out.mkdir(parents=True,exist_ok=True)
    (out/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8");write_provenance(args.method,out,config)
    init=initial_state(config);data=load_pilot_data(config)
    checkpoint_hash=sha256_file(ROOT/config["initial_checkpoint"])
    audit=[{"method":args.method,"client_id":cid,"predictor_sha256":checkpoint_hash,"parameter_count":len(state_vector(init)),"match":True} for cid in config["client_ids"]]
    pd.DataFrame(audit).to_csv(out/"baseline_initialization_audit.csv",index=False)
    start=time.perf_counter();states,metrics,stats,comm,special=RUNNERS[args.method](config,data,init,out);runtime=time.perf_counter()-start
    pd.DataFrame(metrics).to_csv(out/"round_metrics.csv",index=False);pd.DataFrame(stats).to_csv(out/"personalization_statistics.csv",index=False)
    pd.DataFrame(comm).to_csv(out/"communication_report.csv",index=False)
    ckpt={"method":args.method,"round":config["rounds"],"client_ids":config["client_ids"],"states":states,"special_state":special}
    torch.save(ckpt,out/"round_040_checkpoint.pt")
    loaded=torch.load(out/"round_040_checkpoint.pt",map_location="cpu",weights_only=False)
    roundtrip=max(float(torch.max(torch.abs(state_vector(a)-state_vector(b)))) for a,b in zip(states,loaded["states"]))
    report={"method":args.method,"total_runtime_seconds":runtime,"cpu_peak_rss_bytes":max(x["rss_bytes"] for x in comm),
        "gpu_peak_memory_bytes":0,"communication_bytes":int(sum(x["server_to_client_bytes"]+x["client_to_server_bytes"]+x["extra_scalar_vector_bytes"] for x in comm)),
        "forward_passes":sum(x["forward_passes"] for x in comm),"backward_passes":sum(x["backward_passes"] for x in comm),
        "optimizer_steps":sum(x["optimizer_steps"] for x in comm),"checkpoint_roundtrip_max_error":roundtrip,"test_rows_read":0,**special}
    (out/"resource_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)

if __name__=="__main__":main()
