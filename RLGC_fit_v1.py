"""RLGC nonlinear batch model search, using externally defined pandas DataFrame df.
All physical quantities are SI: F, S, Ohm, H, Hz.
Requires: numpy, pandas, scipy, torch, joblib.
Usage: exec(open('rlgc_exhaustive_autofit.py', encoding='utf-8').read()); results = run_rlgc_search(df)
"""
import ast
import re
import time
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
from scipy.optimize import minimize
from sklearn.model_selection import GroupShuffleSplit

CFG = dict(
    freq_hz=1e4,
    c_keysight='C_keysight', g_keysight='G_keysight',
    c_v93k='C_v93k', g_v93k='G_v93k',
    high='high_pins', low='low_pins',
    out_dir='rlgc_autofit',
    seed=42, test_fraction=0.25,
    # QUICK mode evaluates many architectures; finalist training is deeper.
    screen_maxiter=35, final_maxiter=250, refit_maxiter=400,
    top_k=16, folds=3, screen_starts=1, final_starts=3,
    # Optional: constrain explored candidates for faster iteration.
    families=('pair', 'pair_smu', 'pin', 'pin_smu'),
    aggregations=('sum','mean','parallel','series','sqrt','partial','maxpin'),
    t_aggregations=('count','y0_weighted'),
    # Both ways to define branch electrical length
    t_models=('rlgc','independent'),
    # Fit all on same initial grouped split. Finalists use repeated grouped splits.
    cv_group='configuration', # or 'smu_signature'
    # numerical residual normalization in SI
    c_scale_f=5e-13, g_scale_s=1e-9,
    # log-domain starting physical values; not converted units
    r_init_ohm=1e6, l_init_h=1e-4, g_init_s=1e-8, c_init_f=1e-12,
    # t models use length factor. Gamma*length dimensionless; R,L,G,C are per unit length
    length_init_m=1.0,
    log_halfwidth=9.0,
    # regularize log parameters, since many branches underidentified
    log_regularization=1e-4,
    # terminal loss choice: pseudo-Huber scaled residuals
    robust_delta=5.0,
    # add a global complex shunt/admittance offset (two fitted real components)
    offsets=('none','global'),
    # limits: put a cap on candidate count if needed; None evaluates all
    max_candidates=None,
    dtype=torch.float64,
)

SMUS=[tuple(range(1,7)),tuple(range(7,15)),tuple(range(15,23)),
      tuple(range(23,31)),tuple(range(31,39)),tuple(range(39,45))]
SMU_OF={p:g for g,arr in enumerate(SMUS) for p in arr}


def _parse_pins(x):
    if isinstance(x,str):
        try: x=ast.literal_eval(x)
        except (ValueError,SyntaxError): x=[int(s) for s in re.findall(r'\d+',x)]
    if isinstance(x,(int,np.integer)): x=[int(x)]
    x=tuple(sorted(int(p) for p in x))
    if not x or len(set(x))!=len(x) or any(p not in SMU_OF for p in x):
        raise ValueError(f'Invalid pin set: {x}')
    return x


def _pair_zone(h,l):
    if SMU_OF[h]!=SMU_OF[l]: return 4
    d=abs(h-l)
    if d==1: return 0 if max(h,l)%2==0 else 1
    if d==2: return 2
    return 3


def _build_data(frame,cfg):
    d=frame.copy().reset_index(drop=True)
    cols=[cfg[k] for k in ('c_keysight','g_keysight','c_v93k','g_v93k','high','low')]
    if any(c not in d for c in cols):
        raise ValueError(f'Missing columns: {set(cols)-set(d.columns)}')
    highs=[_parse_pins(z) for z in d[cfg['high']]]
    lows=[_parse_pins(z) for z in d[cfg['low']]]
    for i,(h,l) in enumerate(zip(highs,lows)):
        if set(h)&set(l): raise ValueError(f'H/L overlap at row {i}')
    n=len(d); w=2*np.pi*cfg['freq_hz']
    ck=d[cfg['c_keysight']].to_numpy(float)
    gk=d[cfg['g_keysight']].to_numpy(float)
    cv=d[cfg['c_v93k']].to_numpy(float)
    gv=d[cfg['g_v93k']].to_numpy(float)
    yk=gk+1j*w*ck; yv=gv+1j*w*cv
    if not np.isfinite(yk).all() or not np.isfinite(yv).all(): raise ValueError('Nonfinite C/G')
    cfg_keys=[str((h,l)) for h,l in zip(highs,lows)]
    signatures=[str(tuple(sum(p in s for p in h) for s in SMUS)+
                    tuple(sum(p in s for p in l) for s in SMUS)) for h,l in zip(highs,lows)]
    groups=np.array(cfg_keys if cfg['cv_group']=='configuration' else signatures)
    if np.unique(groups).size<2: raise ValueError('At least two distinct CV groups required')
    return dict(frame=d,highs=highs,lows=lows,yk=yk,yv=yv,w=w,n=n,groups=groups)


def _pair_features(highs,lows,mode):
    n=len(highs)
    # 5 pair zones + local combination increments
    # same-SMU increments (g,zone 0..3), cross-SMU increments (high-group,low-group)
    dim=5 if mode=='pair' else 5+6*4+6*6
    X=np.zeros((n,dim),dtype=float)
    for i,(H,L) in enumerate(zip(highs,lows)):
        for h in H:
            for l in L:
                z=_pair_zone(h,l); X[i,z]+=1
                if mode=='pair_smu':
                    gh,gl=SMU_OF[h],SMU_OF[l]
                    if gh==gl: X[i,5+4*gh+z]+=1
                    else: X[i,5+24+6*gh+gl]+=1
    return X


def _pin_features(highs,lows,mode):
    n=len(highs)
    # Pin-specific complex RLGC per pin, separate H and L role (88 groups)
    # Optional 6*6 H-SMU/L-SMU interaction increments
    dim=88 if mode=='pin' else 88+36
    X=np.zeros((n,dim),dtype=float)
    for i,(H,L) in enumerate(zip(highs,lows)):
        for h in H: X[i,h-1]+=1
        for l in L: X[i,44+l-1]+=1
        if mode=='pin_smu':
            # each populated H/L SMU combination contributes once;
            # independent of pin counts; same-SMU included
            for a in set(SMU_OF[p] for p in H):
                for b in set(SMU_OF[p] for p in L):
                    X[i,88+6*a+b]+=1
    return X


def _make_feature(highs,lows,family,aggregation):
    X=(_pair_features(highs,lows,family) if family.startswith('pair')
       else _pin_features(highs,lows,family))
    # Group-level aggregation weights, not a linear regression surrogate.
    # maxpin operates on branch incidence saturation, preserving zone groups.
    if aggregation=='maxpin': X=np.minimum(X,1.0)
    # Remove entirely absent columns to improve fitting; retain map for inference
    active=np.where(np.any(X!=0,axis=0))[0]
    return X[:,active],active


def _assemble_feature(highs,lows,family,aggregation,active):
    X=(_pair_features(highs,lows,family) if family.startswith('pair')
       else _pin_features(highs,lows,family))
    if aggregation=='maxpin': X=np.minimum(X,1.0)
    return X[:,active]


def _parameter_init(m,t_model,offset,cfg):
    # Parameter components per branch: [ln R, ln L, ln G, ln C]
    physical=np.array([cfg['r_init_ohm'],cfg['l_init_h'],cfg['g_init_s'],cfg['c_init_f']])
    anchor=np.log(physical)
    # Optimize theta deviations from logarithmic anchor for stable conditioning
    v=np.zeros((m,4),dtype=float).ravel()
    # Independent t uses log-length for each branch, RLGC uses shared length
    n_len=m if t_model=='independent' else 1
    v=np.r_[v,np.zeros(n_len)]
    if offset=='global': v=np.r_[v,np.zeros(2)]
    return v,anchor,n_len


def _predict_torch(v,X,YV,cfg,metadata):
    """Differentiable complex RLGC transmission transformation.
    Branch RLGC -> Z0 and tanh(gamma*l); aggregate branch characteristic
    admittances into effective Y0, and average/weighted-average branch t.
    Inverse two-port: YK=Y0*(YV-Y0*t)/(Y0-YV*t),
    plus optional complex shunt admittance offset.
    """
    m=metadata['n_features']; mode=metadata['aggregation']
    rg=v[:4*m].reshape(m,4)
    logphys=rg+metadata['anchor']
    vals=torch.exp(logphys)
    R,L,G,C=(vals[:,j] for j in range(4))
    j=1j; w=cfg['freq_hz']*2*np.pi
    z=R+j*w*L
    y=G+j*w*C
    y0=torch.sqrt(y/z)
    gamma=torch.sqrt(z*y)
    loglength=v[4*m:4*m+metadata['n_len']]
    length=cfg['length_init_m']*torch.exp(loglength)
    t=torch.tanh(gamma*length)
    # Torch inputs [n,m]; real multiplicities
    count=X.sum(dim=1).clamp(min=1.)
    Xc=X.to(torch.complex128)
    raw=Xc @ y0
    mean=raw/count
    if mode in ('sum','maxpin'): effective_y0=raw
    elif mode=='mean': effective_y0=mean
    elif mode=='parallel': effective_y0=raw
    elif mode=='series': effective_y0=1.0/(Xc@(1.0/y0)+1e-30)
    elif mode=='sqrt': effective_y0=raw/torch.sqrt(count)
    elif mode=='partial': effective_y0=0.25*raw+0.75*mean
    else: raise ValueError(mode)
    if metadata['t_aggregation']=='count':
        effective_t=(Xc@t)/count
    else:
        effective_t=(Xc@(y0*t))/(raw+1e-30)
    # Limit catastrophic near singular outputs by smooth-ish numerical handling
    denominator=effective_y0-YV*effective_t
    eps=1e-22
    den=denominator+eps
    result=effective_y0*(YV-effective_y0*effective_t)/den
    if metadata['offset']=='global':
        # physical offset remains S
        tail=v[4*m+metadata['n_len']:]
        result=result+(tail[0]+1j*tail[1])*1e-10
    return result


def _fit(Xnp, yv_np, yk_np, metadata,cfg,maxiter,seed=0,starts=1):
    n,m=Xnp.shape
    if m==0: return None
    dtype=cfg['dtype']; cdtype=torch.complex128
    X=torch.as_tensor(Xnp,dtype=dtype)
    YV=torch.as_tensor(yv_np,dtype=cdtype)
    YK=torch.as_tensor(yk_np,dtype=cdtype)
    x0,anchor,nlen=_parameter_init(m,metadata['t_model'],metadata['offset'],cfg)
    md=dict(metadata,anchor=torch.as_tensor(anchor,dtype=dtype),n_len=nlen,n_features=m)
    bounds=[(-cfg['log_halfwidth'],cfg['log_halfwidth'])]*(4*m+nlen)
    if metadata['offset']=='global': bounds.extend([(-1e4,1e4)]*2)
    rng=np.random.default_rng(seed)
    best=None
    c_scale=cfg['c_scale_f'];g_scale=cfg['g_scale_s'];delta=cfg['robust_delta']

    def fun_grad(q):
        v=torch.tensor(q,dtype=dtype,requires_grad=True)
        pred=_predict_torch(v,X,YV,cfg,md)
        err_c=(pred.imag-YK.imag)/(2*np.pi*cfg['freq_hz']*c_scale)
        err_g=(pred.real-YK.real)/g_scale
        e=torch.cat([err_c,err_g])
        # Smooth pseudo-Huber avoids single catastrophic inversion dominating
        loss=(delta**2*(torch.sqrt(1+(e/delta)**2)-1)).mean()
        # Small log ridge to avoid unconstrained degrees of freedom
        loss=loss+cfg['log_regularization']*torch.mean(v[:4*m]**2)
        loss.backward()
        val=float(loss.detach().cpu())
        jac=v.grad.detach().cpu().numpy().copy()
        if not np.isfinite(val) or not np.isfinite(jac).all():
            return 1e100,np.zeros_like(q)
        return val,jac

    for s in range(starts):
        init=x0.copy()
        if s: init[:4*m]+=rng.normal(0,0.5,4*m)
        res=minimize(fun_grad,init,method='L-BFGS-B',jac=True,
                     bounds=bounds,options=dict(maxiter=maxiter,ftol=1e-10,maxls=25))
        if best is None or res.fun<best.fun: best=res
    fitted=dict(metadata,theta=best.x,active=metadata['active'],
                anchor=anchor,n_len=nlen,n_features=m,
                train_loss=float(best.fun),success=bool(best.success),nit=int(best.nit))
    return fitted


def _predict_numpy(model,Xnp,yv_np,cfg):
    X=torch.as_tensor(Xnp,dtype=cfg['dtype'])
    YV=torch.as_tensor(yv_np,dtype=torch.complex128)
    md=dict(model,anchor=torch.as_tensor(model['anchor'],dtype=cfg['dtype']))
    with torch.no_grad():
        yp=_predict_torch(torch.tensor(model['theta'],dtype=cfg['dtype']),X,YV,cfg,md)
    return yp.numpy()


def _score(y_true,y_pred,w,cfg):
    if not np.isfinite(y_pred).all():
        return dict(C_RMSE_F=np.inf,C_MAE_F=np.inf,G_RMSE_S=np.inf,
                    G_MAE_S=np.inf,Combined=np.inf)
    ec=(y_pred.imag-y_true.imag)/w
    eg=y_pred.real-y_true.real
    return dict(C_RMSE_F=float(np.sqrt(np.mean(ec**2))),
                C_MAE_F=float(np.mean(abs(ec))),
                G_RMSE_S=float(np.sqrt(np.mean(eg**2))),
                G_MAE_S=float(np.mean(abs(eg))),
                Combined=float(np.sqrt(np.mean((ec/cfg['c_scale_f'])**2+
                                                (eg/cfg['g_scale_s'])**2))))


def run_rlgc_search(df,**overrides):
    cfg=dict(CFG);cfg.update(overrides)
    np.random.seed(cfg['seed']);torch.set_num_threads(max(1,min(torch.get_num_threads(),8)))
    info=_build_data(df,cfg)
    out=Path(cfg['out_dir']);out.mkdir(parents=True,exist_ok=True)
    n=info['n'];yv=info['yv'];yk=info['yk'];w=info['w'];groups=info['groups']
    specs=list(product(cfg['families'],cfg['aggregations'],
                       cfg['t_aggregations'],cfg['t_models'],cfg['offsets']))
    if cfg['max_candidates'] is not None: specs=specs[:cfg['max_candidates']]
    print(f'Rows={n}, distinct configurations={len(set(groups))}, candidates={len(specs)}',flush=True)
    tr,te=next(GroupShuffleSplit(n_splits=1,test_size=cfg['test_fraction'],
                                  random_state=cfg['seed']).split(np.arange(n),groups=groups))
    screen=[];start=time.time()
    for idx,(family,agg,tagg,tm,offset) in enumerate(specs):
        X,active=_make_feature(info['highs'],info['lows'],family,agg)
        metadata=dict(family=family,aggregation=agg,t_aggregation=tagg,
                      t_model=tm,offset=offset,active=active)
        try:
            fitted=_fit(X[tr],yv[tr],yk[tr],metadata,cfg,
                        cfg['screen_maxiter'],seed=cfg['seed'],starts=cfg['screen_starts'])
            yp=_predict_numpy(fitted,X[te],yv[te],cfg)
            m=_score(yk[te],yp,w,cfg)
        except Exception as exc:
            m=dict(C_RMSE_F=np.inf,C_MAE_F=np.inf,G_RMSE_S=np.inf,G_MAE_S=np.inf,Combined=np.inf)
            print('ERROR',idx,family,agg,tagg,tm,offset,str(exc)[:100],flush=True)
        screen.append(dict(id=idx,Family=family,Aggregation=agg,T_aggregation=tagg,
                           T_model=tm,Offset=offset,Params=4*len(active),**m))
        if (idx+1)%10==0:
            print(f'Screen {idx+1}/{len(specs)} | best C RMSE=' 
                  f'{min(x["C_RMSE_F"] for x in screen):.6e} F | '
                  f'{time.time()-start:.0f}s',flush=True)
    screen_df=pd.DataFrame(screen).sort_values(['C_RMSE_F','Combined']).reset_index(drop=True)
    screen_df.to_csv(out/'screen.csv',index=False)
    top_ids=screen_df.loc[np.isfinite(screen_df['C_RMSE_F']),'id'].head(cfg['top_k']).tolist()
    if not top_ids: raise RuntimeError('All models failed. See screen.csv')
    print('\n=== SCREEN TOP ===\n',screen_df.head(20).to_string(index=False),flush=True)
    cv=GroupShuffleSplit(n_splits=cfg['folds'],test_size=cfg['test_fraction'],
                         random_state=cfg['seed']+100)
    fold_rows=[]
    for fold,(train,test) in enumerate(cv.split(np.arange(n),groups=groups),1):
        for i in top_ids:
            family,agg,tagg,tm,offset=specs[i]
            X,active=_make_feature(info['highs'],info['lows'],family,agg)
            meta=dict(family=family,aggregation=agg,t_aggregation=tagg,
                      t_model=tm,offset=offset,active=active)
            try:
                model=_fit(X[train],yv[train],yk[train],meta,cfg,cfg['final_maxiter'],
                           seed=cfg['seed']+fold,starts=cfg['final_starts'])
                yp=_predict_numpy(model,X[test],yv[test],cfg)
                ms=_score(yk[test],yp,w,cfg)
            except Exception as exc:
                print('FINAL ERROR',i,str(exc)[:100],flush=True)
                ms=dict(C_RMSE_F=np.inf,C_MAE_F=np.inf,G_RMSE_S=np.inf,G_MAE_S=np.inf,Combined=np.inf)
            fold_rows.append(dict(id=i,Fold=fold,Family=family,Aggregation=agg,
                                  T_aggregation=tagg,T_model=tm,Offset=offset,**ms))
            print(f'Final fold {fold} id {i}: C={ms["C_RMSE_F"]:.6e} F, '
                  f'G={ms["G_RMSE_S"]:.6e} S',flush=True)
    folds=pd.DataFrame(fold_rows);folds.to_csv(out/'folds.csv',index=False)
    ranking=(folds.groupby(['id','Family','Aggregation','T_aggregation','T_model','Offset'])
             [['C_RMSE_F','C_MAE_F','G_RMSE_S','G_MAE_S','Combined']]
             .mean().reset_index().sort_values(['C_RMSE_F','Combined']))
    ranking.to_csv(out/'ranking.csv',index=False)
    print('\n=== VALIDATION RANKING ===\n',ranking.to_string(index=False),flush=True)
    best_id=int(ranking.iloc[0]['id'])
    family,agg,tagg,tm,offset=specs[best_id]
    X,active=_make_feature(info['highs'],info['lows'],family,agg)
    meta=dict(family=family,aggregation=agg,t_aggregation=tagg,t_model=tm,offset=offset,active=active)
    model=_fit(X,yv,yk,meta,cfg,cfg['refit_maxiter'],seed=cfg['seed']+999,
               starts=max(cfg['final_starts'],3))
    yp=_predict_numpy(model,X,yv,cfg)
    report=info['frame'].copy()
    report['C_pred_F']=yp.imag/w
    report['G_pred_S']=yp.real
    report['C_error_F']=report['C_pred_F'].to_numpy()-yk.imag/w
    report['G_error_S']=report['G_pred_S'].to_numpy()-yk.real
    report['abs_C_error_F']=abs(report['C_error_F'])
    report.to_csv(out/'winner_predictions.csv',index=False)
    payload=dict(model=model,config={k:v for k,v in cfg.items() if k!='dtype'},
                 ranking=ranking.iloc[0].to_dict(),smus=SMUS)
    joblib.dump(payload,out/'winner.joblib')
    print('\n=== WINNER ===',ranking.iloc[0].to_dict(),flush=True)
    print('IN-SAMPLE:',_score(yk,yp,w,cfg),flush=True)
    print('Files:',out.resolve(),flush=True)
    return dict(ranking=ranking,screen=screen_df,folds=folds,model=model,
                predictions=report,config=cfg)


def predict_rlgc(saved_path,df_new):
    """Use trained model on new rows. Requires high_pins, low_pins, C_v93k, G_v93k."""
    payload=joblib.load(saved_path);cfg=dict(CFG);cfg.update(payload['config'])
    model=payload['model'];data=df_new.copy()
    highs=[_parse_pins(x) for x in data[cfg['high']]]
    lows=[_parse_pins(x) for x in data[cfg['low']]]
    X=_assemble_feature(highs,lows,model['family'],model['aggregation'],model['active'])
    w=2*np.pi*cfg['freq_hz']
    yv=data[cfg['g_v93k']].to_numpy(float)+1j*w*data[cfg['c_v93k']].to_numpy(float)
    pred=_predict_numpy(model,X,yv,cfg)
    result=data.copy();result['C_pred_F']=pred.imag/w;result['G_pred_S']=pred.real
    return result


if __name__=='__main__':
    if 'df' not in globals():
        print('Load df into your Python session, then: results = run_rlgc_search(df)')
    else:
        results=run_rlgc_search(df)
