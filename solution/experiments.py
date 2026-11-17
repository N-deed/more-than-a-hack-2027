"""Reproduce the experiments reported in README.md, on EXTERNAL data only (Metrica sample games).

    python solution/experiments.py --sample ../metrica-sample-data/data

1. v1 vs v2 rules on the three games (game 3 was used for calibration, games 1 and 2 were not).
2. v2 + team-shape model during ball gaps (trained leave-one-game-out).
3. Learned per-frame model + Viterbi decoding instead of the rules (leave-one-game-out).
Every game is degraded with the challenge gap model (calibrate_metrica.degrade), two seeds.
"""
import argparse
import itertools
import sys
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "evaluation"), os.path.dirname(HERE)]
import shape_model as SM  # noqa: E402
import possession as P  # noqa: E402
import calibrate_metrica as C  # noqa: E402
import external_games as E  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sample", required=True, help="the data/ folder of metrica sample-data")
G = E.load_all(ap.parse_args().sample)
F=[SM.team_features(g['px'],g['py'],g['team'],g['period']) for g in G]
def data(i, step=5):
    r=G[i]['ref']; m=((r=='H')|(r=='A')) & ~np.isnan(F[i]).any(1)
    idx=np.where(m)[0][::step]; return F[i][idx], (r[idx]=='H').astype(float)
models={}
for test in range(3):
    tr=[i for i in range(3) if i!=test]
    models[test]=SM.fit_logistic(np.vstack([data(i)[0] for i in tr]), np.concatenate([data(i)[1] for i in tr]))
DEG={}
for i,g in enumerate(G):
    ps=C.player_speed(g['px'],g['py'])
    DEG[i]=[(C.degrade(g['px'],g['py'],g['bx'],g['by'],seed=s), ps) for s in (0,1)]
def run(i, params, use_shape):
    for k,v in params.items(): setattr(P,k,v)
    g=G[i]; out=[]
    for (dpx,dpy,dbx,dby),ps in DEG[i]:
        with np.errstate(all='ignore'): ms=np.nanmean(ps,axis=1)
        act=P.rolling_mean_nan(ms,75); bs=P.ball_speed(dbx,dby)
        dead=P.dead_mask(dbx,dby,bs,act,[],C.PITCH)
        ev=P.control_evidence(dpx,dpy,g['team'],dbx,dby)
        sp=None
        if use_shape:
            f=SM.team_features(dpx,dpy,g['team'],g['period']); sp=SM.predict_home(f,models[i])
        lab=P.label_match(ev,dead,[],len(dbx),~np.isnan(dbx),sp)
        r=C.score(g['ref'],lab); out.append((r['spell_f1_mean'],r['macro_f1'],r['n_pred_spells']))
    return np.mean(out,axis=0)

def ball_feats(px,py,team,bx,by):
    n,S=px.shape; ts=team if team.ndim==2 else np.broadcast_to(team[None,:],(n,S))
    d=np.hypot(px-bx[:,None],py-by[:,None])
    dh=np.nanmin(np.where(ts==0,d,np.inf),1); da=np.nanmin(np.where(ts==1,d,np.inf),1)
    dh=np.minimum(dh,15); da=np.minimum(da,15)
    nh=np.sum((ts==0)&(d<3),1); na=np.sum((ts==1)&(d<3),1)
    f=np.vstack([da-dh, np.log1p(da)-np.log1p(dh), nh-na]).T.astype(float)
    f[np.isnan(bx)]=np.nan
    return f
def feats(g,px,py,bx,by):
    bf=ball_feats(px,py,g['team'],bx,by)
    bf=np.nan_to_num(bf)  # 0 when no ball (neutral, antisymmetric)
    sf=SM.team_features(px,py,g['team'],g['period'])
    # smooth ball features over 0.4s (antisymmetric)
    return np.hstack([bf, np.nan_to_num(sf)])
def viterbi(logit, a, b, lam):
    # 2 states H(1)/A(0); emission log p ratio ~ logit; switch cost lam
    L=logit[a:b]; n=len(L)
    sH=0.0; sA=0.0; bpH=np.zeros(n,bool); bpA=np.zeros(n,bool)
    # use log-sigmoid emissions
    lpH=-np.logaddexp(0,-L); lpA=-np.logaddexp(0,L)
    for t in range(n):
        nH=max(sH, sA-lam); bpH[t]= sA-lam> sH
        nA=max(sA, sH-lam); bpA[t]= sH-lam> sA
        sH=nH+lpH[t]; sA=nA+lpA[t]
    st=np.zeros(n,bool); cur = sH>=sA
    for t in range(n-1,-1,-1):
        st[t]=cur
        cur = (not cur) if (bpH[t] if cur else bpA[t]) else cur
    return st
FT = {}
for i,g in enumerate(G):
    FT[i]=feats(g,g['px'],g['py'],g['bx'],g['by'])
def train(tr):
    X=[];y=[]
    for i in tr:
        r=G[i]['ref']; m=np.where((r=='H')|(r=='A'))[0][::3]; X.append(FT[i][m]); y.append((r[m]=='H').astype(float))
    return SM.fit_logistic(np.vstack(X),np.concatenate(y),l2=1.0)
def run_hmm(i, model, lam, deg_k=None):
    g=G[i]; out=[]
    for (dpx,dpy,dbx,dby),ps in DEG[i]:
        with np.errstate(all='ignore'): ms=np.nanmean(ps,axis=1)
        act=P.rolling_mean_nan(ms,75); bs=P.ball_speed(dbx,dby)
        dead=P.dead_mask(dbx,dby,bs,act,[],C.PITCH)
        f=feats(g,dpx,dpy,dbx,dby); mean,scale,w=model
        logit=(f/scale)@w
        lab=np.full(len(dbx),'D',dtype='<U1')
        for a,b in P.runs(~dead):
            st=viterbi(logit,a,b,lam); lab[a:b]=np.where(st,'H','A')
        r=C.score(g['ref'],lab); out.append((r['spell_f1_mean'],r['macro_f1'],r['n_pred_spells']))
    return np.mean(out,axis=0)

V2 = dict(WALKING_MS=1.8, STILL_MIN_S=3.0, NOBALL_MIN_S=4.0, MIN_DEAD_S=2.0, KICK_SPEED_MS=4.0,
          NOBALL_ALONE_S=None, CONFIRM_FRAMES=8, CONTROL_RADIUS_M=1.5, CONTEST_MARGIN_M=0.5,
          RUN_GAP_FRAMES=50, GAP_DATING=True, USE_SHAPE=False)
V1 = dict(V2, WALKING_MS=1.6, CONFIRM_FRAMES=13, CONTROL_RADIUS_M=2.0, GAP_DATING=False)

print("columns: spellF1, macroF1, predicted possessions (mean of 2 seeds)")
for name, p in (("v1", V1), ("v2", V2)):
    print(name, [run(i, p, False).round(3).tolist() for i in range(3)])
for sp_, sc in itertools.product([0.7, 0.8, 0.9], [1.0, 2.0, 4.0]):
    p = dict(V2, USE_SHAPE=True, SHAPE_P=sp_, SHAPE_CONFIRM_S=sc)
    print("v2+shape P>=%.1f %.0fs" % (sp_, sc), [run(i, p, True).round(3).tolist() for i in range(3)])
for k, v in V2.items():
    setattr(P, k, v)
for i in range(3):
    FT[i] = feats(G[i], G[i]["px"], G[i]["py"], G[i]["bx"], G[i]["by"])
for test in range(3):
    model = train([i for i in range(3) if i != test])
    for lam in (5, 10, 20, 40, 80):
        print("viterbi", G[test]["name"], "switch cost", lam, run_hmm(test, model, lam).round(3).tolist())
