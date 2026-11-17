"""SkillCorner open data check (EXTERNAL: 20 A-League 2024/25 matches), see README.md section 6.

    python solution/experiments_skillcorner.py --skillcorner <folder> --metrica <metrica data/>

<folder> holds, for every match id: <id>.jsonl (tracking_extrapolated), <id>_match.json,
<id>_phases.csv (phases_of_play), downloaded from github.com/SkillCorner/opendata.
Grid on 10 matches (even indices), report on the 10 others, and compare with Metrica.
"""
import argparse
import itertools
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "evaluation"), os.path.dirname(HERE)]
import possession as P  # noqa: E402
import calibrate_metrica as C  # noqa: E402
import skillcorner_games as SCG  # noqa: E402
import external_games as E  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--skillcorner", required=True)
ap.add_argument("--metrica", required=True)
args = ap.parse_args()
GS = SCG.load_all(args.skillcorner)
G = E.load_all(args.metrica)
FPS=25
def degrade_mask(g, seed=0, missing=0.41, noise=0.3):
    rng=np.random.default_rng(seed); bx=g['bx'].copy(); by=g['by'].copy(); n=len(bx)
    graded=g['ref']!='X'
    drop=np.isnan(bx)
    while drop[graded].mean()<missing:
        s=rng.integers(0,n); drop[s:s+int(rng.uniform(4,15)*FPS)]=True
    bx[drop]=np.nan; by[drop]=np.nan
    bx=bx+rng.normal(0,noise,n); by=by+rng.normal(0,noise,n)
    px=g['px']+rng.normal(0,noise,g['px'].shape); py=g['py']+rng.normal(0,noise,g['py'].shape)
    return px,py,bx,by
PRE={}
def prep(i, seed=0):
    if (i,seed) not in PRE:
        g=GS[i]; px,py,bx,by=degrade_mask(g,seed)
        ps=C.player_speed(g['px'],g['py'])
        with np.errstate(all='ignore'): act=P.rolling_mean_nan(np.nanmean(ps,axis=1),75)
        PRE[(i,seed)]=(px,py,bx,by,act,P.ball_speed(bx,by))
    return PRE[(i,seed)]
def run_sc(i, params, seed=0):
    for k,v in params.items(): setattr(P,k,v)
    g=GS[i]; px,py,bx,by,act,bs=prep(i,seed)
    dead=P.dead_mask(bx,by,bs,act,[],g['pitch'])
    ev=P.control_evidence(px,py,g['team'],bx,by)
    lab=P.label_match(ev,dead,[],len(bx),~np.isnan(bx))
    r=C.score(g['ref'],lab); return np.array([r['spell_f1_mean'],r['macro_f1'],r['n_pred_spells'],r['n_gt_spells']])
V2=dict(WALKING_MS=1.8,STILL_MIN_S=3.0,NOBALL_MIN_S=4.0,MIN_DEAD_S=2.0,KICK_SPEED_MS=4.0,NOBALL_ALONE_S=None,CONFIRM_FRAMES=8,CONTROL_RADIUS_M=1.5,CONTEST_MARGIN_M=0.5,RUN_GAP_FRAMES=50,GAP_DATING=True,USE_SHAPE=False)
V1=dict(V2,WALKING_MS=1.6,CONFIRM_FRAMES=13,CONTROL_RADIUS_M=2.0,GAP_DATING=False)
TRAIN=list(range(0,20,2)); TEST=list(range(1,20,2))
def mean_over(ids,p): return np.mean([run_sc(i,p) for i in ids],axis=0)


def run_metrica(i, params, seed):
    for k, v in params.items():
        setattr(P, k, v)
    g = G[i]
    dpx, dpy, dbx, dby = C.degrade(g["px"], g["py"], g["bx"], g["by"], seed=seed)
    ps = C.player_speed(g["px"], g["py"])
    act = P.rolling_mean_nan(np.nanmean(ps, axis=1), 75)
    dead = P.dead_mask(dbx, dby, P.ball_speed(dbx, dby), act, [], C.PITCH)
    ev = P.control_evidence(dpx, dpy, g["team"], dbx, dby)
    lab = P.label_match(ev, dead, [], len(dbx), ~np.isnan(dbx))
    return C.score(g["ref"], lab)["spell_f1_mean"]


cands = {
    "v1": V1, "v2": V2,
    "skillcorner_best": dict(V2, CONFIRM_FRAMES=20, CONTROL_RADIUS_M=2.0, WALKING_MS=1.4),
    "compromise": dict(V2, CONFIRM_FRAMES=13, CONTROL_RADIUS_M=2.0, WALKING_MS=1.8),
}
for name, p in cands.items():
    m = [np.mean([run_metrica(i, p, s) for s in (0, 1)]) for i in range(3)]
    te = mean_over(TEST, p)[0]
    print("{:18s} metrica {} mean {:.3f} | skillcorner test {:.3f} | overall {:.3f}".format(
        name, np.round(m, 3), np.mean(m), te, (np.mean(m) + te) / 2))
