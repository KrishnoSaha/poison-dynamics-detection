import sys, glob, json, os
import numpy as np
def auc(s, y):
    n=len(y); n1=int(y.sum()); n0=n-n1
    if n1==0 or n0==0: return float('nan')
    o=np.argsort(s,kind='mergesort'); r=np.empty(n); r[o]=np.arange(1,n+1)
    return (r[y==1].sum()-n1*(n1+1)/2.0)/(n1*n0)
def feats(L,C):
    k=max(1,L.shape[1]//10)
    f={'mean_loss':L.mean(1),'early_loss':L[:,:k].mean(1),'final_loss':L[:,-k:].mean(1),
       'drop_loss':L[:,:k].mean(1)-L[:,-k:].mean(1),'vol_loss':L.std(1)}
    if C is not None:
        f.update({'mean_conf':C.mean(1),'final_conf':C[:,-k:].mean(1),'rise_conf':C[:,-k:].mean(1)-C[:,:k].mean(1)})
    return f
files=sorted(f for p in sys.argv[1:] for f in glob.glob(p))
if not files: sys.exit('no files matched')
rows=[]
for p in files:
    z=np.load(p,allow_pickle=True)
    L=z['loss'].astype(float); C=z['conf'].astype(float) if 'conf' in z else None
    y=z['y'].astype(int); P=z['is_poisoned'].astype(int) if 'is_poisoned' in z else np.zeros(len(y),int)
    if P.sum()==0: print('skip (no poison):',p); continue
    tgt=0
    if 'config' in z:
        try: tgt=int(json.loads(str(z['config'])).get('target',0))
        except Exception: pass
    sel=(y==tgt); F=feats(L,C)
    print('==',os.path.basename(p),' n=%d poison=%d target=%d'%(len(y),P.sum(),tgt))
    print('   %-11s %9s %12s'%('feature','all','same-label'))
    best=None
    for k,v in F.items():
        a=auc(v,P); b=auc(v[sel],P[sel]); print('   %-11s %9.3f %12.3f'%(k,a,b))
        if best is None or abs(b-.5)>abs(best[1]-.5): best=(k,b)
    rows.append((os.path.basename(p)[:36],best[0],best[1],'higher' if best[1]>.5 else 'LOWER'))
print('\nSUMMARY (AUC above 0.5 means poison is HIGHER)')
for r in rows: print('%-37s %-11s AUC=%.3f poison is %s'%r)
