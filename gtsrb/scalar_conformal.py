import sys, glob, json, os
import numpy as np
REPS=5
def feats(L,C):
    k=max(1,L.shape[1]//10)
    f={'mean_loss':L.mean(1),'early_loss':L[:,:k].mean(1),'final_loss':L[:,-k:].mean(1),
       'drop_loss':L[:,:k].mean(1)-L[:,-k:].mean(1)}
    if C is not None:
        f.update({'mean_conf':C.mean(1),'final_conf':C[:,-k:].mean(1),'rise_conf':C[:,-k:].mean(1)-C[:,:k].mean(1)})
    return f
def run(s,y,P,ncal,alpha,rng):
    flag=np.zeros(len(s),bool); calmask=np.zeros(len(s),bool)
    for c in np.unique(y):
        idx=np.where((y==c)&(P==0))[0]
        if len(idx)<ncal+20: continue
        cal=rng.choice(idx,ncal,replace=False); calmask[cal]=True
        srt=np.sort(s[cal]); j=int(np.ceil((ncal+1)*(1-alpha)))-1
        thr=srt[min(j,ncal-1)]
        m=(y==c); flag[m]=s[m]>thr
    clean_test=(P==0)&~calmask
    tpr=flag[P==1].mean() if P.sum() else np.nan
    return tpr, flag[clean_test].mean()
files=sorted(f for p in sys.argv[1:] for f in glob.glob(p))
if not files: sys.exit('no files matched')
for p in files:
    z=np.load(p,allow_pickle=True)
    L=z['loss'].astype(float); C=z['conf'].astype(float) if 'conf' in z else None
    y=z['y'].astype(int); P=z['is_poisoned'].astype(int) if 'is_poisoned' in z else np.zeros(len(y),int)
    cnt=np.bincount(y[P==0]); ncal=500 if np.median(cnt)>=1250 else 60
    print('==',os.path.basename(p),'n=%d poison=%d ncal=%d'%(len(y),P.sum(),ncal))
    print('   %-11s %-6s %9s %9s %9s %9s'%('feature','tail','TPR@5%','FPR@5%','TPR@1%','FPR@1%'))
    for k,v in feats(L,C).items():
        for tail,s in (('upper',v),('lower',-v)):
            out=[]
            for a in (0.05,0.01):
                r=np.array([run(s,y,P,ncal,a,np.random.default_rng(i)) for i in range(REPS)])
                out+= [r[:,0].mean(), r[:,1].mean()]
            print('   %-11s %-6s %9.3f %9.3f %9.3f %9.3f'%(k,tail,*out))
