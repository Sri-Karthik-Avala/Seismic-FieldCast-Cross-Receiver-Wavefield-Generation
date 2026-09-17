import numpy as np, time
S=np.load('work/S.npy');Y=np.load('work/Y.npy');G=np.load('work/G.npy');T=np.load('work/T.npy')
N=len(Y)
def sk1(p,y): return max(0.0,1-np.sqrt(((p-y)**2).mean())/np.sqrt((y**2).mean()))
def shift(a,s):  # a [...,600], integer shift s (positive = delay)
    o=np.zeros_like(a); s=int(s)
    if s==0: return a.copy()
    if s>0: o[...,s:]=a[...,:-s]; o[...,:s]=a[...,:1]
    else: o[...,:s]=a[...,-s:]; o[...,s:]=a[...,-1:]
    return o
M=S.mean(1)
# 1 oracle global shift of mean4
best=[];bs=[]
for i in range(N):
    sc=[(sk1(shift(M[i],s),Y[i]),s) for s in range(-60,61,2)]
    v,s=max(sc); best.append(v); bs.append(s)
print('ORACLE global shift mean4 %.4f  shift mean %.1f std %.1f'%(np.mean(best),np.mean(bs),np.std(bs)))
# 2 oracle per-support shift then lstsq
res=[]
for i in range(N):
    Sh=[]
    for j in range(4):
        sc=[(np.corrcoef(shift(S[i,j],s).ravel(),Y[i].ravel())[0,1],s) for s in range(-80,81,2)]
        v,s=max(sc); Sh.append(shift(S[i,j],s))
    A=np.stack(Sh).reshape(4,-1).T
    c,_,_,_=np.linalg.lstsq(A,Y[i].ravel(),rcond=None)
    res.append(sk1((A@c).reshape(3,600),Y[i]))
print('ORACLE per-support shift + lstsq %.4f'%np.mean(res))
# 3 oracle time-varying: smooth per-support weights (basis of 12 raised cosines)
K=12; tt=np.arange(600)
B=np.stack([np.exp(-0.5*((tt-c0)/45.)**2) for c0 in np.linspace(0,599,K)]); B/=B.sum(0,keepdims=True)
res=[]
for i in range(N):
    cols=[]
    for j in range(4):
        for k in range(K): cols.append((S[i,j]*B[k]).ravel())
    A=np.stack(cols).T
    c,_,_,_=np.linalg.lstsq(A,Y[i].ravel(),rcond=None)
    res.append(sk1((A@c).reshape(3,600),Y[i]))
print('ORACLE smooth time-varying weights (48 dof) %.4f'%np.mean(res))
# 4 how much of Y is "shared event shape"? oracle = per-row best of (mean of 3 targets in group)? skip
# 5 onset picks: does target onset predict from geometry?
def onset(a):
    e=a.mean(0); e=e/max(e.max(),1e-9)
    idx=np.where(e>0.5)[0]
    return idx[0] if len(idx) else 100
oS=np.array([[onset(S[i,j]) for j in range(4)] for i in range(N)])
oY=np.array([onset(Y[i]) for i in range(N)])
d=np.linalg.norm(G-T[:,None,:],axis=2)
dS=None
print('onset target mean %.1f std %.1f ; supports mean %s'%(oY.mean(),oY.std(),oS.mean(0).round(1)))
print('corr(oY, oS.mean) %.3f'%np.corrcoef(oY,oS.mean(1))[0,1])
# fit source per row from support onsets assuming t=t0+R/v, v grid
from itertools import product
def fitsrc(coords,tobs,v):
    # least squares over grid then refine
    best=None
    xs=np.linspace(coords[:,0].min()-40,coords[:,0].max()+40,25)
    ys=np.linspace(coords[:,1].min()-40,coords[:,1].max()+40,25)
    for x in xs:
        for y in ys:
            R=np.sqrt((coords[:,0]-x)**2+(coords[:,1]-y)**2)
            t0=np.mean(tobs-R/v*10)
            r=((tobs-(t0+R/v*10))**2).sum()
            if best is None or r<best[0]: best=(r,x,y,t0)
    return best
pred=[];
for i in range(N):
    r,x,y,t0=fitsrc(G[i],oS[i].astype(float),5.0)
    R=np.sqrt((T[i,0]-x)**2+(T[i,1]-y)**2)
    pred.append(t0+R/5.0*10)
pred=np.array(pred)
print('fitted-source onset pred: corr %.3f  MAE %.1f frames ; naive(meanonset) MAE %.1f'%(np.corrcoef(pred,oY)[0,1],np.abs(pred-oY).mean(),np.abs(oS.mean(1)-oY).mean()))
