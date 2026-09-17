import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
S=np.load('work/S.npy');Y=np.load('work/Y.npy');G=np.load('work/G.npy');T=np.load('work/T.npy')
tr=pd.read_csv('train.csv',usecols=['event_group']); grp=tr.event_group.values
N=len(Y)
def sk(P): 
    r=np.sqrt(((P-Y)**2).mean(axis=(1,2))); s=np.sqrt((Y**2).mean(axis=(1,2)))
    return np.maximum(0,1-r/np.maximum(s,1e-12)).mean()
M=S.mean(1)
print('mean4 %.4f'%sk(M))
for s_ in [0.5,1,2,3,5,8]:
    print('  smooth sigma=%.1f %.4f'%(s_,sk(gaussian_filter1d(M,s_,axis=-1))))
# geometric / linear-domain mean
L=np.expm1(S); print('linear-domain mean %.4f'%sk(np.log1p(L.mean(1))))
print('geom(logmean)=mean4 same')
for p in [0.5,0.75,1.5,2.0]:
    print('  power-mean p=%.2f %.4f'%(p,sk(np.log1p((( L**p).mean(1))**(1/p)))))
# median of 4
print('median4 %.4f'%sk(np.median(S,1)))
print('mean of 3 nearest %.4f'%sk(S[:,:3].mean(1)))
print('mean of 2 nearest %.4f'%sk(S[:,:2].mean(1)))
# per-band scalar fit (grouped CV)
from sklearn.model_selection import GroupKFold
gkf=GroupKFold(n_splits=5)
P=np.zeros_like(Y)
for trI,teI in gkf.split(np.arange(N),groups=grp):
    a=np.array([ (M[trI,b]*Y[trI,b]).sum()/max((M[trI,b]**2).sum(),1e-9) for b in range(3)])
    P[teI]=a[None,:,None]*M[teI]
print('CV per-band scalar %.4f  a=%s'%(sk(P),a.round(3)))
# distance-softmax weights, grouped CV over tau
d=np.linalg.norm(G-T[:,None,:],axis=2)
for tau in [5,10,20,40,80,1e6]:
    w=np.exp(-d/tau); w/=w.sum(1,keepdims=True)
    print('  softmax tau=%.0f %.4f'%(tau,sk((S*w[:,:,None,None]).sum(1))))
# combine: smooth + per-band scalar
Ms=gaussian_filter1d(M,2.0,axis=-1)
P=np.zeros_like(Y)
for trI,teI in gkf.split(np.arange(N),groups=grp):
    a=np.array([ (Ms[trI,b]*Y[trI,b]).sum()/max((Ms[trI,b]**2).sum(),1e-9) for b in range(3)])
    P[teI]=a[None,:,None]*Ms[teI]
print('CV smooth2+per-band scalar %.4f'%sk(P))
# ridge on stacked features per (band,time) -> too big; instead: linear map from 12 support chans to 3 out chans, global, CV
X=S.reshape(N,12,600)
P=np.zeros_like(Y)
for trI,teI in gkf.split(np.arange(N),groups=grp):
    A=X[trI].transpose(0,2,1).reshape(-1,12); B=Y[trI].transpose(0,2,1).reshape(-1,3)
    W=np.linalg.solve(A.T@A+1e-3*np.eye(12),A.T@B)
    P[teI]=(X[teI].transpose(0,2,1)@W).transpose(0,2,1)
print('CV global 12->3 linear %.4f'%sk(P))
print(W.round(2))
