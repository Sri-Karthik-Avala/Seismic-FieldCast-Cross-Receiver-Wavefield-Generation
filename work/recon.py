import numpy as np, pandas as pd, json, time
t0=time.time()
tr=pd.read_csv('train.csv')
S=np.stack([np.load('supports/train/%s.npy'%i) for i in tr.id]).astype(np.float32)  # [N,4,3,600]
Y=np.stack([np.array(json.loads(w),dtype=np.float32) for w in tr.wavefield_json])   # [N,3,600]
G=np.stack([np.array([[o['x_km'],o['y_km'],o['z_km']] for o in json.loads(j)]) for j in tr.support_receivers_json]).astype(np.float32) #[N,4,3]
T=tr[['target_x_km','target_y_km','target_z_km']].values.astype(np.float32)
np.save('work/S.npy',S); np.save('work/Y.npy',Y); np.save('work/G.npy',G); np.save('work/T.npy',T)
print('loaded',S.shape,Y.shape,time.time()-t0)
def skill(P):
    r=np.sqrt(((P-Y)**2).mean(axis=(1,2))); s=np.sqrt((Y**2).mean(axis=(1,2)))
    return np.maximum(0,1-r/np.maximum(s,1e-12))
print('Y stats min %.3f max %.3f mean %.3f'%(Y.min(),Y.max(),Y.mean()))
print('S stats min %.3f max %.3f mean %.3f'%(S.min(),S.max(),S.mean()))
print('scale ||Y|| mean %.3f med %.3f min %.3f max %.3f'%(np.sqrt((Y**2).mean(axis=(1,2))).mean(),np.median(np.sqrt((Y**2).mean(axis=(1,2)))),np.sqrt((Y**2).mean(axis=(1,2))).min(),np.sqrt((Y**2).mean(axis=(1,2))).max()))
d=np.linalg.norm(G-T[:,None,:],axis=2)  # [N,4] target-support dist
print('dist to supports: col means',d.mean(0),'sorted-check frac',np.mean(np.all(np.diff(d,axis=1)>=0,axis=1)))
print('dist range',d.min(),d.max())
# baselines
print('zero        %.4f'%skill(np.zeros_like(Y)).mean())
for j in range(4):
    print('support_%d  %.4f'%(j,skill(S[:,j]).mean()))
print('mean4       %.4f'%skill(S.mean(1)).mean())
for p in [1,2,3]:
    w=1.0/np.maximum(d,1e-3)**p; w=w/w.sum(1,keepdims=True)
    print('idw p=%d     %.4f'%(p,skill((S*w[:,:,None,None]).sum(1)).mean()))
# best global scalar on mean4
M=S.mean(1)
for a in [0.7,0.8,0.9,1.0,1.1,1.2,1.3]:
    print('  a=%.1f*mean4 %.4f'%(a,skill(a*M).mean()))
# oracle per-row scalar on mean4
num=(M*Y).sum(axis=(1,2)); den=(M*M).sum(axis=(1,2))
a=num/np.maximum(den,1e-9)
print('oracle scalar mean4 %.4f  (a mean %.3f std %.3f)'%(skill(a[:,None,None]*M).mean(),a.mean(),a.std()))
# oracle per-band scalar
numb=(M*Y).sum(axis=2); denb=(M*M).sum(axis=2); ab=numb/np.maximum(denb,1e-9)
print('oracle per-band scalar %.4f'%skill(ab[:,:,None]*M).mean())
# oracle best-linear-combo of 4 supports (per row, least squares over 1800 pts)
sk=[]
for i in range(len(Y)):
    A=S[i].reshape(4,-1).T; b=Y[i].reshape(-1)
    c,_,_,_=np.linalg.lstsq(A,b,rcond=None); p=(A@c).reshape(3,600)
    sk.append(max(0,1-np.sqrt(((p-Y[i])**2).mean())/np.sqrt((Y[i]**2).mean())))
print('ORACLE lin-combo of 4 supports %.4f'%np.mean(sk))
# oracle: best per-band per-support linear combo
sk=[]
for i in range(len(Y)):
    tot=0
    for b_ in range(3):
        A=S[i,:,b_,:].T; y=Y[i,b_]
        c,_,_,_=np.linalg.lstsq(A,y,rcond=None); tot+=((A@c-y)**2).sum()
    sk.append(max(0,1-np.sqrt(tot/1800)/np.sqrt((Y[i]**2).mean())))
print('ORACLE per-band lin-combo %.4f'%np.mean(sk))
print('t',time.time()-t0)
