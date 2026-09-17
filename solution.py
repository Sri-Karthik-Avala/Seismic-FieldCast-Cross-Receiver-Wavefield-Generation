# made by - Karthik
import sys, json, csv, math
from pathlib import Path

public_dir = Path(sys.argv[1])
submission_out = Path(sys.argv[2])
submission_out.parent.mkdir(parents=True, exist_ok=True)

NF = 600
NB = 3

_ids = []
with open(public_dir / "test.csv", newline="", encoding="utf-8") as _fh:
    for _r in csv.DictReader(_fh):
        _ids.append(_r["id"])

_acc = [[0.0] * NF for _ in range(NB)]
_cnt = 0
with open(public_dir / "train.csv", newline="", encoding="utf-8") as _fh:
    for _r in csv.DictReader(_fh):
        _w = json.loads(_r["wavefield_json"])
        for _b in range(NB):
            _src = _w[_b]
            _dst = _acc[_b]
            for _t in range(NF):
                _dst[_t] += _src[_t]
        _cnt += 1

_prior = json.dumps(
    [[round(_acc[_b][_t] / max(_cnt, 1), 4) for _t in range(NF)] for _b in range(NB)],
    separators=(",", ":"),
)
with open(submission_out, "w", newline="", encoding="utf-8") as _fh:
    _wr = csv.writer(_fh)
    _wr.writerow(["id", "wavefield_json"])
    for _i in _ids:
        _wr.writerow([_i, _prior])

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 1234
N_FOLDS = 5
N_SEEDS = 1
EPOCHS = 45
BATCH = 16
LR = 3e-3
WD = 1e-4
WIDTH = 32
DILATIONS = [1, 2, 4, 8, 16, 32]
SIGMA_GRID = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0]
BASE_SIGMAS = [2.0, 8.0, 20.0]
AUG_SHIFT = 40
AUG_GAIN_LO = 0.80
AUG_GAIN_HI = 1.25
AUG_BAND_SCALE = 0.06
AUG_BAND_OFF = 0.03
AUG_NOISE = 0.02
VAL_MAX = 30.0
NTHREAD = 8

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cpu")
torch.set_num_threads(NTHREAD)


def gauss_kernel(sigma):
    if sigma <= 0:
        k = torch.zeros(1, dtype=torch.float32)
        k[0] = 1.0
        return k
    r = int(max(1, math.ceil(3.0 * sigma)))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def smooth(x, sigma):
    k = gauss_kernel(sigma)
    if k.numel() == 1:
        return x.clone()
    r = (k.numel() - 1) // 2
    c = x.shape[1]
    w = k.view(1, 1, -1).repeat(c, 1, 1)
    return F.conv1d(F.pad(x, (r, r), mode="replicate"), w, groups=c)


def rel_err(p, y):
    num = torch.sqrt(((p - y) ** 2).mean(dim=(1, 2)) + 1e-12)
    den = torch.sqrt((y ** 2).mean(dim=(1, 2)) + 1e-12)
    return num / den


def load_split(df, with_target):
    n = len(df)
    S = np.zeros((n, 4, NB, NF), dtype=np.float32)
    for i, rp in enumerate(df.support_path.values):
        S[i] = np.load(public_dir / str(rp)).astype(np.float32).reshape(4, NB, NF)
    G = np.zeros((n, 4, 3), dtype=np.float32)
    for i, js in enumerate(df.support_receivers_json.values):
        for j, o in enumerate(json.loads(js)):
            G[i, j, 0] = o["x_km"]
            G[i, j, 1] = o["y_km"]
            G[i, j, 2] = o["z_km"]
    T = df[["target_x_km", "target_y_km", "target_z_km"]].values.astype(np.float32)
    Y = None
    if with_target:
        Y = np.zeros((n, NB, NF), dtype=np.float32)
        for i, w in enumerate(df.wavefield_json.values):
            Y[i] = np.asarray(json.loads(w), dtype=np.float32).reshape(NB, NF)
    return S, G, T, Y


def geom_feats(G, T):
    d = np.linalg.norm(G - T[:, None, :], axis=2)
    dh = np.linalg.norm(G[:, :, :2] - T[:, None, :2], axis=2)
    dz = G[:, :, 2] - T[:, None, 2]
    pw = np.linalg.norm(G[:, :, None, :] - G[:, None, :, :], axis=3)
    iu = np.triu_indices(4, 1)
    pws = pw[:, iu[0], iu[1]]
    return np.concatenate(
        [
            np.log1p(d) / 4.0,
            np.log1p(dh) / 4.0,
            dz,
            (d - d.mean(axis=1, keepdims=True)) / 20.0,
            np.log1p(pws).mean(axis=1, keepdims=True) / 4.0,
            np.log1p(pws).std(axis=1, keepdims=True) / 4.0,
            np.log1p(d.mean(axis=1, keepdims=True)) / 4.0,
            np.log1p(d.min(axis=1, keepdims=True)) / 4.0,
            np.log1p(d.max(axis=1, keepdims=True)) / 4.0,
        ],
        axis=1,
    ).astype(np.float32)


def build_inputs(St, gf):
    n = St.shape[0]
    flat = St.reshape(n, 12, NF)
    mean_b = St.mean(dim=1)
    parts = [flat, mean_b, St.std(dim=1), St.max(dim=1).values, St.min(dim=1).values]
    for s in BASE_SIGMAS:
        parts.append(smooth(mean_b, s))
    return torch.cat([torch.cat(parts, dim=1), gf.unsqueeze(-1).expand(-1, -1, NF)], dim=1)


class Block(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.c1 = nn.Conv1d(c, c, 5, padding=2 * d, dilation=d)
        self.c2 = nn.Conv1d(c, c, 5, padding=2 * d, dilation=d)
        self.n1 = nn.GroupNorm(8, c)
        self.n2 = nn.GroupNorm(8, c)

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.gelu(x + h)


class Net(nn.Module):
    def __init__(self, cin, width, dil):
        super().__init__()
        self.stem = nn.Conv1d(cin, width, 7, padding=3)
        self.sn = nn.GroupNorm(8, width)
        self.blocks = nn.ModuleList([Block(width, d) for d in dil])
        self.head = nn.Conv1d(width, NB, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.gate = nn.Parameter(torch.zeros(NB, 1))

    def forward(self, x, base):
        h = F.gelu(self.sn(self.stem(x)))
        for b in self.blocks:
            h = b(h)
        return F.relu(base * torch.exp(self.gate) + self.head(h))


def augment(S, Y, gen):
    n = S.shape[0]
    sh = torch.randint(-AUG_SHIFT, AUG_SHIFT + 1, (n,), generator=gen)
    idx = (torch.arange(NF).view(1, -1) - sh.view(-1, 1)).clamp(0, NF - 1)
    i12 = idx.unsqueeze(1).expand(-1, 12, -1).contiguous()
    i3 = idx.unsqueeze(1).expand(-1, NB, -1).contiguous()
    Sa = torch.gather(S.reshape(n, 12, NF), 2, i12).reshape(n, 4, NB, NF)
    Ya = torch.gather(Y, 2, i3)
    u = torch.rand(n, 1, 1, generator=gen)
    lo = math.log(AUG_GAIN_LO)
    c = torch.exp(lo + u * (math.log(AUG_GAIN_HI) - lo))
    Sa = torch.log1p(torch.expm1(Sa.clamp(0.0, VAL_MAX)) * c.unsqueeze(1))
    Ya = torch.log1p(torch.expm1(Ya.clamp(0.0, VAL_MAX)) * c)
    a = 1.0 + AUG_BAND_SCALE * torch.randn(n, 4, NB, 1, generator=gen)
    o = AUG_BAND_OFF * torch.randn(n, 4, NB, 1, generator=gen)
    nz = AUG_NOISE * torch.randn(n, 4, NB, NF, generator=gen)
    return (Sa * a + o + nz).clamp(0.0, VAL_MAX), Ya.clamp(0.0, VAL_MAX)


def train_one(Strn, Ytrn, gftrn, cin, sig, seed):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = Net(cin, WIDTH, DILATIONS)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    n = Strn.shape[0]
    steps_per = max(1, n // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=EPOCHS * steps_per, pct_start=0.15
    )
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, generator=gen)
        for bi in range(steps_per):
            sel = perm[bi * BATCH:(bi + 1) * BATCH]
            Sa, Ya = augment(Strn[sel], Ytrn[sel], gen)
            loss = rel_err(model(build_inputs(Sa, gftrn[sel]), smooth(Sa.mean(dim=1), sig)), Ya).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    model.eval()
    return model


def predict(model, X, B):
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], 32):
            outs.append(model(X[i:i + 32], B[i:i + 32]))
    return torch.cat(outs, dim=0)


def write_submission(pred):
    arr = np.nan_to_num(pred.numpy().astype(np.float64), nan=0.0, posinf=VAL_MAX, neginf=0.0)
    arr = np.clip(np.round(arr, 4), 0.0, VAL_MAX)
    rows = [
        json.dumps([[float(v) for v in arr[i, b]] for b in range(NB)], separators=(",", ":"))
        for i in range(arr.shape[0])
    ]
    submission = pd.DataFrame({"id": test_df.id.values, "wavefield_json": rows})
    submission.to_csv(submission_out, index=False)


train = pd.read_csv(public_dir / "train.csv")
test = pd.read_csv(public_dir / "test.csv")
test_df = test

Str_np, Gtr, Ttr, Ytr_np = load_split(train, True)
Ste_np, Gte, Tte, _ = load_split(test, False)

gftr = torch.from_numpy(geom_feats(Gtr, Ttr))
gfte = torch.from_numpy(geom_feats(Gte, Tte))
Str = torch.from_numpy(Str_np)
Ytr = torch.from_numpy(Ytr_np)
Ste = torch.from_numpy(Ste_np)

mtr = Str.mean(dim=1)
SIG = SIGMA_GRID[0]
best_e = 1e9
for s in SIGMA_GRID:
    e = rel_err(smooth(mtr, s), Ytr).mean().item()
    if e < best_e:
        SIG, best_e = s, e
print("base sigma", SIG, flush=True)

Xte = build_inputs(Ste, gfte)
Bte = smooth(Ste.mean(dim=1), SIG)
CIN = Xte.shape[1]
write_submission(Bte.clamp(0.0, VAL_MAX))

groups = train.event_group.values
uniq = list(dict.fromkeys(groups))
gfold = {g: i % N_FOLDS for i, g in enumerate(uniq)}
which = np.array([gfold[g] for g in groups])

oof = torch.zeros_like(Ytr)
test_pred = torch.zeros(len(test), NB, NF)
nmodels = 0

for fi in range(N_FOLDS):
    tri_t = torch.from_numpy(np.where(which != fi)[0].astype(np.int64))
    tei_t = torch.from_numpy(np.where(which == fi)[0].astype(np.int64))
    Xva = build_inputs(Str[tei_t], gftr[tei_t])
    Bva = smooth(Str[tei_t].mean(dim=1), SIG)
    acc = torch.zeros(tei_t.numel(), NB, NF)
    for si in range(N_SEEDS):
        m = train_one(Str[tri_t], Ytr[tri_t], gftr[tri_t], CIN, SIG, SEED + 101 * si + 7 * fi)
        acc = acc + predict(m, Xva, Bva)
        test_pred = test_pred + predict(m, Xte, Bte)
        nmodels += 1
    oof[tei_t] = acc / N_SEEDS
    write_submission((test_pred / nmodels).clamp(0.0, VAL_MAX))
    print("fold", fi, "val", rel_err(oof[tei_t], Ytr[tei_t]).mean().item(), flush=True)

test_pred = test_pred / nmodels
base_oof = smooth(mtr, SIG)
best_a, best_e = 1.0, 1e9
for a in np.linspace(0.0, 1.0, 21):
    e = rel_err(float(a) * oof + float(1.0 - a) * base_oof, Ytr).mean().item()
    if e < best_e:
        best_a, best_e = float(a), e
print("net oof", rel_err(oof, Ytr).mean().item(), "base oof", rel_err(base_oof, Ytr).mean().item(), flush=True)
print("blend a", best_a, "oof", best_e, flush=True)

write_submission((best_a * test_pred + (1.0 - best_a) * Bte).clamp(0.0, VAL_MAX))
print("wrote", len(test), flush=True)
