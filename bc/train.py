import argparse, h5py, numpy as np, torch, torch.nn as nn
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--out", required=True)
p.add_argument("--epochs", type=int, default=100)
args = p.parse_args()
torch.manual_seed(0); np.random.seed(0)

# ---- load episodes, split BY EPISODE (never by timestep: that leaks data) ----
with h5py.File(args.data) as f:
    eps = [(f[k]["obs"][:], f[k]["actions"][:]) for k in sorted(f.keys())]
idx = np.random.permutation(len(eps))
n_val = max(1, int(0.2 * len(eps)))
val_eps = [eps[i] for i in idx[:n_val]]
tr_eps = [eps[i] for i in idx[n_val:]]
cat = lambda E, j: np.concatenate([e[j] for e in E]).astype(np.float32)
Xtr, Ytr, Xva, Yva = cat(tr_eps, 0), cat(tr_eps, 1), cat(val_eps, 0), cat(val_eps, 1)
print(f"train: {len(tr_eps)} eps / {len(Xtr)} samples | val: {len(val_eps)} eps / {len(Xva)} samples")

# ---- normalize observations with TRAIN statistics only ----
mean, std = Xtr.mean(0), Xtr.std(0) + 1e-6
norm = lambda X: (X - mean) / std
Xtr_t, Xva_t = torch.tensor(norm(Xtr)), torch.tensor(norm(Xva))
Ytr_t, Yva_t = torch.tensor(Ytr), torch.tensor(Yva)

# ---- model: small MLP, tanh output because actions live in [-1, 1] ----
model = nn.Sequential(
    nn.Linear(Xtr.shape[1], 256), nn.ReLU(),
    nn.Linear(256, 256), nn.ReLU(),
    nn.Linear(256, 256), nn.ReLU(),
    nn.Linear(256, Ytr.shape[1]), nn.Tanh())
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
loss_fn = nn.MSELoss()

best = 1e9
for epoch in range(args.epochs):
    model.train()
    perm = torch.randperm(len(Xtr_t))
    for i in range(0, len(perm), 256):
        b = perm[i:i+256]
        loss = loss_fn(model(Xtr_t[b]), Ytr_t[b])
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()
    model.eval()
    with torch.no_grad():
        tr = loss_fn(model(Xtr_t), Ytr_t).item()
        va = loss_fn(model(Xva_t), Yva_t).item()
    if va < best:
        best = va
        torch.save({"model": model.state_dict(), "mean": mean, "std": std}, args.out)
    if epoch % 10 == 0 or epoch == args.epochs - 1:
        print(f"epoch {epoch:3d} train {tr:.4f} val {va:.4f}")
print(f"best val loss {best:.4f} -> {args.out}")
