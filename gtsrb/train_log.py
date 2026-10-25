import argparse, json, os, ssl
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import poison as P

# Disable SSL verification for dataset downloads on macOS
ssl._create_default_https_context = ssl._create_unverified_context

def load_dataset(name, root='./data', n_synth=4000):
    if name == 'synth':
        rng = np.random.default_rng(0)
        C, H, W, K = 1, 16, 16, 4
        templates = rng.standard_normal((K, C, H, W)).astype(np.float32)
        templates = templates / np.abs(templates).max()
        y = rng.integers(0, K, n_synth)
        X = templates[y] + 0.35 * rng.standard_normal((n_synth, C, H, W)).astype(np.float32)
        X = (X - X.min()) / (X.max() - X.min())
        ytr, Xtr = y[:int(.8 * n_synth)], X[:int(.8 * n_synth)]
        yte, Xte = y[int(.8 * n_synth):], X[int(.8 * n_synth):]
        return Xtr, ytr, Xte, yte, K
    from torchvision import datasets
    if name == 'mnist':
        tr = datasets.MNIST(root, train=True, download=True)
        te = datasets.MNIST(root, train=False, download=True)
        Xtr = tr.data.numpy()[:, None].astype(np.float32) / 255.0
        Xte = te.data.numpy()[:, None].astype(np.float32) / 255.0
        return Xtr, tr.targets.numpy(), Xte, te.targets.numpy(), 10
    if name == 'cifar10':
        tr = datasets.CIFAR10(root, train=True, download=True)
        te = datasets.CIFAR10(root, train=False, download=True)
        Xtr = tr.data.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        Xte = te.data.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        return Xtr, np.array(tr.targets), Xte, np.array(te.targets), 10
    if name == 'cifar100':
        tr = datasets.CIFAR100(root, train=True, download=True)
        te = datasets.CIFAR100(root, train=False, download=True)
        Xtr = tr.data.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        Xte = te.data.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        return Xtr, np.array(tr.targets), Xte, np.array(te.targets), 100
    if name == 'gtsrb':
        # GTSRB images are variable-sized PIL images; resize all to 32x32 to match
        # the 32x32 pipeline (same spatial size as CIFAR), keep 3 colour channels.
        from PIL import Image
        def to_array(ds):
            X = np.zeros((len(ds), 3, 32, 32), np.float32)
            y = np.zeros(len(ds), np.int64)
            for i in range(len(ds)):
                img, lab = ds[i]                       # img is a PIL.Image
                img = img.convert('RGB').resize((32, 32))
                X[i] = np.asarray(img, np.float32).transpose(2, 0, 1) / 255.0
                y[i] = lab
            return X, y
        tr = datasets.GTSRB(root, split='train', download=True)
        te = datasets.GTSRB(root, split='test', download=True)
        Xtr, ytr = to_array(tr)
        Xte, yte = to_array(te)
        return Xtr, ytr, Xte, yte, 43
    raise ValueError(name)

class CNN(nn.Module):
    def __init__(self, c_in, n_cls, w=32):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(c_in, w, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(w, 2 * w, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(2), nn.Flatten())
        self.head = nn.Linear(2 * w * 4, n_cls)
    def features(self, x): return self.f(x)
    def forward(self, x): return self.head(self.f(x))

class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False)
    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut

class PreActResNet18(nn.Module):
    def __init__(self, c_in, n_cls, w=64):
        super().__init__()
        self.in_planes = w
        self.conv1 = nn.Conv2d(c_in, w, 3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(w, 2, 1)
        self.layer2 = self._make_layer(2 * w, 2, 2)
        self.layer3 = self._make_layer(4 * w, 2, 2)
        self.layer4 = self._make_layer(8 * w, 2, 2)
        self.bn = nn.BatchNorm2d(8 * w)
        self.head = nn.Linear(8 * w, n_cls)
    def _make_layer(self, planes, n_blocks, stride):
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []
        for s in strides:
            layers.append(PreActBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)
    def features(self, x):
        out = self.conv1(x)
        out = self.layer1(out); out = self.layer2(out)
        out = self.layer3(out); out = self.layer4(out)
        out = F.relu(self.bn(out))
        return F.adaptive_avg_pool2d(out, 1).flatten(1)
    def forward(self, x): return self.head(self.features(x))

_VGG19_CFG = [64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M']
class VGG19BN(nn.Module):
    def __init__(self, c_in, n_cls):
        super().__init__()
        layers = []
        in_c = c_in
        for v in _VGG19_CFG:
            if v == 'M':
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers += [nn.Conv2d(in_c, v, 3, padding=1, bias=False),
                           nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_c = v
        self.f = nn.Sequential(*layers)
        self.head = nn.Linear(512, n_cls)
    def features(self, x): return self.f(x).flatten(1)
    def forward(self, x): return self.head(self.features(x))

@torch.no_grad()
def probe(model, X, y, device, bs=512):
    model.eval()
    L, C = [], []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        yb = torch.from_numpy(y[i:i + bs]).long().to(device)
        logits = model(xb)
        L.append(F.cross_entropy(logits, yb, reduction='none').cpu().numpy())
        C.append(torch.softmax(logits, 1).gather(1, yb[:, None])[:, 0].cpu().numpy())
    model.train()
    return np.concatenate(L), np.concatenate(C)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='synth', choices=['synth', 'mnist', 'cifar10', 'cifar100', 'gtsrb'])
    ap.add_argument('--attack', default='badnets', choices=list(P.ATTACKS) + ['none'])
    ap.add_argument('--rate', type=float, default=0.05)
    ap.add_argument('--target', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--probe_every', type=int, default=20)
    ap.add_argument('--adaptive_shape', action='store_true')
    ap.add_argument('--adaptive', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='dynamics.npz')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--model', type=str, default='cnn', choices=['cnn', 'preactresnet18', 'vgg19bn'])
    a = ap.parse_args()

    dev = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') if a.device == 'auto' else a.device
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    Xtr, ytr, Xte, yte, K = load_dataset(a.dataset)
    if a.attack == 'none': m = np.zeros(len(Xtr), bool)
    elif a.attack == 'label_flip': Xtr, ytr, m = P.label_flip(Xtr, ytr, rate=a.rate, seed=a.seed, n_classes=K)
    else: Xtr, ytr, m = P.ATTACKS[a.attack](Xtr, ytr, target=a.target, rate=a.rate, seed=a.seed)
    print(f"device={dev}  dataset={a.dataset}  attack={a.attack}  poisoned {m.sum()}/{len(m)} ({100*m.mean():.1f}%)  classes={K}")

    if a.model == 'cnn': model = CNN(Xtr.shape[1], K).to(dev)
    elif a.model == 'preactresnet18': model = PreActResNet18(Xtr.shape[1], K).to(dev)
    elif a.model == 'vgg19bn': model = VGG19BN(Xtr.shape[1], K).to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    g = torch.Generator().manual_seed(a.seed)

    L_hist, C_hist, steps = [], [], []
    step = 0
    w_poison = 1.0
    m_t = torch.from_numpy(m.astype(np.float32)).to(dev)
    w_i = np.ones(len(Xtr), np.float32)
    twin = np.zeros(len(Xtr), int)
    if a.adaptive_shape and m.sum() > 0:
        twin[m] = np.random.default_rng(a.seed).choice(np.where(~m)[0], m.sum(), replace=True)
    w_t = torch.from_numpy(w_i).to(dev)
    l0, c0 = probe(model, Xtr, ytr, dev)
    L_hist.append(l0); C_hist.append(c0); steps.append(0)

    for ep in range(a.epochs):
        perm = torch.randperm(len(Xtr), generator=g).numpy()
        for i in range(0, len(perm) - a.bs + 1, a.bs):
            b = perm[i:i + a.bs]
            xb, yb = torch.from_numpy(Xtr[b]).to(dev), torch.from_numpy(ytr[b]).long().to(dev)
            opt.zero_grad()
            per = F.cross_entropy(model(xb), yb, reduction='none')
            if a.adaptive_shape: (per * w_t[b]).mean().backward()
            elif a.adaptive:
                wb = torch.where(m_t[b] > 0.5, torch.tensor(w_poison, device=dev), torch.tensor(1.0, device=dev))
                (per * wb).mean().backward()
            else: per.mean().backward()
            opt.step()
            step += 1
            if step % a.probe_every == 0:
                l, c = probe(model, Xtr, ytr, dev)
                L_hist.append(l); C_hist.append(c); steps.append(step)
                if a.adaptive and m.sum() > 0:
                    w_poison = float(np.clip(w_poison + 2.0 * (l[m].mean() - l[~m].mean()), 0.02, 2.0))
                if a.adaptive_shape and m.sum() > 0:
                    w_i[m] = np.clip(w_i[m] + 3.0 * (l[m] - l[twin[m]]), 0.01, 3.0)
                    w_t = torch.from_numpy(w_i).to(dev)
        print(f"  epoch {ep+1}/{a.epochs}  checkpoints so far {len(steps)}")

    def predict(X, bs=256):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                out.append(model(torch.from_numpy(X[i:i + bs]).to(dev)).argmax(1).cpu().numpy())
        return np.concatenate(out)

    clean_acc = float((predict(Xte) == yte).mean())
    asr = float('nan')
    if a.attack in P.TRIGGER_ATTACKS:
        Xt, yt = P.trigger_test_set(Xte, yte, target=a.target, kind=a.attack, seed=a.seed)
        asr = float((predict(Xt) == yt).mean())
    print(f"  clean_acc={clean_acc:.4f}  asr={asr}")

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    np.savez_compressed(a.out, loss=np.stack(L_hist, 1), conf=np.stack(C_hist, 1),
                        y=ytr, is_poisoned=m, steps=np.array(steps), adaptive=bool(a.adaptive),
                        config=json.dumps(vars(a)), clean_acc=clean_acc, asr=asr)
    print(f"wrote {a.out}")

if __name__ == '__main__': main()