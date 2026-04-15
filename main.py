"""
LoopViT: Adaptive-depth ViT via weight-shared looping blocks.

A unit of #bpu transformer blocks is looped K times. During training, K is
sampled uniformly from {1, ..., K_max} and an output distillation loss
teaches low-K paths to match high-K outputs. At inference, K trades compute
for accuracy.

Usage:
  # LoopViT (plain+cons, recommended)
  python main.py --model continuous --block_type plain --blocks_per_unit 2 --K 4 \
      --cons_w 1.0 --cons_warmup 50 --cons_mode output --dropout 0.1 --epochs 300

  # Baseline (standard ViT)
  python main.py --model baseline --block_type plain --K 8 --dropout 0.1 --epochs 300
"""

import argparse
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms



# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        proj_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ---------------------------------------------------------------------------
# Adaptive LayerNorm (modulated by time embedding)
# ---------------------------------------------------------------------------
class AdaLN(nn.Module):
    def __init__(self, dim, t_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, 2 * dim))

    def forward(self, x, t_emb):
        sc = self.proj(t_emb).unsqueeze(1)
        scale, shift = sc.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


# ---------------------------------------------------------------------------
# Unified transformer block.
#   mode="plain":  no time conditioning, standard pre-norm residual
#   mode="direct": time-conditioned via AdaLN, standard residual
#   mode="euler":  time-conditioned via AdaLN, dt-scaled residual (velocity form)
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, t_dim=64, mode="plain",
                 dropout=0.0):
        super().__init__()
        self.mode = mode
        self.attn = Attention(dim, num_heads=n_heads,
                              attn_drop=dropout, proj_drop=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )
        if mode in ("direct", "euler"):
            self.t_embed = nn.Sequential(
                nn.Linear(2, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
            )
            self.norm1 = AdaLN(dim, t_dim)
            self.norm2 = AdaLN(dim, t_dim)
        else:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, t_start=None, t_end=None):
        if self.mode == "euler":
            B = x.shape[0]
            dt = t_end - t_start
            t_in = torch.stack([t_start, t_end]).unsqueeze(0).expand(B, 2)
            t_emb = self.t_embed(t_in)
            x = x + dt * self.attn(self.norm1(x, t_emb))
            x = x + dt * self.mlp(self.norm2(x, t_emb))
            return x
        elif self.mode == "direct":
            B = x.shape[0]
            t_in = torch.stack([t_start, t_end]).unsqueeze(0).expand(B, 2)
            t_emb = self.t_embed(t_in)
            x = x + self.attn(self.norm1(x, t_emb))
            x = x + self.mlp(self.norm2(x, t_emb))
            return x
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x


class BlockUnit(nn.Module):
    """N blocks that fire together as one 'step'."""

    def __init__(self, dim, n_heads, n_blocks=1, mlp_ratio=4.0, t_dim=64, mode="plain",
                 dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(dim, n_heads, mlp_ratio, t_dim, mode, dropout) for _ in range(n_blocks)]
        )

    def forward(self, x, t_start=None, t_end=None):
        for block in self.blocks:
            x = block(x, t_start, t_end)
        return x




# ---------------------------------------------------------------------------
# Continuous-Depth ViT
# ---------------------------------------------------------------------------
class ContinuousDepthViT(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_ch=3,
        n_cls=10,
        dim=192,
        n_heads=4,
        mlp_ratio=4.0,
        t_dim=64,
        blocks_per_unit=1,
        block_type="euler",
        dropout=0.0,
    ):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_ch, dim, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, dim) * 0.02)
        self.block = BlockUnit(dim, n_heads, blocks_per_unit, mlp_ratio, t_dim, mode=block_type, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_cls)

    def embed(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        return x + self.pos_embed

    def forward_steps(self, h, t0, t1, K, return_intermediates=False):
        """Apply shared block K times over [t0, t1].
        If return_intermediates, returns (h_final, [(t_i, h_i), ...]) with
        h_i = hidden state *before* the i-th step (i.e. at t_i).
        """
        dev = h.device
        dt = (t1 - t0) / K
        intermediates = []
        for i in range(K):
            if return_intermediates:
                intermediates.append((t0 + i * dt, h))
            ts = torch.tensor(t0 + i * dt, device=dev)
            te = torch.tensor(t0 + (i + 1) * dt, device=dev)
            h = self.block(h, ts, te)
        return (h, intermediates) if return_intermediates else h

    def classify(self, h):
        return self.head(self.norm(h[:, 0]))

    def forward(self, x, K=8):
        return self.classify(self.forward_steps(self.embed(x), 0.0, 1.0, K))


# ---------------------------------------------------------------------------
# Standard ViT baseline (independent layers, same dim)
# ---------------------------------------------------------------------------
class StandardViT(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_ch=3,
        n_cls=10,
        dim=192,
        n_heads=4,
        mlp_ratio=4.0,
        n_layers=8,
        block_type="plain",
        dropout=0.0,
    ):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_ch, dim, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [Block(dim, n_heads, mlp_ratio, mode=block_type, dropout=dropout) for _ in range(n_layers)]
        )
        self.n_layers = n_layers
        self.block_type = block_type
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_cls)

    def forward(self, x):
        B = x.shape[0]
        dev = x.device
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) + self.pos_embed
        if self.block_type == "plain":
            for blk in self.blocks:
                x = blk(x)
        else:
            dt = 1.0 / self.n_layers
            for i, blk in enumerate(self.blocks):
                ts = torch.tensor(i * dt, device=dev)
                te = torch.tensor((i + 1) * dt, device=dev)
                x = blk(x, ts, te)
        return self.head(self.norm(x[:, 0]))


# ---------------------------------------------------------------------------
# MeanFlow-style JVP consistency loss
# ---------------------------------------------------------------------------
def consistency_loss_two_path(model, intermediates):
    """
    Two-path consistency: one big step should match two smaller steps.

    Pick three grid points t_s < t_m < t_e from cached intermediates.
      - One-step:  B(h_ts, t_s, t_e)           — receives gradients
      - Two-step:  B(B(h_ts, t_s, t_m), t_m, t_e) — stop-gradiented

    All interval widths are in the regime the block is trained on.
    No JVP, no tiny-epsilon, no torch.func.
    """
    if len(intermediates) < 3:
        return torch.tensor(0.0, device=intermediates[0][1].device)

    dev = intermediates[0][1].device
    n = len(intermediates)

    # Pick three sorted indices: i < j < k
    indices = sorted(random.sample(range(n), 3))
    t_s, h_ts = intermediates[indices[0]]
    t_m, _ = intermediates[indices[1]]
    t_e, _ = intermediates[indices[2]]

    h_ts = h_ts.detach()

    # Two-step path (stop-gradiented — this is the "teacher")
    with torch.no_grad():
        h_mid = model.block(
            h_ts,
            torch.tensor(t_s, device=dev),
            torch.tensor(t_m, device=dev),
        )
        h_two = model.block(
            h_mid,
            torch.tensor(t_m, device=dev),
            torch.tensor(t_e, device=dev),
        )

    # One-step path (receives gradients — this is the "student")
    h_one = model.block(
        h_ts,
        torch.tensor(t_s, device=dev),
        torch.tensor(t_e, device=dev),
    )

    return F.mse_loss(h_one, h_two)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
DATASET_CONFIG = {
    "cifar10": {"img_size": 32, "n_cls": 10, "mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616)},
    "imagenet64": {"img_size": 64, "n_cls": 1000, "mean": (0.4815, 0.4578, 0.4082), "std": (0.2686, 0.2613, 0.2758)},
}


class HFImageDataset(Dataset):
    """Wraps a HuggingFace dataset split for use with PyTorch DataLoader."""
    def __init__(self, hf_dataset, transform=None):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, item["label"]


def get_dataloaders(dataset_name, batch_size, num_workers=4):
    cfg = DATASET_CONFIG[dataset_name]
    img_size = cfg["img_size"]
    mean, std = cfg["mean"], cfg["std"]

    if dataset_name == "cifar10":
        tfm_train = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        tfm_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_dl = DataLoader(
            datasets.CIFAR10("./data", train=True, download=True, transform=tfm_train),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
        )
        test_dl = DataLoader(
            datasets.CIFAR10("./data", train=False, download=True, transform=tfm_test),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
        )
    elif dataset_name == "imagenet64":
        from datasets import load_dataset

        tfm_train = transforms.Compose([
            transforms.RandomCrop(img_size, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        tfm_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        print("Loading ImageNet64 from HuggingFace...")
        hf_ds = load_dataset("benjamin-paine/imagenet-1k-64x64")
        train_dl = DataLoader(
            HFImageDataset(hf_ds["train"], transform=tfm_train),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True,
        )
        test_dl = DataLoader(
            HFImageDataset(hf_ds["validation"], transform=tfm_test),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=True,
        )
    return train_dl, test_dl, cfg


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", choices=["continuous", "baseline"], default="continuous"
    )
    p.add_argument(
        "--dataset", choices=["cifar10", "imagenet64"], default="cifar10",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--K", type=int, default=8)
    p.add_argument(
        "--cons_w", type=float, default=0.0, help="Weight of consistency loss"
    )
    p.add_argument(
        "--cons_warmup",
        type=int,
        default=0,
        help="Linearly ramp cons_w from 0 over this many epochs (0 = no warmup)"
    )
    p.add_argument(
        "--cons_mode",
        choices=["output", "linear", "two_path"],
        default="output",
        help="Consistency loss: 'output' (logit MSE), 'linear' (logit linearity), 'two_path' (hidden state MSE)",
    )
    p.add_argument("--dim", type=int, default=192)
    p.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    p.add_argument(
        "--blocks_per_unit",
        type=int,
        default=1,
        help="Number of shared blocks per unit (continuous model only)",
    )
    p.add_argument(
        "--block_type",
        choices=["euler", "direct", "plain"],
        default="euler",
        help="Block type: 'euler' (dt-scaled), 'direct' (time-conditioned, no dt), 'plain' (no time)",
    )
    p.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save checkpoint after training (e.g. model.pt)",
    )
    p.add_argument(
        "--load",
        type=str,
        default=None,
        help="Path to load checkpoint for eval or resume",
    )
    p.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training, run eval at multiple K values",
    )
    p.add_argument(
        "--eval_ks",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated K values to evaluate",
    )
    p.add_argument(
        "--ckpt_every",
        type=int,
        default=0,
        help="Save checkpoint every N epochs (0 = only at end)",
    )
    p.add_argument(
        "--k_schedule",
        type=str,
        default=None,
        help="Progressive depth schedule, e.g. '1:25,2:25,4:25,8:25'. "
        "Format is K:epochs pairs. Overrides --epochs and --K.",
    )
    args = p.parse_args()

    # Parse progressive depth schedule
    if args.k_schedule:
        k_phases = []
        for part in args.k_schedule.split(","):
            k_val, n_ep = part.split(":")
            k_phases.append((int(k_val), int(n_ep)))
        total_epochs = sum(n for _, n in k_phases)
        # Build epoch -> K mapping
        epoch_to_k = []
        for k_val, n_ep in k_phases:
            epoch_to_k.extend([k_val] * n_ep)
        # Override K to max for model construction and eval
        args.K = max(k for k, _ in k_phases)
        args.epochs = total_epochs
        print(f"Progressive schedule: {k_phases} ({total_epochs} total epochs)")
    else:
        epoch_to_k = None

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dl, test_dl, ds_cfg = get_dataloaders(args.dataset, args.bs)
    img_size = ds_cfg["img_size"]
    n_cls = ds_cfg["n_cls"]

    if args.model == "continuous":
        model = ContinuousDepthViT(
            img_size=img_size,
            n_cls=n_cls,
            dim=args.dim,
            n_heads=4,
            blocks_per_unit=args.blocks_per_unit,
            block_type=args.block_type,
            dropout=args.dropout,
        ).to(dev)
    else:
        model = StandardViT(img_size=img_size, n_cls=n_cls, dim=args.dim, n_heads=4, n_layers=args.K, block_type=args.block_type, dropout=args.dropout).to(dev)

    n_params = sum(p.numel() for p in model.parameters())
    bpu = args.blocks_per_unit if args.model == "continuous" else "-"
    print(
        f"Model: {args.model} | Params: {n_params:,} | K={args.K} | blocks/unit={bpu}"
    )
    print("-" * 70)

    if args.load:
        ckpt = torch.load(args.load, map_location=dev, weights_only=True)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded checkpoint from {args.load} (epoch {ckpt.get('epoch', '?')})")

    # ---- eval-only mode ----
    if args.eval_only:
        eval_ks = [int(k) for k in args.eval_ks.split(",")]
        model.eval()
        with torch.no_grad():
            for kk in eval_ks:
                t_correct = t_total = 0
                for imgs, labels in test_dl:
                    imgs, labels = imgs.to(dev), labels.to(dev)
                    if args.model == "continuous":
                        logits = model(imgs, K=kk)
                    else:
                        logits = model(imgs)
                    t_correct += (logits.argmax(1) == labels).sum().item()
                    t_total += labels.size(0)
                print(f"K={kk:3d}: {t_correct / t_total * 100:.1f}%")
        return

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot_loss = tot_task = tot_cons = correct = total = 0

        # Consistency weight with optional warmup
        if args.cons_warmup > 0:
            cons_w_eff = args.cons_w * min(1.0, epoch / args.cons_warmup)
        else:
            cons_w_eff = args.cons_w

        # Progressive schedule: fixed K for this epoch
        if epoch_to_k is not None:
            K_epoch = epoch_to_k[epoch - 1]
        else:
            K_epoch = None

        n_batches = len(train_dl)
        epoch_t0 = time.time()
        for step, (imgs, labels) in enumerate(train_dl):
            imgs, labels = imgs.to(dev), labels.to(dev)

            if args.model == "continuous":
                h0 = model.embed(imgs)

                if K_epoch is not None:
                    K_cur = K_epoch
                else:
                    # Variable depth: sample K from {1, 2, 4, K_max} each batch
                    K_choices = list(range(1, args.K + 1))
                    K_cur = random.choice(K_choices)

                if cons_w_eff > 0 and args.cons_mode == "output":
                    # Output distillation: run K_max steps, classify at both
                    # K_cur (student) and K_max (teacher). Single forward pass.
                    h_final, intermediates = model.forward_steps(
                        h0, 0.0, 1.0, args.K, return_intermediates=True,
                    )
                    # Teacher: classify at K_max (stop-grad)
                    with torch.no_grad():
                        teacher_logits = model.classify(h_final)
                    # Student: classify at K_cur (prefix of the same trajectory)
                    if K_cur < args.K:
                        _, h_student = intermediates[K_cur]  # h after K_cur steps
                        logits = model.classify(h_student)
                        cons = F.mse_loss(logits, teacher_logits)
                    else:
                        logits = model.classify(h_final)
                        cons = torch.tensor(0.0, device=dev)
                    task = F.cross_entropy(logits, labels)

                    loss = task + cons_w_eff * cons
                    tot_cons += cons.item()
                elif cons_w_eff > 0 and args.cons_mode == "linear":
                    # Logit linearity: intermediate logits should lie on a
                    # line between start and end logits.
                    h_final, intermediates = model.forward_steps(
                        h0, 0.0, 1.0, K_cur, return_intermediates=True,
                    )
                    logits = model.classify(h_final)
                    task = F.cross_entropy(logits, labels)

                    if K_cur >= 3:
                        # Pick 3 sorted grid points
                        indices = sorted(random.sample(range(len(intermediates)), 3))
                        t_s, h_s = intermediates[indices[0]]
                        t_m, h_m = intermediates[indices[1]]
                        t_e, h_e = intermediates[indices[2]]
                        # Classify all three (stop-grad on endpoints)
                        with torch.no_grad():
                            logits_s = model.classify(h_s.detach())
                            logits_e = model.classify(h_e.detach())
                        logits_m = model.classify(h_m)
                        # Linear interpolation target
                        alpha = (t_m - t_s) / (t_e - t_s)
                        target = logits_s + alpha * (logits_e - logits_s)
                        cons = F.mse_loss(logits_m, target)
                    else:
                        cons = torch.tensor(0.0, device=dev)

                    loss = task + cons_w_eff * cons
                    tot_cons += cons.item()
                elif cons_w_eff > 0:
                    # Hidden-state consistency (jvp or two_path)
                    h_final, intermediates = model.forward_steps(
                        h0, 0.0, 1.0, K_cur, return_intermediates=True,
                    )
                    logits = model.classify(h_final)
                    task = F.cross_entropy(logits, labels)

                    cons = consistency_loss_two_path(model, intermediates)
                    loss = task + cons_w_eff * cons
                    tot_cons += cons.item()
                else:
                    h_final = model.forward_steps(h0, 0.0, 1.0, K_cur)
                    logits = model.classify(h_final)
                    task = F.cross_entropy(logits, labels)
                    loss = task
            else:
                logits = model(imgs)
                task = F.cross_entropy(logits, labels)
                loss = task

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot_loss += loss.item()
            tot_task += task.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

            # Step-level logging every 100 steps
            if (step + 1) % 100 == 0:
                elapsed = time.time() - epoch_t0
                eta_epoch = elapsed / (step + 1) * (n_batches - step - 1)
                print(
                    f"  step {step+1}/{n_batches} | loss {tot_loss/(step+1):.4f} | acc {correct/total*100:.1f}% | {elapsed:.0f}s elapsed | eta {eta_epoch:.0f}s",
                    flush=True,
                )

        sched.step()

        # ---- eval ----
        model.eval()
        t_correct = t_total = 0
        with torch.no_grad():
            for imgs, labels in test_dl:
                imgs, labels = imgs.to(dev), labels.to(dev)
                if args.model == "continuous":
                    logits = model(imgs, K=args.K)
                else:
                    logits = model(imgs)
                t_correct += (logits.argmax(1) == labels).sum().item()
                t_total += labels.size(0)

        n = len(train_dl)
        log = f"Ep {epoch:3d}"
        if epoch_to_k is not None:
            log += f" K={K_epoch}"
        log += f" | loss {tot_loss / n:.4f} | task {tot_task / n:.4f}"
        if args.model == "continuous":
            log += f" | cons {tot_cons / n:.4f}"
        epoch_time = time.time() - epoch_t0
        log += f" | train {correct / total * 100:.1f}% | test {t_correct / t_total * 100:.1f}% | {epoch_time:.0f}s"
        print(log)

        # --- adaptive-step probe every 10 epochs ---
        if args.model == "continuous" and epoch % 10 == 0:
            probe_ks = [1, 2, 4]
            for kk in probe_ks:
                kc = 0
                with torch.no_grad():
                    for imgs, labels in test_dl:
                        imgs, labels = imgs.to(dev), labels.to(dev)
                        kc += (model(imgs, K=kk).argmax(1) == labels).sum().item()
                print(f"    K={kk}: {kc / t_total * 100:.1f}%")

        # ---- periodic checkpoint ----
        if args.save and args.ckpt_every > 0 and epoch % args.ckpt_every == 0:
            ckpt_path = args.save.replace(".pt", f"_ep{epoch}.pt")
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "args": vars(args)},
                ckpt_path,
            )
            print(f"Checkpoint saved to {ckpt_path}")

    # ---- save final checkpoint ----
    if args.save:
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": args.epochs,
                "args": vars(args),
            },
            args.save,
        )
        print(f"Saved checkpoint to {args.save}")


if __name__ == "__main__":
    train()
