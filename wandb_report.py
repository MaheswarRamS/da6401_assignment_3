import argparse
import warnings
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
)
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings('ignore', category=UndefinedMetricWarning)


from dataset      import Multi30kDataset, collate_fn
from lr_scheduler import NoamScheduler
from model        import Transformer, make_src_mask, make_tgt_mask
from train        import LabelSmoothingLoss, evaluate_bleu, load_checkpoint
import model as model_mod


# ══════════════════════════════════════════════════════════════════════
#  Shared config + helpers
# ══════════════════════════════════════════════════════════════════════

CONFIG = dict(
    d_model=256, N=3, num_heads=8, d_ff=1024, dropout=0.3,
    batch_size=64, num_epochs=15, warmup_steps=4000, smoothing=0.1,
)


def get_data(device):
    """Build the three splits and their DataLoaders."""
    train_ds = Multi30kDataset(split='train')
    val_ds   = Multi30kDataset(split='validation',
                               src_vocab=train_ds.src_vocab,
                               tgt_vocab=train_ds.tgt_vocab)
    test_ds  = Multi30kDataset(split='test',
                               src_vocab=train_ds.src_vocab,
                               tgt_vocab=train_ds.tgt_vocab)
    pad_idx = train_ds.tgt_vocab.pad_idx
    coll = partial(collate_fn, pad_idx=pad_idx)

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                              shuffle=True, collate_fn=coll, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG['batch_size'],
                              shuffle=False, collate_fn=coll, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=32,
                              shuffle=False, collate_fn=coll, num_workers=0, pin_memory=True)

    return (train_ds.src_vocab, train_ds.tgt_vocab, pad_idx,
            train_loader, val_loader, test_loader)


def build_model(src_vocab, tgt_vocab, device):
    """Standard Transformer factory using CONFIG."""
    return Transformer(
        src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab),
        d_model=CONFIG['d_model'], N=CONFIG['N'],
        num_heads=CONFIG['num_heads'], d_ff=CONFIG['d_ff'],
        dropout=CONFIG['dropout'], checkpoint_path=None,
    ).to(device)


def sklearn_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute accuracy, precision, recall, F1 using scikit-learn.
    Inputs: flat 1-D numpy arrays of non-pad token indices.
    Precision/recall/F1 use 'weighted' averaging (frequency-weighted).
    """
    if y_true.size == 0:
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall':    recall_score(y_true, y_pred,    average='weighted', zero_division=0),
        'f1':        f1_score(y_true, y_pred,        average='weighted', zero_division=0),
    }


def run_epoch_with_sklearn(loader, model, loss_fn, optimizer, scheduler,
                           epoch_num, is_train, device, pad_idx):
    """
    Single pass over the data. Returns (avg_loss, metrics_dict).
    Collects predictions/targets and runs sklearn metrics.
    """
    model.train() if is_train else model.eval()
    total_loss, n_batches = 0.0, 0
    all_preds, all_targets = [], []

    desc = f'Epoch {epoch_num} [{"train" if is_train else "eval"}]'
    for src, tgt in tqdm(loader, desc=desc, leave=False):
        src, tgt = src.to(device), tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        src_mask = make_src_mask(src,    pad_idx)
        tgt_mask = make_tgt_mask(tgt_in, pad_idx)

        with torch.set_grad_enabled(is_train):
            logits = model(src, tgt_in, src_mask, tgt_mask)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)),
                           tgt_out.reshape(-1))

        # Collect predictions/targets for sklearn (non-pad positions only)
        with torch.no_grad():
            preds = logits.argmax(dim=-1)               # [B, L]
            mask = (tgt_out != pad_idx)
            all_preds.append(preds[mask].cpu().numpy())
            all_targets.append(tgt_out[mask].cpu().numpy())

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    y_true = np.concatenate(all_targets) if all_targets else np.array([])
    y_pred = np.concatenate(all_preds)   if all_preds   else np.array([])
    metrics = sklearn_metrics(y_true, y_pred)
    return avg_loss, metrics


def train_and_eval(name, model, optimizer, scheduler, loss_fn,
                   train_loader, val_loader, test_loader,
                   tgt_vocab, pad_idx, device,
                   extra_per_epoch_metric=None):
    """
    Standard training loop with train/val loss + sklearn metrics per epoch.
    Final test loss, sklearn metrics, and BLEU at the end.
    """
    for epoch in range(CONFIG['num_epochs']):
        train_loss, train_m = run_epoch_with_sklearn(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch, True, device, pad_idx)
        val_loss, val_m = run_epoch_with_sklearn(
            val_loader, model, loss_fn, None, None,
            epoch, False, device, pad_idx)

        log_dict = {
            'epoch':           epoch,
            'train_loss':      train_loss,
            'train_accuracy':  train_m['accuracy'],
            'train_precision': train_m['precision'],
            'train_recall':    train_m['recall'],
            'train_f1':        train_m['f1'],
            'val_loss':        val_loss,
            'val_accuracy':    val_m['accuracy'],
            'val_precision':   val_m['precision'],
            'val_recall':      val_m['recall'],
            'val_f1':          val_m['f1'],
        }
        extras = extra_per_epoch_metric(model) if extra_per_epoch_metric is not None else {}
        log_dict.update(extras)
        wandb.log(log_dict)

        msg = (f'[{name}] ep{epoch:02d}  '
               f'tr_loss={train_loss:.3f} tr_acc={train_m["accuracy"]:.3f} '
               f'tr_f1={train_m["f1"]:.3f}  '
               f'va_loss={val_loss:.3f} va_acc={val_m["accuracy"]:.3f} '
               f'va_f1={val_m["f1"]:.3f}')
        for k, v in extras.items():
            msg += f'  {k}={v:.3f}'
        print(msg)

    # Final test metrics
    test_loss, test_m = run_epoch_with_sklearn(
        test_loader, model, loss_fn, None, None,
        CONFIG['num_epochs'], False, device, pad_idx)
    test_bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)

    wandb.log({
        'test_loss':      test_loss,
        'test_accuracy':  test_m['accuracy'],
        'test_precision': test_m['precision'],
        'test_recall':    test_m['recall'],
        'test_f1':        test_m['f1'],
        'test_bleu':      test_bleu,
    })
    print(f'[{name}] FINAL  test_loss={test_loss:.3f}  '
          f'test_acc={test_m["accuracy"]:.3f}  test_f1={test_m["f1"]:.3f}  '
          f'test_bleu={test_bleu:.2f}')


# ══════════════════════════════════════════════════════════════════════
#  2.1  Noam scheduler vs fixed learning rate
# ══════════════════════════════════════════════════════════════════════

def task_2_1(device):
    src_vocab, tgt_vocab, pad_idx, train_loader, val_loader, test_loader = get_data(device)

    for name, use_noam in [('noam', True), ('fixed_lr_1e-4', False)]:
        wandb.init(project='DA6401_A3_2.1', name=name,
                   config={**CONFIG, 'scheduler': name}, reinit=True)

        model   = build_model(src_vocab, tgt_vocab, device)
        loss_fn = LabelSmoothingLoss(len(tgt_vocab), pad_idx, CONFIG['smoothing'])

        if use_noam:
            optimizer = torch.optim.Adam(model.parameters(), lr=1.0,
                                         betas=(0.9, 0.98), eps=1e-9)
            scheduler = NoamScheduler(optimizer, CONFIG['d_model'], CONFIG['warmup_steps'])
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,
                                         betas=(0.9, 0.98), eps=1e-9)
            scheduler = None

        train_and_eval(name, model, optimizer, scheduler, loss_fn,
                       train_loader, val_loader, test_loader,
                       tgt_vocab, pad_idx, device)
        wandb.finish()


# ══════════════════════════════════════════════════════════════════════
#  2.2  Scaled vs unscaled attention + grad-norm logging
# ══════════════════════════════════════════════════════════════════════

def unscaled_attention(Q, K, V, mask=None):
    """Same as scaled_dot_product_attention but without 1/sqrt(d_k)."""
    scores = torch.matmul(Q, K.transpose(-2, -1))
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V), attn


def qk_grad_norm(model):
    """Mean L2 norm of W_q and W_k gradients across all attention blocks."""
    norms = []
    for name, p in model.named_parameters():
        if ('W_q' in name or 'W_k' in name) and p.grad is not None:
            norms.append(p.grad.norm().item())
    return float(np.mean(norms)) if norms else 0.0


def task_2_2(device):
    src_vocab, tgt_vocab, pad_idx, train_loader, val_loader, test_loader = get_data(device)

    for name, scaled in [('scaled', True), ('unscaled', False)]:
        wandb.init(project='DA6401_A3_2.2', name=name,
                   config={**CONFIG, 'scaled': scaled}, reinit=True)

        original_attn = model_mod.scaled_dot_product_attention
        if not scaled:
            model_mod.scaled_dot_product_attention = unscaled_attention

        try:
            model     = build_model(src_vocab, tgt_vocab, device)
            loss_fn   = LabelSmoothingLoss(len(tgt_vocab), pad_idx, CONFIG['smoothing'])
            optimizer = torch.optim.Adam(model.parameters(), lr=1.0,
                                         betas=(0.9, 0.98), eps=1e-9)
            scheduler = NoamScheduler(optimizer, CONFIG['d_model'], CONFIG['warmup_steps'])

            # Phase 1: log Q/K gradient norms for first 1000 steps
            model.train()
            step = 0
            for src, tgt in train_loader:
                if step >= 1000: break
                src, tgt = src.to(device), tgt.to(device)
                tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
                logits = model(src, tgt_in,
                               make_src_mask(src,    pad_idx),
                               make_tgt_mask(tgt_in, pad_idx))
                loss = loss_fn(logits.reshape(-1, logits.size(-1)),
                               tgt_out.reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                wandb.log({'step':         step,
                           'qk_grad_norm': qk_grad_norm(model),
                           'step_loss':    loss.item()})
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                step += 1

            # Phase 2: full training with sklearn metrics
            train_and_eval(name, model, optimizer, scheduler, loss_fn,
                           train_loader, val_loader, test_loader,
                           tgt_vocab, pad_idx, device)
        finally:
            model_mod.scaled_dot_product_attention = original_attn
            wandb.finish()


# ══════════════════════════════════════════════════════════════════════
#  2.3  Attention heatmaps + head specialization
# ══════════════════════════════════════════════════════════════════════

DEFAULT_SENTENCE = 'ein mann mit einem grünen hut spielt gitarre vor einem kleinen geschäft .'


def capture_last_encoder_attention(model, src, pad_idx):
    """Run encode(); stash attention weights of the LAST encoder layer."""
    captured = {}
    original = model_mod.scaled_dot_product_attention

    target = model.encoder.layers[-1].self_attn
    state  = {'inside': False}
    orig_forward = target.forward

    def wrapped_forward(query, key, value, mask=None):
        state['inside'] = True
        out = orig_forward(query, key, value, mask)
        state['inside'] = False
        return out

    def spy(Q, K, V, mask=None):
        out, attn = original(Q, K, V, mask)
        if state['inside']:
            captured['attn'] = attn.detach().cpu()
        return out, attn

    target.forward = wrapped_forward
    model_mod.scaled_dot_product_attention = spy
    try:
        model.eval()
        with torch.no_grad():
            model.encode(src, make_src_mask(src, pad_idx))
    finally:
        model_mod.scaled_dot_product_attention = original
        target.forward = orig_forward

    return captured['attn']     # [1, h, L, L]


def task_2_3(device, checkpoint_path='checkpoint.pth', sentence=DEFAULT_SENTENCE):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg  = ckpt['model_config']

    model = Transformer(
        src_vocab_size=cfg['src_vocab_size'], tgt_vocab_size=cfg['tgt_vocab_size'],
        d_model=cfg['d_model'], N=cfg['N'], num_heads=cfg['num_heads'],
        d_ff=cfg['d_ff'], dropout=cfg['dropout'], checkpoint_path=None,
    ).to(device)
    load_checkpoint(checkpoint_path, model)

    src_vocab = ckpt['src_vocab']
    pad_idx   = src_vocab.pad_idx

    wandb.init(project='DA6401_A3_2.3', name='attention_maps', config=cfg)

    tokens = sentence.split()
    ids    = [src_vocab.sos_idx] + src_vocab(tokens) + [src_vocab.eos_idx]
    src    = torch.tensor([ids], dtype=torch.long, device=device)
    labels = ['<s>'] + tokens + ['</s>']

    attn = capture_last_encoder_attention(model, src, pad_idx)
    h    = attn.size(1)

    cols = h // 2
    fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 8))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(attn[0, i].numpy(), cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'Head {i}')
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
    plt.tight_layout()
    wandb.log({'encoder_last_layer_attention': wandb.Image(fig)})
    plt.savefig('attention_heatmaps.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    rows = []
    print(f"\n{'head':>4} {'self':>8} {'next':>8} {'prev':>8} {'entropy':>8}")
    for i in range(h):
        A = attn[0, i].numpy()
        diag     = float(np.mean(np.diag(A)))
        next_tok = float(np.mean(np.diag(A,  k=1)))
        prev_tok = float(np.mean(np.diag(A,  k=-1)))
        entropy  = float(-np.sum(A * np.log(A + 1e-12), axis=-1).mean())
        rows.append([i, diag, next_tok, prev_tok, entropy])
        print(f'{i:>4} {diag:>8.3f} {next_tok:>8.3f} {prev_tok:>8.3f} {entropy:>8.3f}')

    wandb.log({'head_specialization': wandb.Table(
        columns=['head', 'self_mass', 'next_mass', 'prev_mass', 'entropy'],
        data=rows,
    )})
    wandb.finish()
    print('\nSaved heatmaps to attention_heatmaps.png and logged to W&B.')


# ══════════════════════════════════════════════════════════════════════
#  2.4  Sinusoidal vs learned positional embeddings
# ══════════════════════════════════════════════════════════════════════

class LearnedPositionalEmbedding(nn.Module):
    """Drop-in replacement for PositionalEncoding using a trainable lookup table."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.pos_embed(positions))


def task_2_4(device):
    src_vocab, tgt_vocab, pad_idx, train_loader, val_loader, test_loader = get_data(device)

    for name, use_learned in [('sinusoidal', False), ('learned', True)]:
        wandb.init(project='DA6401_A3_2.4', name=name,
                   config={**CONFIG, 'pos_encoding': name}, reinit=True)

        model = build_model(src_vocab, tgt_vocab, device)
        if use_learned:
            model.pos_enc = LearnedPositionalEmbedding(
                CONFIG['d_model'], CONFIG['dropout']).to(device)

        loss_fn   = LabelSmoothingLoss(len(tgt_vocab), pad_idx, CONFIG['smoothing'])
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0,
                                     betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, CONFIG['d_model'], CONFIG['warmup_steps'])

        train_and_eval(name, model, optimizer, scheduler, loss_fn,
                       train_loader, val_loader, test_loader,
                       tgt_vocab, pad_idx, device)
        wandb.finish()


# ══════════════════════════════════════════════════════════════════════
#  2.5  Label smoothing ε=0.1 vs ε=0.0  (+ prediction confidence)
# ══════════════════════════════════════════════════════════════════════

_PER_EPOCH_CTX = {}


def avg_confidence_full(model, loader, device, pad_idx):
    """Mean softmax probability assigned to the correct token over the loader."""
    model.eval()
    total_conf, total_tok = 0.0, 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            logits = model(src, tgt_in,
                           make_src_mask(src,    pad_idx),
                           make_tgt_mask(tgt_in, pad_idx))
            probs = F.softmax(logits, dim=-1)
            gold  = probs.gather(-1, tgt_out.unsqueeze(-1)).squeeze(-1)
            mask  = (tgt_out != pad_idx)
            total_conf += gold[mask].sum().item()
            total_tok  += mask.sum().item()
    return total_conf / max(total_tok, 1)


def task_2_5(device):
    src_vocab, tgt_vocab, pad_idx, train_loader, val_loader, test_loader = get_data(device)
    _PER_EPOCH_CTX['val_loader'] = val_loader
    _PER_EPOCH_CTX['pad_idx']    = pad_idx
    _PER_EPOCH_CTX['device']     = device

    def confidence_hook(model):
        return {'prediction_confidence': avg_confidence_full(
            model, _PER_EPOCH_CTX['val_loader'],
            _PER_EPOCH_CTX['device'], _PER_EPOCH_CTX['pad_idx'])}

    for eps in [0.1, 0.0]:
        name = f'smoothing_{eps}'
        wandb.init(project='DA6401_A3_2.5', name=name,
                   config={**CONFIG, 'smoothing': eps}, reinit=True)

        model     = build_model(src_vocab, tgt_vocab, device)
        loss_fn   = LabelSmoothingLoss(len(tgt_vocab), pad_idx, eps)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0,
                                     betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, CONFIG['d_model'], CONFIG['warmup_steps'])

        train_and_eval(name, model, optimizer, scheduler, loss_fn,
                       train_loader, val_loader, test_loader,
                       tgt_vocab, pad_idx, device,
                       extra_per_epoch_metric=confidence_hook)
        wandb.finish()


TASKS = {
    '2.1': task_2_1,
    '2.2': task_2_2,
    '2.3': task_2_3,
    '2.4': task_2_4,
    '2.5': task_2_5,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True,
                        choices=list(TASKS.keys()) + ['all'],
                        help='Which experiment to run.')
    parser.add_argument('--checkpoint', default='checkpoint.pth',
                        help='Checkpoint to use for task 2.3.')
    parser.add_argument('--sentence', default=DEFAULT_SENTENCE,
                        help='German sentence for the 2.3 attention plot.')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Override CONFIG[num_epochs] (e.g. for quick test).')
    args = parser.parse_args()

    if args.epochs is not None:
        CONFIG['num_epochs'] = args.epochs

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print(f'Config: {CONFIG}')

    if args.task == 'all':
        for t in ['2.1', '2.2', '2.3', '2.4', '2.5']:
            print(f'\n========== Running task {t} ==========')
            if t == '2.3':
                task_2_3(device, args.checkpoint, args.sentence)
            else:
                TASKS[t](device)
    elif args.task == '2.3':
        task_2_3(device, args.checkpoint, args.sentence)
    else:
        TASKS[args.task](device)


if __name__ == '__main__':
    main()