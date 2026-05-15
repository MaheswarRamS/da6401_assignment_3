import math
from collections import Counter
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from dataset import Multi30kDataset, collate_fn
from lr_scheduler import NoamScheduler
from model import Transformer, make_src_mask, make_tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        
        log_prob = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            true_dist = torch.full_like(
                log_prob, self.smoothing / (self.vocab_size -1)
            )
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.pad_idx] = 0.0
            pad_rows = (target == self.pad_idx)
            true_dist.masked_fill_(pad_rows.unsqueeze(1), 0.0)
        loss = -(true_dist * log_prob).sum(dim = -1)
        n_tok = (~pad_rows).sum().clamp(min=1)
        return loss.sum() / n_tok

# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
    pad_idx: int =1,
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).

    """
    model.train() if is_train else model.eval()

    total_loss, n_batches = 0.0, 0
    desc = f"Epoch {epoch_num} [{'train' if is_train else 'val'}]"

    for src, tgt in tqdm(data_iter, desc=desc, leave =False):
        src, tgt = src.to(device), tgt.to(device)

        tgt_in, tgt_out = tgt[:, : -1], tgt[:,1:]

        src_mask = make_src_mask(src, pad_idx)
        tgt_mask = make_tgt_mask(tgt_in, pad_idx)
           
        with torch.set_grad_enabled(is_train):
            logits = model(src, tgt_in, src_mask, tgt_mask)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)),tgt_out.reshape(-1))

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    # TODO: Task 3.3 — implement token-by-token greedy decoding
    model.eval()
    with torch.no_grad():
        memory = model.encode(src,src_mask)
        ys = torch.tensor([[start_symbol]], dtype =torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=1)
            logits = model.decode(memory, src_mask, ys, tgt_mask)
            next_tok = logits[:, -1, :].argmax(dim = -1, keepdim =True)
            ys = torch.cat([ys, next_tok], dim = 1)
            if next_tok.item() == end_symbol:
                break
        
        return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def _ngrams(tokens: List[str], n:int):
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) -n + 1)]

def corpus_bleu(
        references: List[List[str]],
        candidates: List[List[str]],
        max_n: int = 4,
) -> float:
    """ Return corpus bleu between 1 to 100 """
    clipped = [0] * max_n
    totals = [0] * max_n
    ref_len, cand_len = 0, 0

    for ref, cand in zip(references, candidates):
        ref_len += len(ref)
        cand_len += len(cand)
        for n in range(1, max_n + 1):
            cand_ng = Counter(_ngrams(cand,n))
            ref_ng = Counter(_ngrams(ref,n))
            for ng, c in cand_ng.items():
                clipped[n-1] += min(c, ref_ng[ng])
            totals[n-1] += max(len(cand) -n + 1, 0)
    if any(c==0 for c in clipped) or any(t==0 for t in totals):
        return 0.0
    
    precision = [c / t for c, t in zip(clipped, totals)]
    geo_mean = math.exp(sum(math.log(p) for p in precision) / max_n)
    bp = 1.0 if cand_len > ref_len else math.exp(1 - ref_len / max(cand_len, 1))
    return bp * geo_mean * 100

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).

    """
    # TODO: Task 3 — loop test set, decode, compute and return BLEU
    model.eval()
    sos_idx = tgt_vocab.stoi['<sos>']
    eos_idx = tgt_vocab.stoi['<eos>']
    pad_idx = tgt_vocab.stoi['<pad>']
    
    def ids_to_tokens(ids:List[int]) -> List[str]:
        out = []
        for i in ids:
            if i == eos_idx:
                break
            if i in (sos_idx, pad_idx):
                continue
            out.append(tgt_vocab.lookup_token(i))
        return out
    refs, hyps = [], []
    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc='BLEU', leave = False):
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            for b in range(src.size(0)):
                src_i = src[b: b + 1]
                src_mask = make_src_mask(src_i, pad_idx)
                ys = greedy_decode(
                    model, src_i, src_mask, max_len, sos_idx,eos_idx,device,
                )
                hyps.append(ids_to_tokens(ys[0].tolist()))
                refs.append(ids_to_tokens(tgt[b].tolist()))
    return corpus_bleu(refs, hyps)

# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
    src_vocab = None,
    tgt_vocab = None,
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'

    model_config must contain all kwargs needed to reconstruct
    Transformer(**model_config), e.g.:
        {'src_vocab_size': ..., 'tgt_vocab_size': ...,
         'd_model': ..., 'N': ..., 'num_heads': ...,
         'd_ff': ..., 'dropout': ...}
    """
    # TODO: implement using torch.save({...}, path)
    torch.save(
        {
            'epoch':   epoch,
            'model_state_dict':    model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'model_config'        : getattr(model, 'config', {}),
            'src_vocab'           : src_vocab if src_vocab is not None else getattr(model, 'src_vocab', None),
            'tgt_vocab'           : tgt_vocab if tgt_vocab is not None else getattr(model, 'tgt_vocab', None),
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).

    """
    # TODO: implement restore logic
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt.get('epoch', 0)

# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """
    # TODO: implement full experiment
    config = {
        'd_model'   : 256,
        'N'         : 3,
        'num_heads' : 8,
        'd_ff'      : 1024,
        'dropout'   : 0.3 ,
        'batch_size': 64,
        'num_epochs': 100,
        'warmup_steps' : 4000,
        'smoothing' : 0.2, 
        }
    
    wandb.init(project = 'DA6401_A3', config=config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data loading
    train_ds = Multi30kDataset(split='train')
    val_ds = Multi30kDataset(split='validation', src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab)
    test_ds = Multi30kDataset(split='test', src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab) 
    src_vocab, tgt_vocab = train_ds.src_vocab, train_ds.tgt_vocab
    pad_idx = tgt_vocab.pad_idx
    collate = lambda b: collate_fn(b, pad_idx=pad_idx)

    train_loader = DataLoader(train_ds, batch_size = config['batch_size'], shuffle =True, collate_fn=collate, num_workers= 0, pin_memory =True, persistent_workers = False)
    val_loader = DataLoader(val_ds, batch_size = config['batch_size'], shuffle =False, collate_fn=collate, num_workers= 0, pin_memory =True, persistent_workers = False)
    test_loader = DataLoader(test_ds, batch_size = 32, shuffle =False, collate_fn=collate, num_workers= 0, pin_memory =True)
    
    # Model, Loss, Optimizer
    model = Transformer(
        src_vocab_size= len(src_vocab),
        tgt_vocab_size= len(tgt_vocab),
        d_model = config['d_model'],
        N = config['N'],
        num_heads=config['num_heads'],
        d_ff= config['d_ff'],
        dropout= config['dropout'],
        ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas =(0.9,0.98), eps=1e-9)
         
    scheduler = NoamScheduler(
        optimizer,d_model=config['d_model'], warmup_steps=config['warmup_steps'])
    
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),pad_idx=pad_idx, smoothing=config['smoothing'])

    # Train 
    best_val_bleu = -1.0
    best_val_loss_for_bleu = float('inf')
    best_val = float('inf')
    bleu_tie_eps = 1e-6

    for epoch in range(config['num_epochs']):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler, epoch, True, device, pad_idx)
        val_loss = run_epoch(val_loader, model, loss_fn, None, None, epoch, False, device, pad_idx)
        val_bleu = evaluate_bleu(model, val_loader, tgt_vocab, device=device)
        wandb.log({'Epoch': epoch, 'Train_loss': train_loss, 'Val_loss': val_loss, 'Val_bleu': val_bleu})
        print(f'Epoch:{epoch:02d} Train_loss = {train_loss:.3f} Val_loss = {val_loss:.3f} Val_bleu = {val_bleu:.3f}') 
        
        if val_loss< best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, 'checkpoint.pth',src_vocab=src_vocab,tgt_vocab=tgt_vocab)
            print(f'Model with best val loss saved')
        
        improved_bleu = val_bleu > best_val_bleu + bleu_tie_eps
        tied_bleu_lower_loss = (
            abs(val_bleu - best_val_bleu) <= bleu_tie_eps
            and val_loss < best_val_loss_for_bleu
        )

        if improved_bleu or tied_bleu_lower_loss:
            best_val_bleu = val_bleu
            best_val_loss_for_bleu = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                'checkpoint.pth',
                src_vocab=src_vocab, tgt_vocab=tgt_vocab
            )
            print(
                'Best combined checkpoint saved as checkpoint.pth '
                f'(Val_bleu={best_val_bleu:.3f}, Val_loss={best_val_loss_for_bleu:.3f})'
            )

    # Final bleu
    load_checkpoint('checkpoint.pth', model)
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    wandb.log({'Test_bleu': bleu,'Best_val_bleu': best_val_bleu})
    print(f'Test_bleu: {bleu:.3f}') 
    wandb.finish()

if __name__ == "__main__":
    run_training_experiment()
