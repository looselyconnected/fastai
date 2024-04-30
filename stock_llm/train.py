import os
import pandas as pd
import time
import math
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from typing import Tuple

import numpy as np
import torch

from model import GPTConfig, GPT
from stockdata import StockData

@dataclass
class TrainerConfig:
    # Only the ones with explicit type will be marshalled by asdict. Thus we only specify type
    # for ones that we want to log (related to model training)

    # I/O
    current_dir = os.path.dirname(os.path.realpath(__file__))
    out_dir = f"{current_dir}/out"
    eval_interval = 100
    log_interval = 1
    eval_iters = 100
    eval_only = False # if True, script exits right after the first eval
    always_save_checkpoint = False # if True, always save a checkpoint after each eval
    init_from = "" # 'scratch' or 'resume'
    # wandb logging
    wandb_log = False # disabled by default
    wandb_project = 'stock_llm'
    wandb_run_name = 'nanogpt'

    # data
    gradient_accumulation_steps: int = 5 * 8 # used to simulate larger batch sizes
    batch_size: int = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size: int = 512

    # model
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    bias: bool = False # do we use bias inside LayerNorm and Linear layers?

    # adamw optimizer
    learning_rate: float = 1e-3 # max learning rate
    max_iters: int = 10000 # total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0 # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr: bool = True # whether to decay the learning rate
    warmup_iters: int = 1000 # how many steps to warm up for
    lr_decay_iters: int = 10000 # should be ~= max_iters per Chinchilla
    min_lr: float = 5e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # system
    device: str = 'mps' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile: bool = False # use PyTorch 2.0 to compile the model to be faster

class Trainer:
    def __init__(self, cfg: TrainerConfig) -> None:
        self.config = cfg

        # we are running on a single gpu, and one process
        seed_offset = 0

        tokens_per_iter = cfg.gradient_accumulation_steps * cfg.batch_size * cfg.block_size
        print(f"tokens per iteration will be: {tokens_per_iter:,}")

        os.makedirs(cfg.out_dir, exist_ok=True)
        torch.manual_seed(1337 + seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        device_type = 'cuda' if 'cuda' in cfg.device else 'cpu' # for later use in torch.autocast
        # note: float16 data type will automatically use a GradScaler
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[cfg.dtype]
        self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
        self.iter_num = 0
        self.best_val_loss = 1e9

        # model init
        model_args = dict(n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd, block_size=cfg.block_size,
                        bias=cfg.bias, vocab_size=None, dropout=cfg.dropout)

        ckpt_path = os.path.join(cfg.out_dir, 'ckpt.pt')
        if not os.path.exists(ckpt_path):
            # init a new model from scratch
            print("Initializing a new model from scratch")
            init_from = 'scratch'
            # determine the vocab size we'll use for from-scratch training
            model_args['vocab_size'] = StockData.LABEL_COUNT + 2 # labels and 2 dividers
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)
        else:
            print(f"Resuming training from {cfg.out_dir}")
            init_from = 'resume'
            # resume training from a checkpoint.
            checkpoint = torch.load(ckpt_path, map_location=cfg.device)
            checkpoint_model_args = checkpoint['model_args']
            # force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
                model_args[k] = checkpoint_model_args[k]
            # create the model
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)
            state_dict = checkpoint['model']
            # fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
            self.iter_num = checkpoint['iter_num']
            self.best_val_loss = checkpoint['best_val_loss']

        # crop down the model block size if desired, using model surgery
        if cfg.block_size < model.config.block_size:
            model.crop_block_size(cfg.block_size)
            model_args['block_size'] = cfg.block_size # so that the checkpoint will have the right value
        model.to(cfg.device)

        # initialize a GradScaler. If enabled=False scaler is a no-op
        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.dtype == 'float16'))

        # optimizer
        self.optimizer = model.configure_optimizers(cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2), device_type)
        if init_from == 'resume':
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        checkpoint = None # free up memory

        # compile the model
        if cfg.compile:
            print("compiling the model... (takes a ~minute)")
            unoptimized_model = model
            model = torch.compile(model) # requires PyTorch 2.0
        self.model = model
        self.model_args = model_args

        # logging
        if cfg.wandb_log:
            import wandb
            wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=asdict(cfg))

        train_val_data = np.load(f"{cfg.current_dir}/data/train_eval.npz")
        self.train_data = train_val_data['train']
        self.val_data = train_val_data['val']

    def get_batch(self, data: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        device = cfg.device
        ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+cfg.block_size])) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+cfg.block_size])) for i in ix])
        if 'cuda' in device:
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for (split, data) in [("train", self.train_data), ("val", self.val_data)]:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                X, Y = self.get_batch(data)
                with self.ctx:
                    logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, it):
        cfg = self.config
        # 1) linear warmup for warmup_iters steps
        if it < cfg.warmup_iters:
            return cfg.learning_rate * it / cfg.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > cfg.lr_decay_iters:
            return cfg.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

    def run(self):
        cfg = self.config
        # training loop
        X, Y = self.get_batch(self.train_data) # fetch the very first batch
        t0 = time.time()
        local_iter_num = 0 # number of iterations in the lifetime of this process
        iter_num = self.iter_num
        best_val_loss = self.best_val_loss

        raw_model = self.model
        running_mfu = -1.0
        while True:
            # determine and set the learning rate for this iteration
            lr = self.get_lr(iter_num) if cfg.decay_lr else cfg.learning_rate
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if iter_num % cfg.eval_interval == 0:
                losses = self.estimate_loss()
                print(f"step {iter_num}: lr {lr} train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                if cfg.wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "train/loss": losses['train'],
                        "val/loss": losses['val'],
                        "lr": lr,
                        "mfu": running_mfu*100, # convert to percentage
                    })
                if losses['val'] < best_val_loss or cfg.always_save_checkpoint:
                    best_val_loss = losses['val']
                    if iter_num > 0:
                        checkpoint = {
                            'model': raw_model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'model_args': self.model_args,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': self.model_args,
                        }
                        print(f"saving checkpoint to {cfg.out_dir}")
                        torch.save(checkpoint, os.path.join(cfg.out_dir, 'ckpt.pt'))
            if cfg.eval_only:
                break

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(cfg.gradient_accumulation_steps):
                with self.ctx:
                    logits, loss = self.model(X, Y)
                    loss = loss / cfg.gradient_accumulation_steps # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y = self.get_batch(self.train_data)
                # backward pass, with gradient scaling if training in fp16
                self.scaler.scale(loss).backward()
            # clip the gradient
            if cfg.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            # step the optimizer and scaler if training in fp16
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            self.optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % cfg.log_interval == 0:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * cfg.gradient_accumulation_steps
                if local_iter_num >= 5: # let the training loop settle a bit
                    mfu = raw_model.estimate_mfu(cfg.batch_size * cfg.gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            iter_num += 1
            local_iter_num += 1

            # termination conditions
            if iter_num > cfg.max_iters:
                break


if __name__ == "__main__":
    config = TrainerConfig()
    config.n_head = 4
    config.n_layer = 4
    config.n_embd = 128
    config.block_size = 512
    config.device = "mps"
    config.compile = False
    config.eval_only = False
    config.learning_rate = 1e-5
    config.min_lr = config.learning_rate / 10
    
    trainer = Trainer(config)
    trainer.run()