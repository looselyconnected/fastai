import os
import pandas as pd
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch

from model import GPTConfig, GPT
from data import data_columns, get_data_for_eval, decode_data, encode_data

ticker = 'SPY'
currentDir = os.path.dirname(os.path.realpath(__file__))

# -----------------------------------------------------------------------------
# configs
# I/O
out_dir = 'out'

# system
device = 'mps' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

# various inits, derived attributes, I/O setup
seed_offset = 0

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model init
model_args = dict()

ckpt_path = os.path.join(out_dir, 'ckpt.pt')
if not os.path.exists(ckpt_path):
    print("can't find checkpoint file: " + ckpt_path)
    exit(1)

# resume training from a checkpoint.
checkpoint = torch.load(ckpt_path, map_location=device)
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

model.to(device)
model.eval()

checkpoint = None # free up memory

cutoff_date = "2024-03-01"
all_data_df = get_data_for_eval(ticker, data_dir=f"{currentDir}/data")
context_df = all_data_df[all_data_df.Date <= cutoff_date]
context = encode_data(context_df)
y = model.generate(context.to(device), max_new_tokens=30*len(data_columns), temperature=0.5)
pred = decode_data(y) # pred includes all the context

# for dates > cutoff_date, calculate the delta of all_data_df and df for the data_columns
delta = pred[data_columns] - all_data_df.iloc[:len(pred)][data_columns]
print(delta)