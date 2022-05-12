#!/usr/bin/env python
# coding: utf-8

# To run distributed:
# 
# ```
# make
# ```
# 
# ```
# CUDA_VISIBLE_DEVICES=3,4,5,1,2,0 python -m torch.distributed.launch --master_port 1235 --nproc_per_node=6 02_train.py --epochs 30 --bs 96 --fp16 to_fp16 --trf_heads 4 --mixup False --chunk_size 500 --trf_dim 512 --loss ce --n_chunks 1 --fit fit_flat_cos --fit_kwargs pct_start=0.5 div_final=100 --tfixup True --pad r --valid_pct 0.025 --trf_act gelu --opt ranger_lamb --lr 3e-3
# ```

# In[1]:


from fastai.basics       import *
from fastai.callback.all import *
from fastai.distributed  import *
from fastai.tabular.all  import *
from fastai.test_utils   import *

import ast
import enum
import gc
import pandas as pd
import pickle
import enum

from collections import defaultdict
from fastcore.script import *
from matplotlib import pyplot as plt
from pathlib import Path
from pytorch_block_sparse.util import ModelPatcher
from sklearn.metrics import roc_auc_score
from torch.distributions.beta import Beta
from torch.utils.data import Dataset


# In[2]:


in_d = Path('../input')


# In[3]:


@call_parse
def main(
    model:         Param("Name", str) = '210105',
    data:          Param("Data version", str) = '210101b',
    load:          Param("Load from", str) = None,
    validate:      Param("", action='store_true') = False,
    chunk_size:    Param("Chunk size", int) = 500,
    n_chunks:      Param("Number of chunks", int) = 1,
    bs:            Param("BS", int) = 96,
    workers:       Param("", int) = 8,
    valid_pct:     Param("Validation set", float) = 0.025,
    trf_dim:       Param("", int) = 512,
    trf_enc:       Param("", int) = 4,
    trf_dec:       Param("", int) = 4,
    trf_heads:     Param("", int) = 4,
    trf_do:        Param("", float) = 0.1,
    trf_act:       Param("", str) = 'gelu',
    lr:            Param("", float) = 3e-3,
    clip:          Param("", float) = 0.,
    
    moms:          Param("Moms for fit_one_cycle", float, nargs='+') = (0.95,0.85,0.95),
    epochs:        Param("Epochs", int) = 30,
    tfixup:        Param("Use T-Fixup init", ast.literal_eval) = True,
    mixup:         Param("Use mixup", ast.literal_eval) = False,
    opt:           Param("Optimizer", str) = 'ranger_lamb',
    opt_kwargs:    Param("Optional args for opt, eg. eps=1e-4", str, nargs='+') = {},
    fit:           Param("fit or fit_one_cycle", str) = 'fit_flat_cos',
    fit_kwargs:    Param("Optional args for fit,eg pct_start=0.1", str, nargs='+') = ['pct_start=0.5', 'div_final=100.'],
    fp16:          Param("fp16 method: to_fp16, to_native_fp16, none", str) = 'to_fp16',
    
    loss:          Param("Loss", str) = 'ce',
    
    wua:           Param("Weight of user_answer term in the loss", float) = 0.,
    pad:           Param ("Pad left of right (l|r)",str,choices=['l','r'])='r',

    local_rank:    Param("--local_rank", int) = None,
): 
    if opt_kwargs: opt_kwargs = {s.split('=')[0]:float(s.split('=')[1]) for s in opt_kwargs}
    if fit_kwargs: fit_kwargs = {s.split('=')[0]:float(s.split('=')[1]) for s in fit_kwargs}
    print(locals())
    globals().update({ 'H' : AttrDict(locals())})
_H = AttrDict


# In[5]:


n_gpus = torch.cuda.device_count() if H.local_rank is None else 1
if H.local_rank is not None:
    torch.cuda.set_device(H.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    print(f"DISTRIBUTED: {H.local_rank}")


# # Read df and meta

# In[6]:


with open(in_d / f'meta_v{H.data}.pkl', 'rb') as f:
    meta = pickle.load(f)


# In[7]:


QCols = enum.IntEnum('QCols', meta.qcols, start=0)
LCols = enum.IntEnum('LCols', meta.lcols, start=0)
Cats  = enum.IntEnum('Cats',  meta.cat_names, start=0)
Conts = enum.IntEnum('Conts', meta.cont_names, start=0)


# In[8]:


#%%time
with open(in_d / f'data_v{H.data}.pkl', 'rb') as f:
    data = pickle.load(f)


# In[9]:


del data.attempt_num_coo
del data.attempts_correct_coo
gc.collect()


# # Y

# In[10]:


lut = meta.icats['answered_correctly'],meta.icats['user_answer']
y_d = {}
for k, v in data.cat_d.items():
    y_d[k] = np.column_stack((lut[0][v[:,Cats.answered_correctly] - 1],lut[1][v[:,Cats.user_answer] - 1]))


# # Chop sequences

# In[11]:


def chop_sequence(d):
    nv = defaultdict(dict)
    for k, v in d.items():
        i = 0
        while i*H.chunk_size < len(v):
            nv[k][i] = v[i*H.chunk_size:(i+1)*H.chunk_size]
            i += 1
    return nv


# In[12]:


cat_d  = chop_sequence(data.cat_d)
cont_d = chop_sequence(data.cont_d)
tags_d = chop_sequence(data.tags_d)
tagw_d = chop_sequence(data.tagw_d)
y_d    = chop_sequence(y_d)


# In[13]:


#assert np.concatenate(list(cat_d.values())).shape[0] == np.concatenate(list(data.cat_d.values())).shape[0]


# In[14]:


print(f'There are {len(data.cat_d)} different users')


# # Train/valid split

# In[15]:


group_keys = sorted(list(cat_d.keys()))


# In[16]:


# Last H.valid_pct is valid set
train_group_keys = group_keys[:int((1 - H.valid_pct) * len(group_keys))]
valid_group_keys = group_keys[int((1 - H.valid_pct) * len(group_keys)):]


# In[17]:


print(f'users: train={len(train_group_keys)}, valid={len(valid_group_keys)}')


# # Data

# ## To dicts

# In[18]:


def split_dict(d, keys):
    return { (u, t): d[u][t] for u in keys for t in d[u].keys() }


# In[19]:


train_x_cat =  split_dict(cat_d, train_group_keys)
train_x_cont = split_dict(cont_d, train_group_keys)
train_x_tags = split_dict(tags_d, train_group_keys)
train_x_tagw = split_dict(tagw_d, train_group_keys)
train_y =      split_dict(y_d, train_group_keys)


# In[20]:


valid_x_cat =  split_dict(cat_d, valid_group_keys)
valid_x_cont = split_dict(cont_d, valid_group_keys)
valid_x_tags = split_dict(tags_d, valid_group_keys)
valid_x_tagw = split_dict(tagw_d, valid_group_keys)
valid_y =      split_dict(y_d, valid_group_keys)


# In[21]:


print(f'seqs: train={len(train_x_cat)}, valid={len(valid_x_cat)}')


# # Dataset

# In[22]:


class InteractionsDataset(Dataset):
    def __init__(self, x_cat, x_cont, x_tags, x_tagw, y, minids=False):
        super().__init__()
        
        self.means = np.expand_dims(meta.means, axis=0) # ready to broadcast
        self.stds  = np.expand_dims(meta.stds , axis=0)
        
        self.n_inp = 5  # number of feature (x) tensors
        
        self.x_cat = x_cat  # SL, XF (sequence len, feature columns) 
        self.x_cont = x_cont
        self.x_tags = x_tags      
        self.x_tagw = x_tagw
        self.y = y  # SL, 1
        
        self.keys = list(self.x_cat.keys()) # list of group keys
        
        if minids:
            self.keys = self.keys[:H.bs*2]

    def __len__(self):
        return len(self.keys) # H.bs * 2

    def __getitem__(self, idx):
        user_id, time_slice = self.keys[idx]
        win = range(max(0, time_slice - H.n_chunks + 1), time_slice + 1)
        x_cat  = np.concatenate([ self.x_cat [(user_id, ts)] for ts in win ])
        x_cont = np.concatenate([ self.x_cont[(user_id, ts)] for ts in win ])
        x_tags = np.concatenate([ self.x_tags[(user_id, ts)] for ts in win ])
        x_tagw = np.concatenate([ self.x_tagw[(user_id, ts)] for ts in win ])
        y      = np.concatenate([ self.y     [(user_id, ts)] for ts in win ])
        
        pad = H.chunk_size * H.n_chunks - x_cat.shape[0]
        
        # Normalize x_cont
        x_cont = (x_cont - self.means) / self.stds
        x_cont[np.isnan(x_cont)] = 0
        
        padt = (0,pad) if H.pad == 'r' else (pad,0)
        
        x_mask = np.zeros(x_cat.shape[0], dtype=np.bool)
        
        x_mask = np.pad(x_mask, padt, constant_values=(True))
        x_cat  = np.pad(x_cat , (padt, (0, 0)), constant_values=(0)).astype(np.int64)
        x_cont = np.pad(x_cont, (padt, (0, 0)), constant_values=(0)).astype(np.float32)
        x_tags = np.pad(x_tags, (padt, (0, 0)), constant_values=(0)).astype(np.int64)
        x_tagw = np.pad(x_tagw, (padt, (0, 0)), constant_values=(0.)).astype(np.float32)
        y      = np.pad(y,      (padt, (0, 0)), constant_values=(-1)).astype(np.int64)

        return x_mask, x_cat, x_cont, x_tags, x_tagw, y


# In[23]:


train_ds = InteractionsDataset(train_x_cat, train_x_cont, train_x_tags, train_x_tagw, train_y)
valid_ds = InteractionsDataset(valid_x_cat, valid_x_cont, valid_x_tags, valid_x_tagw, valid_y)


# In[24]:


x_mask, x_cat, x_cont, x_tags, x_tagw, y = train_ds[0]


# In[25]:


len(train_ds.keys)


# In[26]:


#x_tagw[-47:]


# In[27]:


assert x_cat.shape == (H.chunk_size*H.n_chunks, len(meta.cat_names))
assert x_cont.shape == (H.chunk_size*H.n_chunks, len(meta.cont_names))
assert x_tags.shape == x_tagw.shape == (H.chunk_size*H.n_chunks, 6)
assert y.shape == (H.chunk_size*H.n_chunks, 2)


# In[28]:


train_dl = DataLoader(train_ds, bs=H.bs, shuffle=True, drop_last=True, num_workers=H.workers)
valid_dl = DataLoader(valid_ds, bs=H.bs,                               num_workers=H.workers)


# In[29]:


dls = DataLoaders(train_dl, valid_dl)


# In[30]:


x_mask,x_cat, x_cont, x_tags, x_tagw, y = dls.one_batch()


# In[31]:


np.set_printoptions(precision=4, suppress=True)
torch.set_printoptions(precision=4, linewidth=200, sci_mode=False)


# In[32]:


x_cont[1]


# In[33]:


x_cat.shape, x_cont.shape, x_tags.shape, x_tagw.shape, y.shape


# In[34]:


assert x_cat.isnan().any() == False
assert x_cont.isnan().any() == False
assert x_tags.isnan().any() == False
assert x_tagw.isnan().any() == False


# In[35]:


assert x_cat.shape == (H.bs, H.chunk_size*H.n_chunks, len(meta.cat_names))
assert x_cont.shape == (H.bs, H.chunk_size*H.n_chunks, len(meta.cont_names))
assert x_tags.shape == x_tagw.shape == (H.bs, H.chunk_size*H.n_chunks, 6)
assert y.shape == (H.bs, H.chunk_size*H.n_chunks, 2)


# # Loss and metrics

# In[36]:


def roc_auc(pred, targ):
    pred = torch.softmax(pred, dim=2)
    pred = pred[:,:,1:2] # prediction for True
    idx = targ != -1
    pred = pred[idx]
    targ = targ[idx]
    pred, targ = flatten_check(pred, targ)
    if len(targ.unique()) == 2:
        return roc_auc_score(targ.cpu().numpy(), pred.cpu().numpy())
    else:
        return 0


# In[37]:


loss_fn = nn.CrossEntropyLoss if H.loss=='ce' else globals()[H.loss]
loss    = loss_fn(ignore_index=-1)
loss_nr = loss_fn(ignore_index=-1, reduction='none')

def loss_func(pred, targ, shuffle=None, lam=None):
    b, s, l = pred.shape
    if shuffle is not None:
        targ_shuffled = targ[shuffle].view(b*s)
    pred = pred.view(b*s, l)
    targ = targ.view(b*s)

    if shuffle is not None:
        l0 = loss_nr(pred, targ).view(b, s)
        l1 = loss_nr(pred, targ_shuffled).view(b, s)
        return torch.lerp(l0, l1, lam.view(lam.shape[0], 1)).mean()
    else:
        #print(targ.unique()) # CUDA assert error if any index here is bigger than dimension l (labels) of pred
        return loss(pred, targ)
    
def ua_loss_func(pred, targ, shuffle=None, lam=None):
    loss_fn = loss_func
    l = loss_fn(pred[...,:2],targ[...,:1],shuffle,lam) 
    if H.wua and targ.shape[-1]>1: l += H.wua * loss_fn(pred[...,2:],targ[...,1:],shuffle,lam)
    return l


# In[42]:


class LBMetric(Metric):
    def __init__(self, loss_func, name):
        self.loss_func = loss_func
        self.nam = name
        
    def reset(self):
        self.targs, self.preds = [], []
        
    def accumulate(self, learn):
        self.preds.append(learn.to_detach(learn.pred[...,:2]))
        self.targs.append(learn.to_detach(learn.y[...,:1]))
    
    @property
    def value(self):
        if len(self.preds) == 0: return
        preds = torch.cat(self.preds)
        targs = torch.cat(self.targs)
        r = self.loss_func(preds, targs)
        return r
    
    @property
    def name(self):
        return self.nam


# # Mixup

# In[44]:


class MyMixUp(Callback):
    run_after,run_valid = [Normalize],False
    def __init__(self, alpha=0.4): 
        self.distrib = Beta(tensor(alpha), tensor(alpha))

    def before_batch(self):
        lam = self.distrib.sample((self.y.size(0),)).squeeze().to(self.y.device)
        lam = torch.stack([lam, 1-lam], 1)
        lam = lam.max(1)[0]
        shuffle = torch.randperm(self.y.size(0)).to(self.y.device)
        self.learn.xb = (*self.xb, shuffle, lam)
        self.learn.yb = (*self.yb, shuffle, lam)


# In[45]:


class GradientClipping(Callback):
    "Gradient clipping during training."
    def __init__(self, clip:float = 0.):
        self.clip = clip

    def after_backward(self, **kwargs):
        "Clip the gradient before the optimizer step."
        if self.clip: nn.utils.clip_grad_norm_(self.learn.model.parameters(), self.clip)


# # Model

# In[46]:


class TutorNet(nn.Module):
    def __init__(self, emb_szs, tag_emb_szs, emb_do, n_cont, trf_dim, trf_enc, trf_dec, trf_heads, trf_do, trf_act):
        super().__init__()
        self.nhead,self.trf_dim = trf_heads, trf_dim
        
        tag_emb_szs =(tag_emb_szs[0]+1, trf_dim)

        self.embeds    = nn.ModuleList([nn.Sequential(nn.Embedding(ni+1, nf, max_norm=1.),nn.Linear(nf, trf_dim)) 
                                        for ni, nf in emb_szs])
        self.tagembeds = nn.EmbeddingBag(*tag_emb_szs, max_norm=1., mode='sum')
        self.conts     = nn.Linear(n_cont, trf_dim)
            
        self.trafo = nn.Transformer(
            d_model = trf_dim,
            nhead = trf_heads,
            num_encoder_layers = trf_enc,
            num_decoder_layers = trf_dec,
            dim_feedforward = trf_dim*4,
            dropout = trf_do,
            activation = trf_act,
        )

        self.mlp = nn.Linear(trf_dim, 6)
        
    def forward(self, x_mask, x_cat, x_cont, x_tags, x_tagw, shuffle=None, lam=None):
        b, sl, catf, contf, tagsf = (*x_cat.shape, x_cont.shape[2], x_tags.shape[2])
        
        x_cat  += 1
        x_tags += 1
    
        # compute masks
        causal_mask  = torch.triu(torch.ones(1,sl, sl,dtype=torch.bool,device=x_cat.device), diagonal=1).expand(b,-1,-1)
        x_tci   = x_cat[...,Cats.task_container_id]
        x_tci_s = torch.zeros_like(x_tci)
        x_tci_s[...,1:] = x_tci[...,:-1]
        enc_container_aware_mask =  (x_tci.unsqueeze(-1) == x_tci_s.unsqueeze(-1).permute(0,2,1)) | causal_mask
        dec_container_aware_mask = ~(x_tci.unsqueeze(-1) == x_tci.unsqueeze(-1).permute(0,2,1))   & causal_mask

        padding_mask = x_mask 
                
        # encoder x (shifted q & a)
        enc_cat  = torch.zeros_like(x_cat)
        enc_cont = torch.zeros_like(x_cont)
        enc_tags = torch.zeros_like(x_tags)
        enc_tagw = torch.zeros_like(x_tagw)
        
        enc_cat[:,1:]  = x_cat[:,:-1]
        enc_cont[:,1:] = x_cont[:,:-1]
        enc_tags[:,1:] = x_tags[:,:-1]
        enc_tagw[:,1:] = x_tagw[:,:-1]
        
        # decoder x (nonshifted q)
        dec_cat  = x_cat
        dec_cont = x_cont
        dec_tags = x_tags
        dec_tagw = x_tagw

        # hide correct answer and user answered correctly from decoder
        dec_cat[...,Cats.answered_correctly] = 0
        dec_cat[...,Cats.user_answer] = 0
        dec_cat[...,Cats.qhe] = 0
        dec_cont[...,Conts.qet] = 0
        dec_cont[...,Conts.qet_log] = 0
        
        # print(enc_cont.shape)
        enc_cat  =  enc_cat.view(b * sl, catf)   # b*sl, catf
        enc_tags = enc_tags.view(b * sl, tagsf) # b*sl, tagsf
        enc_tagw = enc_tagw.view(b * sl, tagsf) # b*sl, tagsf

        dec_cat  =  dec_cat.view(b * sl, catf)   # b*sl, catf
        dec_tags = dec_tags.view(b * sl, tagsf) # b*sl, tagsf
        dec_tagw = dec_tagw.view(b * sl, tagsf) # b*sl, tagsf
        
        # embed categorical vars
        enc = torch.mean(torch.stack([
            *[ e(enc_cat[:,i]) for i, e in enumerate(self.embeds) ],
            self.tagembeds(enc_tags, per_sample_weights=enc_tagw),
            self.conts(enc_cont).view(-1,self.trf_dim)
        ]),dim=0)
        
        dec = torch.mean(torch.stack([
            *[ e(dec_cat[:,i]) for i, e in enumerate(self.embeds) ],
            self.tagembeds(dec_tags, per_sample_weights=dec_tagw),
            self.conts(dec_cont).view(-1,self.trf_dim)
        ]),dim=0)
        
        enc = enc.view(b, sl, self.trf_dim)           # b, sl, sum of cat, cont and tag ftrs
        dec = dec.view(b, sl, self.trf_dim)           # b, sl, sum of cat, cont and tag ftrs

        if shuffle is not None:
            enc = torch.lerp(enc, enc[shuffle], lam.view(lam.shape[0], 1, 1))
            dec = torch.lerp(dec, dec[shuffle], lam.view(lam.shape[0], 1, 1))
            padding_mask = None
            enc_container_aware_mask = dec_container_aware_mask = causal_mask | causal_mask[shuffle]
        
        enc = enc.permute(1, 0, 2)          # sl, b, tf (torchformer input)
        dec = dec.permute(1, 0, 2)          # sl, b, tf

        expand_nheads = lambda t: t.unsqueeze(1).expand(t.shape[0],self.nhead,-1,-1).reshape(-1,*t.shape[-2:])
        
        o = self.trafo(
            enc, 
            dec, 
            src_mask = expand_nheads(enc_container_aware_mask),
            tgt_mask = expand_nheads(dec_container_aware_mask),
            memory_mask = expand_nheads(enc_container_aware_mask),
            src_key_padding_mask = padding_mask,
            tgt_key_padding_mask = padding_mask,
            memory_key_padding_mask = padding_mask,
        )                                   # sl, b, tf
        o = o.permute(1, 0, 2)              # b, sl, tf
        o = self.mlp(o)                     # b, sl, of (of=2)
        #print(o)
        return o


# In[47]:


emb_szs = list(zip(meta.n_emb.values(), meta.emb_dim.values()))
tag_emb_szs = meta.tags_n_emb, meta.tags_emb_dim


# In[48]:


model = TutorNet(emb_szs, tag_emb_szs, None, len(meta.cont_names), 
                 H.trf_dim, H.trf_enc, H.trf_dec, H.trf_heads, H.trf_do, H.trf_act)


# # T-Fixup init
# 
# 1. Apply Xavier initialization for all parameters excluding input embeddings. Use Gaussian initialization $N(0,d^{-\frac{1}{2}})$ for input embeddings where d is the embedding dimension.
# 
# 2. Scale $v_{d}$ and $w_{d}$ matrices in each decoder attention block, weight matrices in each decoder MLP block and input embeddings $x$ and $y$ in encoder and decoder by $(9N)^{−\frac{1}{4}}$: [code](https://github.com/layer6ai-labs/T-Fixup/blob/f1fae213ce7b48829f81632d0c96bb039b7c450e/fairseq/modules/transformer_layer.py#L161), [code](https://github.com/layer6ai-labs/T-Fixup/blob/f1fae213ce7b48829f81632d0c96bb039b7c450e/fairseq/models/transformer.py#L378), [code](https://github.com/layer6ai-labs/T-Fixup/blob/f1fae213ce7b48829f81632d0c96bb039b7c450e/fairseq/models/transformer.py#L604)
# 
# 3. Scale $v_{e}$ and $w_{e}$ matrices in each encoder attention block and weight matrices in each encoder MLP block by $0.67N^{−\frac{1}{4}}$: [code](https://github.com/layer6ai-labs/T-Fixup/blob/f1fae213ce7b48829f81632d0c96bb039b7c450e/fairseq/modules/transformer_layer.py#L36)

# In[49]:


def trunc_normal_(x, mean=0., std=1.):
    "Truncated normal initialization (approximation)"
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)


# In[50]:


if H.tfixup:
    for n,p in model.named_parameters():
        if re.match(r'.*bias$|.*bn\.weight$|.*norm.*\.weight',n): continue
        gain = 1.
        if re.match(r'.*decoder.*',n): 
            gain = (9*H.trf_dec)**(-1./4.)
            if re.match(f'.*in_proj_weight$',n): gain *= (2**0.5)
        elif re.match(r'.*encoder.*',n): 
            gain = 0.67*(H.trf_enc**(-1./4.))
            if re.match(f'.*in_proj_weight$',n): gain *= (2**0.5)
        if re.match(r'^embeds|^tagembeds', n): 
            trunc_normal_(p.data,std=(4.5*(H.trf_enc+H.trf_dec))**(-1./4.)*H.trf_dim**(-0.5))
        else:                                  
            nn.init.xavier_normal_(p,gain=gain)


# In[51]:


if H.tfixup:
    class MyModelPatcher(ModelPatcher):
        def new_child_module(self, child_module_name, child_module, patch_info): return nn.Identity()
    mp = MyModelPatcher()
    mp.add_pattern(r".*norm\d?.*",{})
    mp.patch_model(model)


# In[53]:


dls = dls.cuda()
model = model.cuda()


# In[54]:


@delegates(Lamb)
def ranger_lamb(p, lr, mom=0.95, wd=0.01, eps=1e-6, **kwargs):
    return Lookahead(Lamb(p, lr=lr, mom=mom, wd=wd, eps=eps, **kwargs))


# In[55]:


'''
# TESTING
'''
'''
import torch

def convert_module(module, dtype):
    """
    Converts a module's immediate parameters and buffers to dtype.
    """
    for param in module.parameters(recurse=False):
        if param is not None:
            if param.data.dtype.is_floating_point:
                param.data = param.data.to(dtype=dtype)
            if param._grad is not None and param._grad.data.dtype.is_floating_point:
                param._grad.data = param._grad.data.to(dtype=dtype)

    for buf in module.buffers(recurse=False):
        if buf is not None and buf.data.dtype.is_floating_point:
            buf.data = buf.data.to(dtype=dtype)

def convert_network(network, dtype):
    """
    Converts a network's parameters and buffers to dtype.
    """
    for module in network.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.affine is True:
            continue
        convert_module(module, dtype)
        if isinstance(module, torch.nn.RNNBase) or isinstance(module, torch.nn.modules.rnn.RNNBase):
            module.flatten_parameters()
    return network

model = convert_network(model, dtype=torch.float16)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
criterion = lambda x,y: ua_loss_func(x, y)
scaler = torch.cuda.amp.GradScaler()

for step, batch in enumerate(dls[0]):
    with torch.cuda.amp.autocast():
        outputs = model(*[b.half() if i not in [0,1,3] else b for i,b in enumerate(batch[:-1])])
        loss = criterion(outputs, batch[-1]).clone()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    print(step)

sys.exit()
#'''


import torch
def opt_func(params, **kwargs): return OptimWrapper(torch.optim.SGD(params, lr=0.001))
learn = Learner(
    dls,
    model,
    loss_func=ua_loss_func,
    opt_func=opt_func,
    moms = H.moms,
    metrics=[
        LBMetric(loss_func, 'acc_valid_loss'),
        LBMetric(roc_auc, 'acc_roc_auc'),
    ],
)
#    opt_func=partial(globals()[H.opt],**H.opt_kwargs),


# In[56]:


f_fp16 = getattr(learn,H.fp16,None)
if f_fp16: f_fp16()


# In[57]:


def rank0_only(func, *args, **kwargs):
    "Execute `func` in the Rank-0 process first, then in other ranks in parallel."
    if args or kwargs: func = partial(func, *args, **kwargs)
    dummy_l = Learner(DataLoaders(device='cpu'), nn.Linear(1,1), loss_func=lambda: 0)
    res = None
    with dummy_l.distrib_ctx():
        if not rank_distrib(): res = func()
        distrib_barrier()
    return res


# In[58]:


@patch
def load(learn:Learner,fn,with_opt=False):
    def __inner(learn:Learner,fn,with_opt=False):
        m_dict = torch.load(f"{(Path(learn.model_dir) / fn)}.pth")#['model']
        ks = []
        for attempts in range(2):
            try:
                res = learn.model.load_state_dict(m_dict,strict=False)
                print(f"Loaded {fn} ignoring: {' '.join(ks)} and {res}")
                break
            except Exception as e:
                for k in [m[1] for m in [re.match(r"^.*mismatch for ([\w\.]+):",l) for l in str(e).split("\n")] if m is not None]:
                    m_dict.pop(k,None)
                    ks.append(k)
        return learn
    return rank0_only(__inner, learn, fn, with_opt)


# # Load model

# In[59]:


if H.load:
    learn.load(H.load, with_opt=False)
    print(f"Loaded: {H.load}")


# In[60]:


if H.clip: 
    learn.add_cb(GradientClipping(H.clip))
    print('clip on')


# In[61]:


if H.local_rank is not None: 
    learn.to_distributed(H.local_rank)
    print('local_rank on')
if H.mixup: 
    learn.add_cb(MyMixUp(0.5))
    print('mixup on')


# In[62]:


if H.validate:
    res = learn.validate()
    print(f"CV: {res[-1]}")


# # Train

# In[66]:


#short_cb = learn.add_cb(ShortEpochCallback(pct=0.001))


# In[67]:


learn.add_cb(SaveModelCallback(monitor='acc_roc_auc', comp=np.greater, fname=f'best{H.model}'))


# In[68]:


print(H.fit, H.epochs, H.lr, H.fit_kwargs)


# In[ ]:


print(f"Fitting {H.local_rank}")
print(learn.cbs)
sys.exit()
getattr(learn,H.fit)(H.epochs, H.lr,**H.fit_kwargs)


# In[ ]:




