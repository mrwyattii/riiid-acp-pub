#!/usr/bin/env python
# coding: utf-8

# Added cell to measure execution time
import time

start_time = time.time()

from argparse import ArgumentParser
import re
import os
import sys

#import deepspeed

import ast
import enum
import gc
import numpy as np
import pandas as pd
import pickle
import enum

from fastai.basics import AttrDict

from pynvml import *
nvmlInit()

from collections import defaultdict
#from fastcore.script import *
#from matplotlib import pyplot as plt
from pathlib import Path
#from pytorch_block_sparse.util import ModelPatcher
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.distributions.beta import Beta
from torch.utils.data.distributed import DistributedSampler
from pytorch_block_sparse.util import ModelPatcher

# Ugly hack because of notebook that generates data
_H = AttrDict


def parseArgs():
    def dir_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise NotADirectoryError(string)

    parser = ArgumentParser()
    parser.add_argument("--input", type=dir_path, default="../input")
    parser.add_argument("--data_version", type=str, default="210101b")
    parser.add_argument("--model_name", type=str, default="210105")
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--n_chunks", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--data_pct", type=float, default=1)
    parser.add_argument("--val_pct", type=float, default=0.025)
    parser.add_argument("--trf_dim", type=int, default=512)
    parser.add_argument("--trf_enc", type=int, default=4)
    parser.add_argument("--trf_dec", type=int, default=4)
    parser.add_argument("--trf_heads", type=int, default=4)
    parser.add_argument("--trf_do", type=float, default=0.1)
    parser.add_argument("--trf_act", type=str, default="gelu")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--clip", type=float, default=0)
    parser.add_argument("--moms", type=float, nargs="+", default=(0.95, 0.85, 0.95))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--no_use_t_fixup_init", action="store_true")
    parser.add_argument("--use_mixup", action="store_true")
    parser.add_argument("--optimizer", type=str, default="ranger_lamb")
    parser.add_argument("--loss", type=str, default="ce")
    parser.add_argument("--pad", type=str, choices=["l", "r"], default="r")
    parser.add_argument("--wua", type=float, default=0.0)
    parser.add_argument("--local_rank", type=int, default=None)
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count())
    parser.add_argument("--no_torch_dist", action="store_true")
    return parser.parse_args()


################
## DATA STUFF ##
################


def loadData(args):
    with open(os.path.join(args.input, f"meta_v{args.data_version}.pkl"), "rb") as f:
        meta = pickle.load(f)

    QCols = enum.IntEnum("QCols", meta.qcols, start=0)
    LCols = enum.IntEnum("LCols", meta.lcols, start=0)
    Cats = enum.IntEnum("Cats", meta.cat_names, start=0)
    Conts = enum.IntEnum("Conts", meta.cont_names, start=0)

    with open(os.path.join(args.input, f"data_v{args.data_version}.pkl"), "rb") as f:
        data = pickle.load(f)

    del data.attempt_num_coo
    del data.attempts_correct_coo
    gc.collect()

    # # Y
    lut = meta.icats["answered_correctly"], meta.icats["user_answer"]
    y_d = {}
    for k, v in data.cat_d.items():
        y_d[k] = np.column_stack(
            (
                lut[0][v[:, Cats.answered_correctly] - 1],
                lut[1][v[:, Cats.user_answer] - 1],
            )
        )

    # # Chop sequences
    def chop_sequence(d):
        nv = defaultdict(dict)
        for k, v in d.items():
            i = 0
            while i * args.chunk_size < len(v):
                nv[k][i] = v[i * args.chunk_size : (i + 1) * args.chunk_size]
                i += 1
        return nv

    cat_d = chop_sequence(data.cat_d)
    cont_d = chop_sequence(data.cont_d)
    tags_d = chop_sequence(data.tags_d)
    tagw_d = chop_sequence(data.tagw_d)
    y_d = chop_sequence(y_d)

    print(f"There are {len(data.cat_d)} different users")
    return meta, Cats, Conts, (cat_d, cont_d, tags_d, tagw_d, y_d)


def splitTrainVal(args, cat_d, cont_d, tags_d, tagw_d, y_d):
    group_keys = sorted(list(cat_d.keys()))
    group_keys = group_keys[:int(args.data_pct * len(group_keys))]
    train_group_keys = group_keys[: int((1 - args.val_pct) * len(group_keys))]
    valid_group_keys = group_keys[int((1 - args.val_pct) * len(group_keys)) :]
    print(f"users: train={len(train_group_keys)}, valid={len(valid_group_keys)}")

    def split_dict(d, keys):
        return {(u, t): d[u][t] for u in keys for t in d[u].keys()}

    train_x_cat = split_dict(cat_d, train_group_keys)
    train_x_cont = split_dict(cont_d, train_group_keys)
    train_x_tags = split_dict(tags_d, train_group_keys)
    train_x_tagw = split_dict(tagw_d, train_group_keys)
    train_y = split_dict(y_d, train_group_keys)

    valid_x_cat = split_dict(cat_d, valid_group_keys)
    valid_x_cont = split_dict(cont_d, valid_group_keys)
    valid_x_tags = split_dict(tags_d, valid_group_keys)
    valid_x_tagw = split_dict(tagw_d, valid_group_keys)
    valid_y = split_dict(y_d, valid_group_keys)

    print(f"seqs: train={len(train_x_cat)}, valid={len(valid_x_cat)}")
    return (train_x_cat, train_x_cont, train_x_tags, train_x_tagw, train_y), (
        valid_x_cat,
        valid_x_cont,
        valid_x_tags,
        valid_x_tagw,
        valid_y,
    )


class InteractionsDataset(Dataset):
    def __init__(self, args, meta, x_cat, x_cont, x_tags, x_tagw, y, minids=False):
        super().__init__()

        self.args = args
        self.means = np.expand_dims(meta.means, axis=0)  # ready to broadcast
        self.stds = np.expand_dims(meta.stds, axis=0)

        self.n_inp = 5  # number of feature (x) tensors

        self.x_cat = x_cat  # SL, XF (sequence len, feature columns)
        self.x_cont = x_cont
        self.x_tags = x_tags
        self.x_tagw = x_tagw
        self.y = y  # SL, 1

        self.keys = list(self.x_cat.keys())  # list of group keys

        if minids:
            self.keys = self.keys[: args.batch_size * 2]

    def __len__(self):
        return len(self.keys)  # H.bs * 2

    def __getitem__(self, idx):
        user_id, time_slice = self.keys[idx]
        win = range(max(0, time_slice - self.args.n_chunks + 1), time_slice + 1)
        x_cat = np.concatenate([self.x_cat[(user_id, ts)] for ts in win])
        x_cont = np.concatenate([self.x_cont[(user_id, ts)] for ts in win])
        x_tags = np.concatenate([self.x_tags[(user_id, ts)] for ts in win])
        x_tagw = np.concatenate([self.x_tagw[(user_id, ts)] for ts in win])
        y = np.concatenate([self.y[(user_id, ts)] for ts in win])

        pad = self.args.chunk_size * self.args.n_chunks - x_cat.shape[0]

        # Normalize x_cont
        x_cont = (x_cont - self.means) / self.stds
        x_cont[np.isnan(x_cont)] = 0

        padt = (0, pad) if self.args.pad == "r" else (pad, 0)

        x_mask = np.zeros(x_cat.shape[0], dtype=bool)

        x_mask = np.pad(x_mask, padt, constant_values=(True))
        x_cat = np.pad(x_cat, (padt, (0, 0)), constant_values=(0)).astype(np.int64)
        x_cont = np.pad(x_cont, (padt, (0, 0)), constant_values=(0)).astype(np.float32)
        x_tags = np.pad(x_tags, (padt, (0, 0)), constant_values=(0)).astype(np.int64)
        x_tagw = np.pad(x_tagw, (padt, (0, 0)), constant_values=(0.0)).astype(
            np.float32
        )
        y = np.pad(y, (padt, (0, 0)), constant_values=(-1)).astype(np.int64)

        return x_mask, x_cat, x_cont, x_tags, x_tagw, y


################
## LOSS STUFF ##
################


def roc_auc(pred, targ):
    pred = torch.softmax(pred, dim=2)
    pred = pred[:, :, 1:2]  # prediction for True
    idx = targ != -1
    pred = pred[idx]
    targ = targ[idx]
    pred, targ = flatten_check(pred, targ)
    if len(targ.unique()) == 2:
        return roc_auc_score(targ.cpu().numpy(), pred.cpu().numpy())
    else:
        return 0


loss_fn = nn.CrossEntropyLoss
loss = loss_fn(ignore_index=-1)
loss_nr = loss_fn(ignore_index=-1, reduction="none")


def loss_func(pred, targ, shuffle=None, lam=None):
    b, s, l = pred.shape
    if shuffle is not None:
        targ_shuffled = targ[shuffle].view(b * s)
    pred = pred.view(b * s, l)
    targ = targ.view(b * s)

    if shuffle is not None:
        l0 = loss_nr(pred, targ).view(b, s)
        l1 = loss_nr(pred, targ_shuffled).view(b, s)
        return torch.lerp(l0, l1, lam.view(lam.shape[0], 1)).mean()
    else:
        # print(targ.unique()) # CUDA assert error if any index here is bigger than dimension l (labels) of pred
        return loss(pred, targ)


def ua_loss_func(pred, targ, args, shuffle=None, lam=None):
    loss_fn = loss_func
    l = loss_fn(pred[..., :2], targ[..., :1], shuffle, lam)
    if args.wua and targ.shape[-1] > 1:
        l += args.wua * loss_fn(pred[..., 2:], targ[..., 1:], shuffle, lam)
    return l


'''
class LBMetric(Metric):
    def __init__(self, loss_func, name):
        self.loss_func = loss_func
        self.nam = name

    def reset(self):
        self.targs, self.preds = [], []

    def accumulate(self, learn):
        self.preds.append(learn.to_detach(learn.pred[..., :2]))
        self.targs.append(learn.to_detach(learn.y[..., :1]))

    @property
    def value(self):
        if len(self.preds) == 0:
            return
        preds = torch.cat(self.preds)
        targs = torch.cat(self.targs)
        r = self.loss_func(preds, targs)
        return r

    @property
    def name(self):
        return self.nam
'''


#################
## MODEL STUFF ##
#################


class TutorNet(nn.Module):
    def __init__(
        self,
        Cats,
        Conts,
        emb_szs,
        tag_emb_szs,
        emb_do,
        n_cont,
        trf_dim,
        trf_enc,
        trf_dec,
        trf_heads,
        trf_do,
        trf_act,
    ):
        super().__init__()
        self.Cats = Cats
        self.Conts = Conts
        self.nhead, self.trf_dim = trf_heads, trf_dim

        tag_emb_szs = (tag_emb_szs[0] + 1, trf_dim)

        self.embeds = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Embedding(ni + 1, nf, max_norm=1.0), nn.Linear(nf, trf_dim)
                )
                for ni, nf in emb_szs
            ]
        )
        self.tagembeds = nn.EmbeddingBag(*tag_emb_szs, max_norm=1.0, mode="sum")
        self.conts = nn.Linear(n_cont, trf_dim)

        self.trafo = nn.Transformer(
            d_model=trf_dim,
            nhead=trf_heads,
            num_encoder_layers=trf_enc,
            num_decoder_layers=trf_dec,
            dim_feedforward=trf_dim * 4,
            dropout=trf_do,
            activation=trf_act,
        )

        self.mlp = nn.Linear(trf_dim, 6)

    def forward(self, x_mask, x_cat, x_cont, x_tags, x_tagw, shuffle=None, lam=None):
        b, sl, catf, contf, tagsf = (*x_cat.shape, x_cont.shape[2], x_tags.shape[2])

        x_cat += 1
        x_tags += 1

        # compute masks
        causal_mask = torch.triu(
            torch.ones(1, sl, sl, dtype=torch.bool, device=x_cat.device), diagonal=1
        ).expand(b, -1, -1)
        x_tci = x_cat[..., self.Cats.task_container_id]
        x_tci_s = torch.zeros_like(x_tci)
        x_tci_s[..., 1:] = x_tci[..., :-1]
        enc_container_aware_mask = (
            x_tci.unsqueeze(-1) == x_tci_s.unsqueeze(-1).permute(0, 2, 1)
        ) | causal_mask
        dec_container_aware_mask = (
            ~(x_tci.unsqueeze(-1) == x_tci.unsqueeze(-1).permute(0, 2, 1)) & causal_mask
        )

        padding_mask = x_mask

        # encoder x (shifted q & a)
        enc_cat = torch.zeros_like(x_cat)
        enc_cont = torch.zeros_like(x_cont)
        enc_tags = torch.zeros_like(x_tags)
        enc_tagw = torch.zeros_like(x_tagw)

        enc_cat[:, 1:] = x_cat[:, :-1]
        enc_cont[:, 1:] = x_cont[:, :-1]
        enc_tags[:, 1:] = x_tags[:, :-1]
        enc_tagw[:, 1:] = x_tagw[:, :-1]

        # decoder x (nonshifted q)
        dec_cat = x_cat
        dec_cont = x_cont
        dec_tags = x_tags
        dec_tagw = x_tagw

        # hide correct answer and user answered correctly from decoder
        dec_cat[..., self.Cats.answered_correctly] = 0
        dec_cat[..., self.Cats.user_answer] = 0
        dec_cat[..., self.Cats.qhe] = 0
        dec_cont[..., self.Conts.qet] = 0
        dec_cont[..., self.Conts.qet_log] = 0

        # print(enc_cont.shape)
        enc_cat = enc_cat.view(b * sl, catf)  # b*sl, catf
        enc_tags = enc_tags.view(b * sl, tagsf)  # b*sl, tagsf
        enc_tagw = enc_tagw.view(b * sl, tagsf)  # b*sl, tagsf

        dec_cat = dec_cat.view(b * sl, catf)  # b*sl, catf
        dec_tags = dec_tags.view(b * sl, tagsf)  # b*sl, tagsf
        dec_tagw = dec_tagw.view(b * sl, tagsf)  # b*sl, tagsf

        # embed categorical vars
        enc = torch.mean(
            torch.stack(
                [
                    *[e(enc_cat[:, i]) for i, e in enumerate(self.embeds)],
                    self.tagembeds(enc_tags, per_sample_weights=enc_tagw),
                    self.conts(enc_cont).view(-1, self.trf_dim),
                ]
            ),
            dim=0,
        )

        dec = torch.mean(
            torch.stack(
                [
                    *[e(dec_cat[:, i]) for i, e in enumerate(self.embeds)],
                    self.tagembeds(dec_tags, per_sample_weights=dec_tagw),
                    self.conts(dec_cont).view(-1, self.trf_dim),
                ]
            ),
            dim=0,
        )

        enc = enc.view(b, sl, self.trf_dim)  # b, sl, sum of cat, cont and tag ftrs
        dec = dec.view(b, sl, self.trf_dim)  # b, sl, sum of cat, cont and tag ftrs

        if shuffle is not None:
            enc = torch.lerp(enc, enc[shuffle], lam.view(lam.shape[0], 1, 1))
            dec = torch.lerp(dec, dec[shuffle], lam.view(lam.shape[0], 1, 1))
            padding_mask = None
            enc_container_aware_mask = dec_container_aware_mask = (
                causal_mask | causal_mask[shuffle]
            )

        enc = enc.permute(1, 0, 2)  # sl, b, tf (torchformer input)
        dec = dec.permute(1, 0, 2)  # sl, b, tf

        expand_nheads = (
            lambda t: t.unsqueeze(1)
            .expand(t.shape[0], self.nhead, -1, -1)
            .reshape(-1, *t.shape[-2:])
        )

        o = self.trafo(
            enc,
            dec,
            src_mask=expand_nheads(enc_container_aware_mask),
            tgt_mask=expand_nheads(dec_container_aware_mask),
            memory_mask=expand_nheads(enc_container_aware_mask),
            src_key_padding_mask=padding_mask,
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=padding_mask,
        )  # sl, b, tf
        o = o.permute(1, 0, 2)  # b, sl, tf
        o = self.mlp(o)  # b, sl, of (of=2)
        # print(o)
        return o


def main(args):
    # Torch Distributed
    if args.local_rank is not None and not args.no_torch_dist:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        print(f"DISTRIBUTED: {args.local_rank}")

    # Load in dataset
    meta, Cats, Conts, data = loadData(args)
    train, valid = splitTrainVal(args, *data)
    train_ds = InteractionsDataset(args, meta, *train)
    valid_ds = InteractionsDataset(args, meta, *valid)

    # Check data
    x_mask, x_cat, x_cont, x_tags, x_tagw, y = train_ds[0]
    assert x_cat.shape == (args.chunk_size * args.n_chunks, len(meta.cat_names))
    assert x_cont.shape == (args.chunk_size * args.n_chunks, len(meta.cont_names))
    assert x_tags.shape == x_tagw.shape == (args.chunk_size * args.n_chunks, 6)
    assert y.shape == (args.chunk_size * args.n_chunks, 2)

    # Create dataloaders
    train_sampler = DistributedSampler(train_ds, args.world_size, args.local_rank)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
    )

    # Setup model
    emb_szs = list(zip(meta.n_emb.values(), meta.emb_dim.values()))
    tag_emb_szs = meta.tags_n_emb, meta.tags_emb_dim
    model = TutorNet(
        Cats,
        Conts,
        emb_szs,
        tag_emb_szs,
        None,
        len(meta.cont_names),
        args.trf_dim,
        args.trf_enc,
        args.trf_dec,
        args.trf_heads,
        args.trf_do,
        args.trf_act,
    )

    # T-Fixup init
    if not args.no_use_t_fixup_init:

        def trunc_normal_(x, mean=0.0, std=1.0):
            "Truncated normal initialization (approximation)"
            # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
            return x.normal_().fmod_(2).mul_(std).add_(mean)

        class MyModelPatcher(ModelPatcher):
            def new_child_module(self, child_module_name, child_module, patch_info):
                return nn.Identity()

        for n, p in model.named_parameters():
            if re.match(r".*bias$|.*bn\.weight$|.*norm.*\.weight", n):
                continue
            gain = 1.0
            if re.match(r".*decoder.*", n):
                gain = (9 * args.trf_dec) ** (-1.0 / 4.0)
                if re.match(f".*in_proj_weight$", n):
                    gain *= 2**0.5
            elif re.match(r".*encoder.*", n):
                gain = 0.67 * (args.trf_enc ** (-1.0 / 4.0))
                if re.match(f".*in_proj_weight$", n):
                    gain *= 2**0.5
            if re.match(r"^embeds|^tagembeds", n):
                trunc_normal_(
                    p.data,
                    std=(4.5 * (args.trf_enc + args.trf_dec)) ** (-1.0 / 4.0)
                    * args.trf_dim ** (-0.5),
                )
            else:
                nn.init.xavier_normal_(p, gain=gain)

            mp = MyModelPatcher()
            mp.add_pattern(r".*norm\d?.*", {})
            mp.patch_model(model)

    # DeepSpeed only
    if args.no_torch_dist:
        ds_config = {
            "train_micro_batch_size_per_gpu": args.batch_size,
            "optimizer": {
                "type": "ADAM",
                "params": {
                    "lr": 1e-4
                }
            },
            "fp16": {
                "enabled": False
            },
            "zero_optimization": {
                "stage": 1,
                "cpu_offload": True
            }
        }
        '''
                "offload_optimizer": {
                    "device": "cpu"
                }
        '''

        criterion = (lambda x,y: ua_loss_func(x, y, args))
        model, _, _, _ = deepspeed.initialize(model=model,
                                              model_parameters=model.parameters(),
                                              config=ds_config)
        start = time.time()
        for step, batch in enumerate(train_dl):
            if step % 100 == 0 and args.local_rank == 0:
                print(f"batch step {step} / {len(train_dl)}")
            #batch = [val.to(device).half() if (i==2 or i==4) else val.to(device) for i,val in enumerate(batch)]
            batch = [val.cuda(args.local_rank) for val in batch]
            outputs = model(*batch[:-1])
            loss = criterion(outputs, batch[-1])
            model.backward(loss)
            model.step()

        if args.local_rank == 0:
            print(f" Total Time: {time.time() - start}")

    # TORCH DISTRIBUTED
    else:
        if args.local_rank == 0:
            h = nvmlDeviceGetHandleByIndex(0)
            print("MEMORY USAGE 1 (init):", torch.cuda.memory_allocated(0)/(1024**3), '/', torch.cuda.max_memory_allocated(0)/(1024**3))
            info = nvmlDeviceGetMemoryInfo(h)
            print("total/free/used", f"{info.total/(1024**3)}/{info.free/(1024**3)}/{info.used/(1024**3)}")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = (lambda x,y: ua_loss_func(x, y, args))
        scaler = torch.cuda.amp.GradScaler()

        #model = model.half()
        #if args.local_rank == 0:
        #    print(model)
        #    print(sum(p.numel() for p in model.parameters()))
        #    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

        model = model.cuda(args.local_rank)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        if args.local_rank == 0:
            print("MEMORY USAGE 2 (model):", torch.cuda.memory_allocated(0)/(1024**3), '/', torch.cuda.max_memory_allocated(0)/(1024**3))
            info = nvmlDeviceGetMemoryInfo(h)
            print("total/free/used", f"{info.total/(1024**3)}/{info.free/(1024**3)}/{info.used/(1024**3)}")

        start = time.time()
        for step, batch in enumerate(train_dl):
            optimizer.zero_grad()
            if step % 100 == 0 and args.local_rank == 0:
                print(f"batch step {step} / {len(train_dl)}, total datapoints: {len(train_dl.dataset)}")
                if step == 0:
                    sizes = [a.element_size() * a.nelement() for a in batch]
                    print("data sizes:", sizes)
            inputs = [val.cuda(args.local_rank) for val in batch[:-1]]
            if args.local_rank == 0:
                print(f"Step {step}, MEMORY USAGE 3 (train loop):", torch.cuda.memory_allocated(0)/(1024**3), '/', torch.cuda.max_memory_allocated(0)/(1024**3))
                info = nvmlDeviceGetMemoryInfo(h)
                print("total/free/used", f"{info.total/(1024**3)}/{info.free/(1024**3)}/{info.used/(1024**3)}")
                print("Tensor size (GB):", sum([a.element_size() * a.nelement() for a in batch])/(1e9))

            with torch.cuda.amp.autocast():
                outputs = model(*inputs)
                del inputs
                loss = criterion(outputs, batch[-1].cuda(args.local_rank))
                del outputs
            if step == 3:
                sys.exit()
            scaler.scale(loss).backward()
            del loss
            scaler.step(optimizer)
            scaler.update()
            #loss.backward()
            #optimizer.step()

        if args.local_rank == 0:
            print(f" Total Time: {time.time() - start}")


if __name__ == "__main__":
    args = parseArgs()
    main(args)
    sys.exit()
