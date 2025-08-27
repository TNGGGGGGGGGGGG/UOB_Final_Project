#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import shutil
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
from openai import OpenAI
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# ─── Prompt 配置 ─────────────────────────────────────────────────────────────
PROMPT_FILE = "prompt.txt"
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    PROMPT_TEMPLATE = f.read().strip()

# ─── 原始只读数据路径 ─────────────────────────────────────────────────────────
ORIG_POOL_CSV   = "data/pool.csv"             # id,keyword,location,text
ORIG_TRAIN_CSV  = "data/train_with_hard.csv"  # id,keyword,location,text,target,p_disaster,p_non_disaster
DEV_CSV         = "data/dev.csv"              # id,keyword,location,text,target
CKPT_DIR        = "Stage2"                    # 初始 checkpoint 目录

# ─── 工作目录 & 副本 ───────────────────────────────────────────────────────────
WORK_DIR   = "work"
os.makedirs(WORK_DIR, exist_ok=True)
POOL_CSV   = os.path.join(WORK_DIR, "pool.csv")
TRAIN_CSV  = os.path.join(WORK_DIR, "train_with_hard.csv")
if not os.path.exists(POOL_CSV):
    shutil.copy(ORIG_POOL_CSV, POOL_CSV)
if not os.path.exists(TRAIN_CSV):
    shutil.copy(ORIG_TRAIN_CSV, TRAIN_CSV)

# ─── DeepSeek-Chat API 配置 ───────────────────────────────────────────────────
os.environ["DEEPSEEK_API_KEY"] = "sk-6b369b3e8d684735b84bbdf7002bc25b"
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com"
)
DS_MODEL = "deepseek-chat"

# ─── 主流程 & 超参 ─────────────────────────────────────────────────────────────
AL_ROUNDS   = 8     # 主动学习轮数
QUERY_K     = 150    # 每轮选样数
MC_ITERS    = 8     # MC-Dropout 前向次数

BATCH_SIZE  = 32
MAX_LEN     = 128
LR          = 1e-5
EPOCHS      = 15
ALPHA       = 0.7   # KD vs CE 权重
TEMPERATURE = 3.0   # 蒸馏温度
SEED        = 42

# Joblib 并行数
N_JOBS = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── 数据集定义 ───────────────────────────────────────────────────────────────
class PoolDataset(Dataset):
    """仅包含 text，用于不确定度选样"""
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt"
        )
        return enc.input_ids.squeeze(0), enc.attention_mask.squeeze(0)


class MixedDataset(Dataset):
    """
    包含 text + 软标签 + 硬标签(class_idx)
    class_idx: disaster → 0, non-disaster → 1
    """
    def __init__(self, csv_path, tokenizer):
        self.df = pd.read_csv(csv_path)
        self.df["class_idx"] = self.df["target"].map({1:0, 0:1})
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        enc  = self.tokenizer(
            r["text"],
            truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt"
        )
        soft = torch.tensor([r["p_disaster"], r["p_non_disaster"]], dtype=torch.float)
        hard = torch.tensor(int(r["class_idx"]), dtype=torch.long)
        return enc.input_ids.squeeze(0), enc.attention_mask.squeeze(0), soft, hard


# ─── BALD 选样 ────────────────────────────────────────────────────────────────
def select_by_bald(model, tokenizer, texts, k, mc_iters):
    model.train()
    ds = PoolDataset(texts, tokenizer)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    all_probs = [[] for _ in texts]

    with torch.no_grad():
        for _ in range(mc_iters):
            offset = 0
            for ids, masks in tqdm(dl, desc="MC-Dropout"):
                ids, masks = ids.to(DEVICE), masks.to(DEVICE)
                logits = model(input_ids=ids, attention_mask=masks).logits
                probs  = F.softmax(logits, dim=1).cpu()
                bs = probs.size(0)
                for i in range(bs):
                    all_probs[offset + i].append(probs[i])
                offset += bs

    scores = []
    for plist in all_probs:
        stacked = torch.stack(plist, dim=0)
        p_bar   = stacked.mean(dim=0, keepdim=True)
        kl_div  = F.kl_div(stacked.log(), p_bar.expand_as(stacked), reduction="batchmean")
        scores.append(kl_div.item())

    model.eval()
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]


# ─── Joblib 并行单条标注 ───────────────────────────────────────────────────────
def _label_one(text):
    prompt = PROMPT_TEMPLATE.format(text=text)
    resp = client.chat.completions.create(
        model=DS_MODEL,
        messages=[
            {"role":"system","content":"Only output a JSON array [target,p_disaster,p_non_disaster]."},
            {"role":"user","content":prompt}
        ],
        temperature=0.0,
        max_tokens=30
    )
    txt = resp.choices[0].message.content.strip()
    try:
        arr = json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r'\[\s*[01]\s*,\s*0?\.\d+\s*,\s*0?\.\d+\s*\]', txt)
        if m:
            arr = json.loads(m.group(0))
        else:
            print("无法解析 LLM 输出:", txt)
            arr = [0, 0.5, 0.5]
    return {
        "text":           text,
        "target":         int(arr[0]),
        "p_disaster":     float(arr[1]),
        "p_non_disaster": float(arr[2])
    }


def label_with_deepseek(texts):
    """并行调用 DeepSeek 返回软标签"""
    results = Parallel(n_jobs=N_JOBS, backend="threading")(
        delayed(_label_one)(t) for t in texts
    )
    return pd.DataFrame(results)


# ─── 在 Dev 集上评估 ─────────────────────────────────────────────────────────
def eval_dev(model, tokenizer):
    model.eval()
    df = pd.read_csv(DEV_CSV)
    preds = []
    for txt in tqdm(df["text"], desc="Eval Dev"):
        enc = tokenizer(txt, truncation=True, padding="max_length",
                        max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**enc).logits
        raw = logits.argmax(dim=-1).item()
        pred = 1 if raw == 0 else 0
        preds.append(pred)
    acc = accuracy_score(df["target"], preds)
    print(f"→ Dev Acc: {acc*100:.2f}%")
    return acc


# ─── KD+CE 混合微调 ─────────────────────────────────────────────────────────
def train_mixed(train_csv, ckpt_dir, out_dir):
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_dir, local_files_only=True, trust_remote_code=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt_dir,
        local_files_only=True,
        trust_remote_code=True,
        num_labels=2
    ).to(DEVICE)

    # 先冻结所有参数
    for n, p in model.named_parameters():
        p.requires_grad = False

    # 解冻 classifier head
    for n, p in model.named_parameters():
        if n.startswith("classifier."):
            p.requires_grad = True

    # 解冻最后 6 层 encoder
    nl = model.config.num_hidden_layers
    for layer_idx in range(nl-6, nl):
        prefix = f"encoder.layer.{layer_idx}."
        for n, p in model.named_parameters():
            if n.startswith(prefix):
                p.requires_grad = True

    ds = MixedDataset(train_csv, tokenizer)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                    generator=torch.Generator().manual_seed(SEED))
    optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    total = len(dl) * EPOCHS
    sched = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=int(0.1*total),
        num_training_steps=total, num_cycles=0.5
    )
    ce_fn = torch.nn.CrossEntropyLoss()
    kl_fn = torch.nn.KLDivLoss(reduction="batchmean")

    losses = []
    for ep in range(1, EPOCHS+1):
        loop = tqdm(dl, desc=f"FT Epoch {ep}/{EPOCHS}")
        for ids, masks, soft, hard in loop:
            ids, masks = ids.to(DEVICE), masks.to(DEVICE)
            soft, hard = soft.to(DEVICE), hard.to(DEVICE)
            optim.zero_grad()
            logits  = model(input_ids=ids, attention_mask=masks).logits
            loss_kd = kl_fn(F.log_softmax(logits/TEMPERATURE, dim=1), soft)
            loss_ce = ce_fn(logits, hard)
            loss    = ALPHA * loss_kd + (1-ALPHA) * loss_ce
            loss.backward()
            optim.step()
            sched.step()
            losses.append(loss.item())
            loop.set_postfix(loss=loss.item())

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    plt.figure(); plt.plot(losses); plt.title("Mixed KD+CE Finetune")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss.png"))
    plt.close()

    return model, tokenizer


# ─── 全流程主函数 ───────────────────────────────────────────────────────────
def main():
    # 初始化读取原始数据
    orig_pool_df  = pd.read_csv(ORIG_POOL_CSV)
    orig_train_df = pd.read_csv(ORIG_TRAIN_CSV)

    # 初始化模型和 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        Path(CKPT_DIR), local_files_only=True, trust_remote_code=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        Path(CKPT_DIR), local_files_only=True,
        trust_remote_code=True, num_labels=2
    ).to(DEVICE)
    prev_ckpt = CKPT_DIR

    # 拷贝 pool 和 train 初始副本
    pool_df = orig_pool_df.copy()
    train_df = orig_train_df.copy()

    # 初始选样 + 标注
    print(f"\n=== Initial Selection ===")
    idxs = select_by_bald(model, tokenizer,
                          pool_df["text"].tolist(), QUERY_K, MC_ITERS)
    samples = pool_df.iloc[idxs]
    new_df = label_with_deepseek(samples["text"].tolist())
    new_df[["id", "keyword", "location"]] = samples[["id", "keyword", "location"]].reset_index(drop=True)

    train_df = pd.concat([train_df, new_df], ignore_index=True)
    pool_df = pool_df.drop(index=idxs).reset_index(drop=True)
    train_df.to_csv(TRAIN_CSV, index=False)
    pool_df.to_csv(POOL_CSV, index=False)

    # 主动学习循环
    for r in range(1, AL_ROUNDS + 1):
        print(f"\n=== AL Round {r}/{AL_ROUNDS} ===")
        # 备份
        train_df.to_csv(os.path.join(WORK_DIR, f"round{r}_train.csv"), index=False)
        pool_df.to_csv(os.path.join(WORK_DIR, f"round{r}_pool.csv"), index=False)

        # 选样
        idxs = select_by_bald(model, tokenizer,
                              pool_df["text"].tolist(), QUERY_K, MC_ITERS)
        samples = pool_df.iloc[idxs]
        new_df = label_with_deepseek(samples["text"].tolist())
        new_df[["id", "keyword", "location"]] = samples[["id", "keyword", "location"]].reset_index(drop=True)

        # 更新
        train_df = pd.concat([train_df, new_df], ignore_index=True)
        pool_df = pool_df.drop(index=idxs).reset_index(drop=True)
        train_df.to_csv(TRAIN_CSV, index=False)
        pool_df.to_csv(POOL_CSV, index=False)
        print(f"  Added {len(new_df)} samples → train size = {len(train_df)}")

        # 微调 & 评估
        out_dir = os.path.join(WORK_DIR, f"round{r}")
        model, tokenizer = train_mixed(TRAIN_CSV, prev_ckpt, out_dir)
        prev_ckpt = out_dir

        print(f"[Round {r}] Dev 准确率：")
        eval_dev(model, tokenizer)

    print("\n🎉 Active Learning 全流程完成！")


if __name__ == "__main__":
    main()
