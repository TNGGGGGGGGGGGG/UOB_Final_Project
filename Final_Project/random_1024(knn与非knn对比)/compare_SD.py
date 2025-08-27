#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
from openai import OpenAI
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# ——— 配置 ——————————————————————————————————

os.environ["DEEPSEEK_API_KEY"] = "sk-6b369b3e8d684735b84bbdf7002bc25b"
API_BASE = "https://api.deepseek.com"
MODEL    = "deepseek-chat"
MAX_TOK  = 1600
TEMP     = 1.3

ORIG_CSV    = "C:/Users/罗腾霄/PycharmProjects/Final_Project/data/generated_1024.csv"
PROMPT_FILE = "compare_SD.txt"
OUT_CSV     = "compare_SD_augmented.csv"

N_GEN       = 2    # 每条生成 2 条示例
N_JOBS      = 8    # 并行进程数

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url=API_BASE)

with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    PROMPT_TPL = f.read().strip()

def gen_similar(text: str, target: int, n: int = N_GEN):
    prompt = PROMPT_TPL.format(text=text, target=target, n=n)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": prompt}
        ],
        temperature=TEMP,
        max_tokens=MAX_TOK
    )
    out = resp.choices[0].message.content.strip()
    results = []
    for line in out.splitlines():
        try:
            obj = json.loads(line)
            results.append(obj)
        except:
            continue
    return results

def process_record(rec):
    """
    输入：{'text': ..., 'target': ...}
    输出：列表，包含原记录和 N_GEN 条新生成记录
    """
    base = {"text": rec["text"], "target": rec["target"]}
    out = [base]
    gens = gen_similar(rec["text"], rec["target"], n=N_GEN)
    out.extend(gens)
    return out

def main():
    df = pd.read_csv(ORIG_CSV)
    records = df[["text", "target"]].to_dict("records")

    # 并行调用
    all_lists = Parallel(
        n_jobs=N_JOBS,
        backend="threading"
    )(
        delayed(process_record)(rec)
        for rec in tqdm(records, desc="Augmenting")
    )

    # 扁平化
    flat = [item for sublist in all_lists for item in sublist]
    out_df = pd.DataFrame(flat)
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved augmented data: {len(out_df)} rows → {OUT_CSV}")

if __name__ == "__main__":
    main()
