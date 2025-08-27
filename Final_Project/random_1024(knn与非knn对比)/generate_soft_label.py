#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
import re
import csv
import pandas as pd
from openai import OpenAI
from joblib import Parallel, delayed
from threading import Lock
from tqdm.auto import tqdm

# ——— 配置 ——————————————————————————————————
os.environ["DEEPSEEK_API_KEY"] = "sk-6b369b3e8d684735b84bbdf7002bc25b"
API_BASE    = "https://api.deepseek.com"
MODEL       = "deepseek-chat"
MAX_TOK     = 100
TEMP        = 0.0

# 文件路径
INPUT_CSV   = "compare_SD_sample1024.csv"
PROMPT_FILE = "soft_label.txt"
OUT_CSV     = "compare_random.csv"

# 并行参数
N_JOBS      = 8

# ——— 初始化客户端 —————————————————————————————
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=API_BASE
)

# ——— 读取 Prompt 模板 ————————————————————————————
def load_prompt_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

# ——— 读取含有逗号的 CSV ————————————————————————————
def load_input_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=",",
        engine="python",
        quotechar='"',
        quoting=csv.QUOTE_ALL,
        skipinitialspace=True
    )

# ——— 调用 DeepSeek —————————————————————————————————
def call_deepseek(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": prompt}
        ],
        temperature=TEMP,
        max_tokens=MAX_TOK,
        top_p=1.0
    )
    return resp.choices[0].message.content or ""

# ——— 处理单条文本 —————————————————————————————————
def process_text(text: str, prompt_tmpl: str) -> list:
    safe = json.dumps(text, ensure_ascii=False)[1:-1]
    prompt = prompt_tmpl.format(text=safe)
    for _ in range(2):
        raw = call_deepseek(prompt).strip()
        m = re.search(r'(\[[^\]]+\])', raw)
        snippet = m.group(1) if m else raw
        try:
            arr = json.loads(snippet)
            if isinstance(arr, list) and len(arr) == 2:
                a, b = float(arr[0]), float(arr[1])
                total = (a + b) or 1.0
                return [round(a/total, 2), round(b/total, 2)]
        except:
            time.sleep(0.5)
    return [0.5, 0.5]

# ——— 并行获得软标签 ————————————————————————————
def get_soft_labels_parallel(df: pd.DataFrame, prompt_tmpl: str, n_jobs: int) -> pd.DataFrame:
    lock = Lock()
    pbar = tqdm(total=len(df), desc="Obtaining soft labels")

    def task(text):
        probs = process_text(text, prompt_tmpl)
        with lock:
            pbar.update(1)
        return probs

    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(task)(txt) for txt in df["text"]
    )
    pbar.close()
    df["p_disaster"], df["p_non_disaster"] = zip(*results)
    return df

# ——— 保存结果 —————————————————————————————————
def save_distill_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved to {path}")

# ——— 主流程 —————————————————————————————————————
def main():
    df = load_input_csv(INPUT_CSV)
    prompt_tmpl = load_prompt_template(PROMPT_FILE)
    df = get_soft_labels_parallel(df, prompt_tmpl, n_jobs=N_JOBS)
    save_distill_csv(df, OUT_CSV)

if __name__ == "__main__":
    main()
