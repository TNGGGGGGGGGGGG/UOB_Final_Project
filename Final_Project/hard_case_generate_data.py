import os
import re
import json
import time
import csv
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ——— Configuration ———
os.environ["DEEPSEEK_API_KEY"] = "sk-6b369b3e8d684735b84bbdf7002bc25b"
DEEPSEEK_API_KEY    = os.getenv("DEEPSEEK_API_KEY")
BASE_URL            = "https://api.deepseek.com"
MODEL               = "deepseek-chat"
NUM_GENERATIONS     = 2
TEMPERATURE         = 1.3
TOP_P               = 0.9
MAX_TOKENS          = 1600
MAX_RETRIES         = 3
SLEEP_SECONDS       = 0.3

INPUT_CSV            = "errors.csv"
PROMPT_TEMPLATE_PATH = "prompt/hard_case-aug.txt"
OUTPUT_CSV           = "acl_data_handling/dev_hard_aug.csv"

# ——— Initialize DeepSeek client ———
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

# ——— Load prompt template ———
# 确保 prompt/hard_case-aug.txt 就是你上面最终版的完整 prompt
with open(PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
    prompt_template = f.read().strip()


def build_prompt(row: pd.Series) -> str:
    """
    把 CSV 里的一整行各个字段填入 prompt 模板。
    模板中应包含占位符：
      {id}, {keyword}, {location}, {text},
      {target}, {pred}, {p_disaster}, {p_non_disaster}, {n_gen}
    """
    return prompt_template.format(
        id               = row["id"],  # 使用 id
        keyword          = row.get("keyword", ""),  # 如果没有 keyword 列，则使用默认值空字符串
        location         = row.get("location", ""),  # 如果没有 location 列，则使用默认值空字符串
        text             = row["text"],
        target           = int(row["target"]),
        pred             = int(row["pred"]),
        p_disaster       = row["p_disaster"],
        p_non_disaster   = row["p_non_disaster"],
        n_gen            = NUM_GENERATIONS
    )


def call_deepseek(prompt: str) -> list:
    for _ in range(MAX_RETRIES):
        resp = client.chat.completions.create(
            model       = MODEL,
            messages    = [
                {"role": "system", "content": "You are a tweet enhancement assistant and only return a JSON array."},
                {"role": "user",   "content": prompt}
            ],
            temperature = TEMPERATURE,
            top_p       = TOP_P,
            max_tokens  = MAX_TOKENS,
            stream      = False
        )
        text = resp.choices[0].message.content.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r'(\[.*\])', text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass
        time.sleep(0.5)
    return []


def process_row(idx: int, row: pd.Series):
    """
    对单条 misclassified 样本做增强，
    保留原行的 id/keyword/location/target/p_* 信息。
    """
    prompt = build_prompt(row)
    gens   = call_deepseek(prompt)
    time.sleep(SLEEP_SECONDS)

    out = []
    for t in gens:
        out.append({
            "id":             idx,  # 给每条生成的数据一个唯一的 id（使用索引）
            "keyword":        row.get("keyword", ""),  # 如果没有 keyword 列，则使用默认值空字符串
            "location":       row.get("location", ""),  # 如果没有 location 列，则使用默认值空字符串
            "text":           t,
            "target":         int(row["target"]),
            "p_disaster":     float(row["target"]),           # hard label => one-hot
            "p_non_disaster": 1.0 - float(row["target"])
        })
    return out


def main():
    # 读取原始数据，假设没有 'id' 列
    df = pd.read_csv(INPUT_CSV, engine="python")

    # 给每一行数据添加一个唯一的 'id' 列
    df['id'] = df.index  # 使用 DataFrame 的行索引作为 id 列

    # 如果没有 keyword 或 location 列，添加这两列并初始化为 ""
    if "keyword" not in df.columns:
        df["keyword"] = ""
    if "location" not in df.columns:
        df["location"] = ""

    hard_cases = df[df["target"] != df["pred"]].reset_index(drop=True)
    print(f"Found {len(hard_cases)} misclassified tweets to augment.")

    all_results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(process_row, idx, row)
            for idx, row in hard_cases.iterrows()
        ]
        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc="Augmenting tweets"):
            try:
                all_results.extend(future.result())
            except Exception as e:
                print(f"Error processing row: {e}")

    # Dedup & save，保存生成的数据到 CSV
    result_df = pd.DataFrame(all_results).drop_duplicates(subset=["text"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    result_df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL)
    print(f"Saved {len(result_df)} unique augmented tweets to {OUTPUT_CSV}")


if __name__ == "__main__":
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Please set DEEPSEEK_API_KEY environment variable")
    main()
