import os
import json
import time
import re
import pandas as pd
from openai import OpenAI
from joblib import Parallel, delayed
import csv
from tqdm.auto import tqdm

# ——— Configuration ———
os.environ["DEEPSEEK_API_KEY"] = "sk-6b369b3e8d684735b84bbdf7002bc25b"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL         = "https://api.deepseek.com"
MODEL            = "deepseek-chat"
MAX_TOKENS       = 100
TEMPERATURE      = 0.0
TOP_P            = 1.0
RETRIES          = 2
PAUSE_SECONDS    = 0.5
NUM_WORKERS      = 8

INPUT_CSV        = "acl_data_handling/dev_hard_aug.csv"
PROMPT_FILE      = "prompt/hard_case_soft_label.txt"
OUTPUT_CSV       = "acl_data_handling/hard_case_soft_label_distill_data.csv"

# ——— Initialize DeepSeek client ———
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

# ——— Load system prompt (should contain placeholders {text} and {target}) ———
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    system_prompt_template = f.read().strip()

# ——— API Call with retry ———
def get_probabilities(text: str, target: int):
    prompt = system_prompt_template.format(text=text, target=target)
    for _ in range(RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user",   "content": f"Tweet: \"{text}\""}
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
            )
            raw = resp.choices[0].message.content.strip()
            try:
                arr = json.loads(raw)
                if isinstance(arr, list) and len(arr) == 2:
                    return round(float(arr[0]), 2), round(float(arr[1]), 2)
            except:
                pass
            m = re.search(r'\[(\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*)\]', raw)
            if m:
                arr = json.loads(f'[{m.group(1)}]')
                return round(float(arr[0]), 2), round(float(arr[1]), 2)
        except Exception as e:
            pass
        time.sleep(PAUSE_SECONDS)
    return 0.5, 0.5  # fallback

# ——— Multi-threaded labeling using joblib ———
def label_all(texts, targets):
    return Parallel(n_jobs=NUM_WORKERS, backend="threading")(
        delayed(get_probabilities)(text, target)
        for text, target in tqdm(zip(texts, targets), total=len(texts), desc="Labeling tweets")
    )

# ——— Main Process ———
def main():
    df = pd.read_csv(INPUT_CSV, engine="python")
    texts   = df["text"].tolist()
    targets = df["target"].tolist()

    probs = label_all(texts, targets)
    df["p_disaster"]     = [p[0] for p in probs]
    df["p_non_disaster"] = [p[1] for p in probs]

    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL)
    print(f"✅ Saved soft-label distillation data ({len(df)} rows) to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
