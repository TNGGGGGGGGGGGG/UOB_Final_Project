import os
import json
import time
import re
import pandas as pd
from openai import OpenAI
from joblib import Parallel, delayed
from threading import Lock
from tqdm.auto import tqdm

# 配置 DeepSeek API Key
os.environ["DEEPSEEK_API_KEY"] = "sk-6b369b3e8d684735b84bbdf7002bc25b"


def call_deepseek(prompt: str,
                  model: str = "deepseek-chat",
                  max_tokens: int = 100,
                  temperature: float = 0.0,
                  top_p: float = 1.0) -> str:
    """
    调用 DeepSeek API 返回模型输出文本
    """
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def load_prompt_template(path: str = "prompt/random_1024(knn与非knn对比).txt") -> str:
    """
    读取 prompt 模板，模板中应包含 {text} 占位
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_input_csv(path: str = "tweets.csv") -> pd.DataFrame:
    """
    读取待打标的 CSV，必须包含 'text' 列
    """
    return pd.read_csv(path)


def process_text(text: str, prompt_tmpl: str) -> list:
    """
    为单条推文生成 [p_disaster, p_non_disaster]。
    使用 prompt_tmpl.format(text=...) 方式填充。
    """
    # JSON 转义并去除外层引号
    safe = json.dumps(text, ensure_ascii=False)[1:-1]
    # 填充 prompt
    prompt = prompt_tmpl.format(text=safe)

    # debug 打印 prompt（可选，调试时打开）
    # print("PROMPT:\n", prompt, "\n-----")

    for _ in range(2):
        raw = call_deepseek(prompt).strip()

        # debug 打印返回（可选，调试时打开）
        # print("RAW RESPONSE:\n", raw, "\n-----")

        # 精准匹配第一个 JSON 数组，不要过度贪婪
        m = re.search(r'(\[[^\]]+\])', raw)
        snippet = m.group(1) if m else raw

        try:
            arr = json.loads(snippet)
            if isinstance(arr, list) and len(arr) == 2:
                a, b = float(arr[0]), float(arr[1])
                total = a + b if (a + b) > 0 else 1.0
                return [round(a / total, 2), round(b / total, 2)]
        except Exception:
            # 解析失败，等待重试
            time.sleep(0.5)

    # 最终 fallback
    return [0.5, 0.5]


def get_soft_labels_parallel(df: pd.DataFrame,
                             prompt_tmpl: str,
                             n_jobs: int = 8) -> pd.DataFrame:
    """
    使用 joblib + tqdm 并行化 DeepSeek API 调用，带真实进度条
    """
    lock = Lock()
    pbar = tqdm(total=len(df), desc="Obtaining soft labels")

    def task_wrapper(text):
        res = process_text(text, prompt_tmpl)
        with lock:
            pbar.update(1)
        return res

    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(task_wrapper)(text) for text in df["text"]
    )
    pbar.close()
    df["p_disaster"], df["p_non_disaster"] = zip(*results)
    return df


def save_distill_csv(df: pd.DataFrame, path: str):
    """
    保存带软标签的 DataFrame 到 CSV
    """
    df.to_csv(path, index=False)
    print(f"Saved to {path}")


def main():
    """
    脚本主流程
    """
    df = load_input_csv("data/generated_1024.csv")
    prompt_tmpl = load_prompt_template("prompt/soft_label.txt")
    df = get_soft_labels_parallel(df, prompt_tmpl, n_jobs=8)
    save_distill_csv(df, "soft_label_distill_data.csv")


if __name__ == "__main__":
    main()
