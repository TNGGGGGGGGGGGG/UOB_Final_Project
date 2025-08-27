import os
import re
import time
import json
import pandas as pd
from joblib import Parallel, delayed

os.environ["DEEPSEEK_API_KEY"] = "sk-29f908f3aff94a4284260e36fc75eae5"

# —— 载入 Prompt 模板 ——
def load_prompt_template(path="prompt_template.txt") -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

# —— 构建 Prompt ——
def build_prompt_for_row(template: str, text: str, target: int) -> str:
    # 强调 target 和生成的内容应该严格匹配
    if target == 1:
        prompt = template.format(text=text, target=1)
    elif target == 0:
        prompt = template.format(text=text, target=0)
    else:
        raise ValueError("Target must be either 0 or 1")
    return prompt

# —— DeepSeek 调用 ——
def call_deepseek(prompt: str) -> str:
    from openai import OpenAI as DeepSeekClient
    client = DeepSeekClient(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        temperature=1.3,
        top_p=0.9,
        max_tokens=1600,
    )
    return resp.choices[0].message.content

# —— OpenAI GPT 调用 ——
def call_gpt_api(prompt: str) -> str:
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.8,
        top_p=0.9,
        max_tokens=500,
    )
    return resp.choices[0].message.content

# —— 解析输出 ——
def parse_singles_from_text(text: str) -> list:
    pattern = r'\{\s*"text":\s*"(.+?)",\s*"target":\s*(\d)\s*\}'
    matches = re.findall(pattern, text, re.DOTALL)
    return [{"text": m[0].strip(), "target": int(m[1])} for m in matches]

# —— 每条原始推文的处理函数 ——
def process_row(idx, row, template, use_deepseek):
    # 获取 target 并打印
    target = int(row["target"])
    print(f"\n[示例] 输入的 Target: {target}")  # 打印 target 值，确保正确
    prompt = build_prompt_for_row(template, row["text"], target)

    # 只在第一次生成时打印 Prompt
    if idx == 0:  # 可以改为检查其他条件，只在第一次生成时打印 Prompt
        print(f"[示例] 输入的 Prompt：")
        print(f"Text: {row['text']}")
        print(f"Target: {row['target']}")
        print(f"Prompt: {prompt}\n")

    try:
        output = call_deepseek(prompt) if use_deepseek else call_gpt_api(prompt)
        tweets = parse_singles_from_text(output)
        if len(tweets) != 4:
            print(f"[{idx+1}] 警告：解析到 {len(tweets)} 条，预期 4 条")
    except Exception as e:
        print(f"[{idx+1}] 出错：{e}")
        tweets = []

    # 把 keyword/location 加回去并立即打印
    results = []
    for tw in tweets:
        rec = {
            "keyword": row.get("keyword", "") or "",
            "location": row.get("location", "") or "",
            "text": tw["text"],
            "target": tw["target"]
        }
        # 实时打印每条生成的新样本 JSON
        print(json.dumps(rec, ensure_ascii=False))
        results.append(rec)

    # 随机延迟，防止短时间过多请求
    time.sleep(0.5)
    return results

def main():
    # 1. 读取数据和模板
    df = pd.read_csv("E:/work/Final_Project/acl_data_handling/expanded_dataset.csv", dtype=str)
    template = load_prompt_template("prompt/prompt_template.txt")
    use_deepseek = True   # 切换 DeepSeek 或 OpenAI

    # 2. 并行调用：n_jobs 设置线程数，backend 'threading' 适合 I/O
    all_batches = Parallel(n_jobs=8, backend="threading")(
        delayed(process_row)(idx, row, template, use_deepseek)
        for idx, row in df.iterrows()
    )

    # 3. 展平并打 ID
    flat = [item for batch in all_batches for item in batch]
    for i, rec in enumerate(flat, start=1):
        rec["id"] = i

    # 4. 保存 CSV
    out_df = pd.DataFrame(flat, columns=["id","keyword","location","text","target"])
    out_df.to_csv("acl_data_handling/generated_1024.csv", index=False, encoding="utf-8-sig")
    print(f"完成：共生成 {len(out_df)} 条样本，保存在 generated_1024.csv")

if __name__ == "__main__":
    main()
