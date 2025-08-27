import os
import pandas as pd

# 配置
AUG_CSV     = "true_data.csv"  # 增强后总数据文件
OUTPUT_DIR  = "./"  # 输出文件夹
SEED        = 42
SIZES       = [32, 64, 128, 256, 512, 1024]

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 读取一次增强后的全部数据
df = pd.read_csv(AUG_CSV)

for size in SIZES:
    # 随机抽取 size 条
    sample_df = df.sample(n=size, random_state=SEED).reset_index(drop=True)
    out_path = os.path.join(OUTPUT_DIR, f"train_size{size}.csv")
    sample_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved {size} samples → {out_path}")
