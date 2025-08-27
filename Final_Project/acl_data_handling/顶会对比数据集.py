import pandas as pd

# 1. 读入原始 TSV，只要四列
df = pd.read_csv(
    'task_informative_text_img_train.tsv',
    sep='\t',
    usecols=['event_name', 'tweet_id', 'tweet_text', 'label']
)

# 2. 映射 label 为数值
mapping = {
    'informative':     1,
    'not_informative': 0
}
df['label'] = df['label'].map(mapping)

# 3. 检查是否有未映射的
if df['label'].isnull().any():
    missing = df.loc[df['label'].isnull(), 'label'].unique()
    raise ValueError(f"发现未映射的 label 值: {missing}")

# 4. 重命名列以匹配推理脚本
df = df.rename(columns={
    'tweet_text': 'text',
    'label':      'target'
})

# 5. 只保留模型需要的两列
df_test = df[['text', 'target']]

# 6. 保存为 test.csv
df_test.to_csv('emnlp_compare_train.csv', index=False)

print("已生成 test.csv，包含列：", df_test.columns.tolist())
print(df_test.head())
