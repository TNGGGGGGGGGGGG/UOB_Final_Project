import pandas as pd
import re

# 假设你的 tsv 文件路径是 'task_informative_text_img_train.tsv'
file_path = 'task_informative_text_img_dev.tsv'

# 从 TSV 文件读取数据
df = pd.read_csv(file_path, sep='\t')

# 将 'label_text' 中的 'informative' 转换为 1，'not_informative' 转换为 0
df['label_text'] = df['label_text'].apply(lambda x: 1 if x == 'informative' else 0)

# 选择 'tweet_text' 和 'label_text' 列
result_df = df[['tweet_text', 'label_text']]

# 数据清洗函数
def clean_text(text):
    # 小写化文本
    text = text.lower()

    # 移除 URL
    text = re.sub(r'http\S+|https\S+', '', text)

    # 移除 mentions（以 @ 开头的词）
    text = re.sub(r'@\S+', '', text)

    # 移除 RT 标记（转发标记）
    text = re.sub(r'\brt\b', '', text)

    # 移除非 ASCII 字符
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # 移除单字符（长度为 1 的字符），可选，如果觉得有用可以跳过这步
    text = re.sub(r'\b\w\b', '', text)

    # 去除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# 对 'tweet_text' 列应用清洗函数
result_df['tweet_text'] = result_df['tweet_text'].apply(clean_text)

# 去除重复的 'tweet_text'，保留唯一的
result_df = result_df.drop_duplicates(subset=['tweet_text'])

result_df.columns = ['text', 'target']

# 保存结果到一个新的 CSV 文件 'emnlp_train.csv'
result_df.to_csv('emnlp_dev_cleaned.csv', index=False)

# 显示保存后的文件内容（可选）
print("Data has been cleaned and saved to 'emnlp_train_cleaned.csv'. Here's a preview:")
print(result_df.head())
