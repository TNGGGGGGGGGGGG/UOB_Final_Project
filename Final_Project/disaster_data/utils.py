import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(input_csv, train_csv, dev_csv, test_csv, test_size, dev_size, random_state, stratify_col):
    df = pd.read_csv(input_csv)

    # 先划分测试集
    temp_df, test_df = train_test_split(
        df,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=df[stratify_col]
    )

    # 再从剩下的划分验证集
    train_df, dev_df = train_test_split(
        temp_df,
        test_size=float(dev_size) / (1 - float(test_size)),
        random_state=int(random_state),
        stratify=temp_df[stratify_col]
    )

    train_df.to_csv(train_csv, index=False)
    dev_df.to_csv(dev_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"Train: {len(train_df)} rows")
    print(f"Dev:   {len(dev_df)} rows")
    print(f"Test:  {len(test_df)} rows")

def main():
    split_dataset(
        "train.csv",
        "train_split.csv",
        "dev.csv",
        "test.csv",
        0.1,
        0.1,
        42,
        "target"
    )

if __name__ == "__main__":
    main()
