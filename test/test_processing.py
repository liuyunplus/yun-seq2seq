import jieba
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from dataloader import build_vocab, save_vocab


def read_df(csv_path):
    return pd.read_csv(csv_path)


def save_df(df, csv_path):
    df.to_csv(csv_path, index=False)


def remove_punctuation(sentence):
    punctuations = string.punctuation
    sentence_no_punct = sentence.translate(str.maketrans('', '', punctuations))
    return sentence_no_punct


def tokenize(df, column):
    def func(text):
        words = jieba.lcut(text, cut_all=False)
        return " ".join(words)
    df[column] = df[column].apply(func)
    return df


BASE_PATH = "./data"
df = read_df(f'{BASE_PATH}/raw_dataset.csv')
df['source'] = df['source'].apply(remove_punctuation)
df['target'] = df['target'].apply(remove_punctuation)
df = tokenize(df, "source")
save_df(df, f"{BASE_PATH}/full_dataset.csv")

train_df, test_df = train_test_split(df, test_size=0.1)
train_df, eval_df = train_test_split(train_df, test_size=0.05)
print(f"total size: {len(df)}, train size: {len(train_df)}, eval size: {len(eval_df)}, test size: {len(test_df)}")
save_df(train_df, f"{BASE_PATH}/train.csv")
save_df(eval_df, f"{BASE_PATH}/eval.csv")
save_df(test_df, f"{BASE_PATH}/test.csv")

# build vocab
source_vocab = build_vocab(f"{BASE_PATH}/full_dataset.csv", 0)
save_vocab(source_vocab, f"{BASE_PATH}/source_vocab.pkl")

target_vocab = build_vocab(f"{BASE_PATH}/full_dataset.csv", 1)
save_vocab(target_vocab, f"{BASE_PATH}/target_vocab.pkl")
