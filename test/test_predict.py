from predictor import Predictor
import pandas as pd
from dataloader import load_vocab


def predict(src_vocab_path, trg_vocab_path, model_path, epoch, test_csv, predict_csv):
    source_vocab = load_vocab(src_vocab_path)
    target_vocab = load_vocab(trg_vocab_path)
    predictor = Predictor(source_vocab, target_vocab, model_path, epoch)
    df = pd.read_csv(test_csv)
    df['predict'] = None
    for index, row in df.iterrows():
        words = str(row['source']).split(" ")
        outputs = predictor.predict(words, 3)
        predict = ",".join(outputs)
        df.at[index, 'predict'] = predict
        print(f"{row['domain']} => {predict}")

    df.to_csv(predict_csv, index=False)
    accuracy = (df['domain'] == df['predict']).sum() / len(df)
    print(f'Accuracy: {accuracy}')


BASE_PATH = "./data"
source_vocab = f"{BASE_PATH}/source_vocab.pkl"
target_vocab = f"{BASE_PATH}/target_vocab.pkl"
test_csv = f"{BASE_PATH}/test.csv"
predict_csv = f"{BASE_PATH}/predict.csv"

model_path = f"{BASE_PATH}/models/train_model_v1"
epoch = 12

predict(source_vocab, target_vocab, model_path, epoch, test_csv, predict_csv)

