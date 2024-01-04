from trainer import train

BASE_PATH = "./data"
source_vocab = f"{BASE_PATH}/source_vocab.pkl"
target_vocab = f"{BASE_PATH}/target_vocab.pkl"
train_csv = f"{BASE_PATH}/train.csv"
eval_csv = f"{BASE_PATH}/eval.csv"

model_name = "train_model_v1"
model_path = f"{BASE_PATH}/models/{model_name}"
log_path = f"{BASE_PATH}/logs/{model_name}"

train(source_vocab, target_vocab, train_csv, eval_csv, model_path, log_path, batch_size=1024, num_epochs=1000, lr=0.001)