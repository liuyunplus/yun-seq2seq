import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
import pickle

UNK, PAD, BOS, EOS = '<unk>', '<pad>', '<bos>', '<eos>'
SPECIALS = [PAD, BOS, EOS, UNK]
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = SPECIALS.index(UNK), SPECIALS.index(PAD), SPECIALS.index(BOS), SPECIALS.index(EOS)


class CustomDataset(Dataset):

    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        features = str(row['source']).split(" ")
        targets = str(row['target']).split(" ")
        return features, targets


class CustomDataLoader(object):

    def __init__(self, source_vocab, target_vocab, batch_size, file_path):
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        dataset = CustomDataset(file_path)
        self.data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True,
                                      collate_fn=self.generate_batch)

    def __iter__(self):
        return iter(self.data_loader)

    def __len__(self):
        return len(self.data_loader)

    def generate_batch(self, data_batch):
        BOS_IDX = self.source_vocab[BOS]
        EOS_IDX = self.source_vocab[EOS]
        PAD_IDX = self.source_vocab[PAD]
        src_batch, trg_batch = [], []
        for src_tokens, trg_tokens in data_batch:
            src_tensor = torch.tensor(input_encode(self.source_vocab, src_tokens), dtype=torch.long)
            # Append <bos> and <eos> before and after
            src_tensor = torch.cat([torch.tensor([BOS_IDX]), src_tensor, torch.tensor([EOS_IDX])])
            src_batch.append(src_tensor)
            trg_tensor = torch.tensor(input_encode(self.target_vocab, trg_tokens), dtype=torch.long)
            # Append <bos> and <eos> before and after
            trg_tensor = torch.cat([torch.tensor([BOS_IDX]), trg_tensor, torch.tensor([EOS_IDX])])
            trg_batch.append(trg_tensor)
        # Ensure that each line has the same length, and lines with insufficient length are filled with PAD.
        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
        trg_batch = pad_sequence(trg_batch, padding_value=PAD_IDX)
        return src_batch, trg_batch


def input_encode(vocab, tokens):
    vocab_map = vocab.get_stoi()
    unk_token = vocab_map.get(UNK, None)
    return [vocab_map.get(token, unk_token) for token in tokens]


def output_decode(target_vocab, outputs):
    result_list = []
    for row in outputs:
        words = []
        for item in row:
            word = target_vocab.get_itos()[item]
            if word in [EOS, UNK, PAD, BOS]:
                continue
            words.append(word)
        result_list.append(" ".join(words))
    return result_list


def build_vocab(file_path, place):
    dataset = CustomDataset(file_path)

    def get_tokens(data_iter, place):
        for features, target in data_iter:
            if place == 0:
                yield features
            else:
                yield target

    vocab = build_vocab_from_iterator(
        get_tokens(dataset, place),
        min_freq=1,
        specials=SPECIALS,
        special_first=True
    )
    return vocab


def save_vocab(word_vocab, filename):
    with open(filename, 'wb') as f:
        pickle.dump(word_vocab, f)


def load_vocab(filename):
    with open(filename, 'rb') as f:
        word_vocab = pickle.load(f)
    return word_vocab
