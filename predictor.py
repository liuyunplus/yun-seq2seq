import torch
import torch.optim as optim
import utils
from checkpoint import Checkpoint
from dataloader import input_encode, output_decode, BOS_IDX, EOS_IDX
from model import simpleSeq2Seq


class Predictor(object):

    def __init__(self, source_vocab, target_vocab, model_folder, epoch, lr=0.001):
        utils.setup_seed(42)
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.device = utils.get_device()
        self.model = simpleSeq2Seq(source_vocab, target_vocab).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.checkpoint = Checkpoint(self.model, self.optimizer, model_folder)
        self.checkpoint.load_checkpoint(epoch)

    def predict(self, words, max_output_len):
        self.model.eval()
        with torch.no_grad():
            features = input_encode(self.source_vocab, words)
            src_tensor = torch.tensor([BOS_IDX] + features + [EOS_IDX], dtype=torch.long).unsqueeze(-1)
            trg_tensor = torch.tensor([BOS_IDX] + [0] * max_output_len, dtype=torch.long).unsqueeze(-1)

            src_tensor, trg_tensor = src_tensor.to(self.device), trg_tensor.to(self.device)
            outputs = self.model(src_tensor, trg_tensor, teacher_force_ratio=0)
            outputs = torch.argmax(outputs, dim=2)
            outputs = outputs[1:].transpose(0, 1)

            result_list = output_decode(self.target_vocab, outputs.tolist())
        return result_list