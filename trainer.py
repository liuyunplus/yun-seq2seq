import time
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from checkpoint import Checkpoint
from loss import FocalLoss
from model import simpleSeq2Seq
from dataloader import CustomDataLoader, load_vocab, PAD_IDX
import utils


class Trainer(object):

    def __init__(self, source_vocab, target_vocab, model_folder, log_path, num_epochs=100, lr=0.001):
        utils.setup_seed(42)
        self.device = utils.get_device()
        self.model = simpleSeq2Seq(source_vocab, target_vocab).to(self.device)
        self.criterion = FocalLoss(ignore_index=PAD_IDX)
        self.writer = SummaryWriter(log_path)
        self.num_epochs = num_epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.checkpoint = Checkpoint(self.model, self.optimizer, model_folder)

    def _eval_batch(self, eval_iter):
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_iter):
                source, target = batch[0], batch[1]
                source, target = source.to(self.device), target.to(self.device)
                outputs = self.model(source, target, teacher_force_ratio=0)
                outputs = outputs.to(target.device)
                loss = self._calc_loss(outputs, target)
                total_loss += loss.item()
                accuracy = self._calc_metrics(outputs, target)
                total_accuracy += accuracy
        avg_loss = total_loss / len(eval_iter)
        avg_accuracy = total_accuracy / len(eval_iter)
        return avg_loss, avg_accuracy

    def _train_batch(self, train_iter):
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        for batch_idx, batch in enumerate(train_iter):
            source, target = batch[0], batch[1]
            source, target = source.to(self.device), target.to(self.device)
            # clear gradient
            self.optimizer.zero_grad()
            # forward propagation
            outputs = self.model(source, target, teacher_force_ratio=1)
            # outputs shape: (target_len, batch_size, output_dim)
            outputs = outputs.to(target.device)
            loss = self._calc_loss(outputs, target)
            # backpropagation and calculating gradients
            loss.backward()
            # clipping the gradient of the model to prevent the problem of gradient explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            # update model parameters
            self.optimizer.step()
            total_loss += loss.item()
            accuracy = self._calc_metrics(outputs, target)
            total_accuracy += accuracy
        avg_loss = total_loss / len(train_iter)
        avg_accuracy = total_accuracy / len(train_iter)
        return avg_loss, avg_accuracy

    def train(self, train_iter, eval_iter):
        start_epoch, _ = self.checkpoint.load_checkpoint()
        for epoch in range(start_epoch, self.num_epochs):
            start_time = time.time()
            train_loss, train_accuracy = self._train_batch(train_iter)
            eval_loss, eval_accuracy = self._eval_batch(eval_iter)
            end_time = time.time()
            training_time = round(end_time - start_time, 2)
            self._log_scalar(epoch, train_loss, train_accuracy, eval_loss, eval_accuracy, training_time)
            self.checkpoint.save_checkpoint(epoch + 1, train_loss)

    def _calc_loss(self, outputs, target):
        outputs = outputs[1:].reshape(-1, outputs.shape[-1])
        target = target[1:].reshape(-1)
        loss = self.criterion(outputs, target)
        return loss

    def _calc_metrics(self, outputs, targets):
        outputs = outputs.argmax(dim=2)
        outputs = outputs[1:-1].transpose(0, 1)
        targets = targets[1:-1].transpose(0, 1)
        predictions = torch.all(torch.eq(outputs, targets), dim=1)
        accuracy = torch.sum(predictions).item() / len(predictions)
        return accuracy

    def _log_scalar(self, epoch, train_loss, train_accuracy, eval_loss, eval_accuracy, training_time):
        self.writer.add_scalar("Eval/accuracy", eval_accuracy, global_step=epoch + 1)
        self.writer.add_scalar("Eval/loss", eval_loss, global_step=epoch + 1)
        self.writer.add_scalar("Train/accuracy", train_accuracy, global_step=epoch + 1)
        self.writer.add_scalar("Train/loss", train_loss, global_step=epoch + 1)
        print(f'Epoch: [{epoch + 1}/{self.num_epochs}], Train loss: {train_loss}, Train accuracy: {train_accuracy}, '
              f'Eval loss: {eval_loss}, Eval accuracy: {eval_accuracy}, Cost time: {training_time}s')


def train(src_vocab_path, trg_vocab_path, train_csv, eval_csv, model_path, log_path, batch_size=1024, num_epochs=100, lr=0.001):
    source_vocab = load_vocab(src_vocab_path)
    target_vocab = load_vocab(trg_vocab_path)

    train_iter = CustomDataLoader(source_vocab, target_vocab, batch_size, train_csv)
    eval_iter = CustomDataLoader(source_vocab, target_vocab, batch_size, eval_csv)

    trainer = Trainer(source_vocab, target_vocab, model_path, log_path, num_epochs, lr)
    trainer.train(train_iter, eval_iter)
