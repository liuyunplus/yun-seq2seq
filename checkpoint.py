from pathlib import Path
import glob
import os
import torch
import utils


class Checkpoint(object):

    def __init__(self, model, optimizer, model_folder):
        self.model = model
        self.optimizer = optimizer
        self.model_folder = model_folder

    def save_checkpoint(self, epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        Path(self.model_folder).mkdir(parents=True, exist_ok=True)
        epoch = str(epoch).zfill(3)
        torch.save(checkpoint, f'{self.model_folder}/checkpoint_{epoch}.pth')

    def load_checkpoint(self, epoch=None):
        checkpoint = self.get_checkpoint(epoch)
        if checkpoint is None:
            return 0, None
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return epoch, loss

    def get_checkpoint(self, epoch=None):
        if epoch:
            epoch = str(epoch).zfill(3)
            files = glob.glob(os.path.join(self.model_folder, f'checkpoint_{epoch}.pth'))
        else:
            files = sorted(glob.glob(os.path.join(self.model_folder, '*.pth')))
        if len(files) == 0:
            return None
        checkpoint = torch.load(files[-1], map_location=utils.get_device())
        return checkpoint
