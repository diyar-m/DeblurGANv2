import logging
from functools import partial

import cv2
import numpy as np
import torch
import torch.optim as optim
import tqdm
import yaml
from joblib import cpu_count
from torch.utils.data import DataLoader

from adversarial_trainer import GANFactory
from dataset import PairedDataset
from metric_counter import MetricCounter
from models.losses import get_loss
from models.models import get_model
from models.networks import get_nets
from schedulers import LinearDecay, WarmRestart

cv2.setNumThreads(0)


class Validator:
    def __init__(self, config, val: DataLoader):
        self.config = config
        self.val_dataset = val
        self.metric_counter = MetricCounter()

    def validate(self, resume_train=False):
        self._init_params()
        self.netG.load_state_dict(torch.load(self.config['val']['model_path'])['model'])
        self._validate()
        torch.cuda.empty_cache()

    def _validate(self):
        self.metric_counter.clear()
        epoch_size = config.get('val_batches_per_epoch') or len(self.val_dataset)
        tq = tqdm.tqdm(self.val_dataset, total=epoch_size)
        tq.set_description('Validation')
        sum_psnr = 0
        sum_ssim = 0
        sum_lens = 0
        i = 0
        self.model.train(True)
        for data in tq:
            inputs, targets = self.model.get_input(data)
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = self.netG(inputs)
            
            curr_psnr, curr_ssim, img_for_vis = self.model.get_images_and_metrics(inputs, outputs, targets)
            print(curr_psnr)
            sum_psnr += curr_psnr
            sum_ssim += curr_ssim
            sum_lens += len(inputs)
            self.metric_counter.add_metrics(curr_psnr, curr_ssim)
            if not i%50:
                self.metric_counter.add_image(img_for_vis, tag='val')
            i += 1
            # if i > epoch_size:
            #     break
            del inputs, targets, outputs
            self.metric_counter.write_to_tensorboard(i, validation=True)

        tq.close()
        print("SSIM ", sum_ssim / sum_lens)
        print("PSNR ", sum_psnr / sum_lens)
        print("Dataset Length ", sum_lens)

    def _init_params(self):
        self.netG, _ = get_nets(self.config['model'])
        self.netG.cuda()
        self.model = get_model(self.config['model'])

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

if __name__ == '__main__':
    with open('config/config.yaml', 'r') as f:
        config = yaml.load(f)

    batch_size = config.pop('batch_size')
    torch.manual_seed(0)
    get_dataloader = partial(DataLoader, batch_size=batch_size, num_workers=cpu_count(), shuffle=True, drop_last=True)

    datasets = map(config.get, ('train', 'val'))
    datasets = map(PairedDataset.from_config, datasets)
    train, val = map(get_dataloader, datasets)
    validator = Validator(config, val=val)

    validator.validate()
