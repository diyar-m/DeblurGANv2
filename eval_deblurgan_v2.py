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


class Trainer:
    def __init__(self, config, val: DataLoader):
        self.config = config
        self.val_dataset = val
        self.metric_counter = MetricCounter()

    def validate(self):
        self._init_params()
        self.netG.load_state_dict(torch.load('best_G_fpn.h5')['model'])
        self.netG.train(True)

        self._validate()
        torch.cuda.empty_cache()

        print(self.metric_counter.loss_message())

    def _validate(self):
        self.metric_counter.clear()
        epoch_size = config.get('val_batches_per_epoch') or len(self.train_dataset)
        tq = tqdm.tqdm(self.val_dataset, total=epoch_size)
        tq.set_description('Validation')
        i = 0
        total_psnr = 0
        total_ssim = 0
        total_samples = 0
        for data in tq:
            inputs, targets = self.model.get_input(data)
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = self.netG(inputs)
            curr_psnr, curr_ssim, img_for_vis = self.model.get_images_and_metrics(inputs, outputs, targets)
            total_ssim += curr_ssim * len(inputs)
            total_psnr += curr_psnr * len(inputs)
            total_samples += len(inputs)

            print("Metrcis:", curr_psnr, curr_ssim)
            print("Totals:", total_ssim, total_psnr, total_samples)
            print("nan:", np.isnan(img_for_vis).any())
            print("inf:", np.isinf(img_for_vis).any())
            self.metric_counter.add_metrics(curr_psnr, curr_ssim)
            if not i:
                self.metric_counter.add_image(img_for_vis, tag='val')
            i += 1
            if i > epoch_size:
                break
            self.metric_counter.write_to_tensorboard(i, validation=True)
            del inputs, targets, outputs

        print("PSNR", total_psnr/total_samples)
        print("SSIM", total_ssim/total_samples)

        tq.close()


    def _init_params(self):
        self.netG, netD = get_nets(self.config['model'])
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

    dataset = config.pop('val')
    dataset = PairedDataset.from_config(dataset)
    val = get_dataloader(dataset)
    validator = Validator(config, val=val)
    validator.validate()
