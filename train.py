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
    def __init__(self, config, train: DataLoader, val: DataLoader):
        self.config = config
        self.train_dataset = train
        self.val_dataset = val
        self.adv_lambda = config['model']['adv_lambda']
        self.metric_counter = MetricCounter()
        self.warmup_epochs = config['warmup_num']

    def train(self, resume_train=False):
        self._init_params()
        start_epoch = 0
        if resume_train:
            start_epoch += self.config['resume']['resume_epoch']
            if start_epoch > self.warmup_epochs:
                self.netG.module.unfreeze()
                self.optimizer_G = self._get_optim(self.netG.parameters())
                self.scheduler_G = self._get_scheduler(self.optimizer_G)
            self.netG.load_state_dict(torch.load('last_G_fpn.h5')['model'])
            self.adv_trainer.patch_d.load_state_dict(torch.load('last_patch_d_fpn.h5')['model'])
            self.adv_trainer.full_d.load_state_dict(torch.load('last_full_d_fpn.h5')['model'])
            self.scheduler_G.load_state_dict(torch.load('last_scheduler_G_fpn.h5')['model'])
            self.scheduler_D.load_state_dict(torch.load('last_scheduler_D_fpn.h5')['model'])
            self.optimizer_G.load_state_dict(torch.load('last_optimizer_G_fpn.h5')['model'])
            self.optimizer_D.load_state_dict(torch.load('last_optimizer_D_fpn.h5')['model'])
            self.optimizer_D.load_state_dict(torch.load('last_optimizer_D_fpn.h5')['model'])
        for epoch in range(start_epoch, config['num_epochs']):
            if (epoch == self.warmup_epochs) and not (self.warmup_epochs == 0):
                self.netG.module.unfreeze()
                self.optimizer_G = self._get_optim(self.netG.parameters(), 'G')
                self.scheduler_G = self._get_scheduler(self.optimizer_G, 'G')
            self._run_epoch(epoch)
            torch.cuda.empty_cache()
            self._validate(epoch)
            torch.cuda.empty_cache()
            self.scheduler_G.step()
            self.scheduler_D.step()

            if self.metric_counter.update_best_model():
                torch.save({
                    'model': self.netG.state_dict()
                }, 'best_G_{}.h5'.format(self.config['experiment_desc']))
                torch.save({
                    'model': self.adv_trainer.patch_d.state_dict()
                }, 'best_patch_d_{}.h5'.format(self.config['experiment_desc']))

                torch.save({
                    'model': self.scheduler_G.state_dict()
                }, 'best_scheduler_G_{}.h5'.format(self.config['experiment_desc']))
                torch.save({
                    'model': self.scheduler_D.state_dict()
                }, 'best_scheduler_D_{}.h5'.format(self.config['experiment_desc']))
                torch.save({
                    'model': self.optimizer_G.state_dict()
                }, 'best_optimizer_G_{}.h5'.format(self.config['experiment_desc']))
                torch.save({
                    'model': self.optimizer_D.state_dict()
                }, 'best_optimizer_D_{}.h5'.format(self.config['experiment_desc']))

            torch.save({
                'model': self.netG.state_dict()
            }, 'last_G_{}.h5'.format(self.config['experiment_desc']))
            torch.save({
                'model': self.adv_trainer.patch_d.state_dict()
            }, 'last_patch_d_{}.h5'.format(self.config['experiment_desc']))
            torch.save({
                'model': self.adv_trainer.full_d.state_dict()
            }, 'last_full_d_{}.h5'.format(self.config['experiment_desc']))
            torch.save({
                'model': self.scheduler_G.state_dict()
            }, 'last_scheduler_G_{}.h5'.format(self.config['experiment_desc']))
            torch.save({
                'model': self.scheduler_D.state_dict()
            }, 'last_scheduler_D_{}.h5'.format(self.config['experiment_desc']))
            torch.save({
                'model': self.optimizer_G.state_dict()
            }, 'last_optimizer_G_{}.h5'.format(self.config['experiment_desc']))
            torch.save({
                'model': self.optimizer_D.state_dict()
            }, 'last_optimizer_D_{}.h5'.format(self.config['experiment_desc']))

            print(self.metric_counter.loss_message())
            logging.debug("Experiment Name: %s, Epoch: %d, Loss: %s" % (
                self.config['experiment_desc'], epoch, self.metric_counter.loss_message()))

    def _run_epoch(self, epoch):
        self.metric_counter.clear()
        for param_group in self.optimizer_G.param_groups:
            lr = param_group['lr']

        epoch_size = config.get('train_batches_per_epoch') or len(self.train_dataset)
        tq = tqdm.tqdm(self.train_dataset, total=epoch_size)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        i = 0
        for data in tq:
            inputs, targets = self.model.get_input(data)
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = self.netG(inputs)
            loss_D = self._update_d(inputs, outputs, targets)
            self.optimizer_G.zero_grad()
            loss_content = self.criterionG(outputs, targets)
            loss_adv = self.adv_trainer.loss_g(inputs, outputs, targets)
            loss_G = loss_content + self.adv_lambda * loss_adv
            loss_G.backward()
            self.optimizer_G.step()
            self.metric_counter.add_losses(loss_G.detach().item(), loss_content.detach().item(), loss_D)
            curr_psnr, curr_ssim, img_for_vis = self.model.get_images_and_metrics(inputs, outputs, targets)
            self.metric_counter.add_metrics(curr_psnr, curr_ssim)
            tq.set_postfix(loss=self.metric_counter.loss_message())
            if not i:
                self.metric_counter.add_image(img_for_vis, tag='train')
            i += 1
            self.metric_counter.write_to_tensorboard(epoch * epoch_size + i)
            if i > epoch_size:
                break
            del inputs, targets, outputs
        tq.close()

    def _validate(self, epoch):
        self.metric_counter.clear()
        epoch_size = config.get('val_batches_per_epoch') or len(self.val_dataset)
        tq = tqdm.tqdm(self.val_dataset, total=epoch_size)
        tq.set_description('Validation')
        i = 0
        for data in tq:
            inputs, targets = self.model.get_input(data)
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = self.netG(inputs)
            # Checked
            loss_content = self.criterionG(outputs, targets)
            loss_adv = self.adv_trainer.loss_g(inputs, outputs, targets)
            loss_G = loss_content + self.adv_lambda * loss_adv
            self.metric_counter.add_losses(loss_G.detach().item(), loss_content.detach().item())
            curr_psnr, curr_ssim, img_for_vis = self.model.get_images_and_metrics(inputs, outputs, targets)
            self.metric_counter.add_metrics(curr_psnr, curr_ssim)
            if not i:
                self.metric_counter.add_image(img_for_vis, tag='val')
            i += 1
            if i > epoch_size:
                break
            del inputs, targets, outputs

        tq.close()
        self.metric_counter.write_to_tensorboard(epoch, validation=True)

    def _update_d(self, inputs, outputs, targets):
        if self.config['model']['d_name'] == 'no_gan':
            return 0
        self.optimizer_D.zero_grad()
        # Checked
        loss_D = self.adv_lambda * self.adv_trainer.loss_d(inputs, outputs, targets)
        loss_D.backward(retain_graph=True)
        self.optimizer_D.step()
        return loss_D.detach().item()

    def _get_optim(self, params, net_type):
        if net_type == 'G':
            lr = self.config['optimizer']['lr_G']
        else:
            lr = self.config['optimizer']['lr_D']
        if self.config['optimizer']['name'] == 'adam':
            optimizer = optim.Adam(params, lr=lr)
        elif self.config['optimizer']['name'] == 'sgd':
            optimizer = optim.SGD(params, lr=lr)
        elif self.config['optimizer']['name'] == 'adadelta':
            optimizer = optim.Adadelta(params, lr=lr)
        else:
            raise ValueError("Optimizer [%s] not recognized." % self.config['optimizer']['name'])
        return optimizer

    def _get_scheduler(self, optimizer, net_type):
        if net_type == 'G':
            min_lr = self.config['scheduler']['min_lr_G']
        else:
            min_lr = self.config['scheduler']['min_lr_D']
        if self.config['scheduler']['name'] == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             mode='min',
                                                             patience=self.config['scheduler']['patience'],
                                                             factor=self.config['scheduler']['factor'],
                                                             min_lr=min_lr)
        elif self.config['optimizer']['name'] == 'sgdr':
            scheduler = WarmRestart(optimizer)
        elif self.config['scheduler']['name'] == 'linear':
            scheduler = LinearDecay(optimizer,
                                    min_lr=min_lr,
                                    num_epochs=self.config['num_epochs'],
                                    start_epoch=self.config['scheduler']['start_epoch'])
        else:
            raise ValueError("Scheduler [%s] not recognized." % self.config['scheduler']['name'])
        return scheduler

    @staticmethod
    def _get_adversarial_trainer(d_name, net_d, criterion_d):
        if d_name == 'no_gan':
            return GANFactory.create_model('NoGAN')
        elif d_name == 'patch_gan' or d_name == 'multi_scale':
            return GANFactory.create_model('SingleGAN', net_d, criterion_d)
        elif d_name == 'double_gan':
            return GANFactory.create_model('DoubleGAN', net_d, criterion_d)
        else:
            raise ValueError("Discriminator Network [%s] not recognized." % d_name)

    def _init_params(self):
        self.criterionG, criterionD = get_loss(self.config['model'])
        self.netG, netD = get_nets(self.config['model'])
        self.netG.cuda()
        self.adv_trainer = self._get_adversarial_trainer(self.config['model']['d_name'], netD, criterionD)
        self.model = get_model(self.config['model'])
        self.optimizer_G = self._get_optim(filter(lambda p: p.requires_grad, self.netG.parameters()), 'G')
        self.optimizer_D = self._get_optim(self.adv_trainer.get_params(), 'D')
        self.scheduler_G = self._get_scheduler(self.optimizer_G, 'G')
        self.scheduler_D = self._get_scheduler(self.optimizer_D, 'D')

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

    datasets = map(config.pop, ('train', 'val'))
    datasets = map(PairedDataset.from_config, datasets)
    train, val = map(get_dataloader, datasets)
    trainer = Trainer(config, train=train, val=val)
    trainer.train(resume_train=config['resume']['resume_training'])
