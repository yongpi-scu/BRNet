import os
import nets
import time
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from utils import metrics
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import datasets.transforms as extend_transforms
from datasets.birads import BIRADS
from torch.utils.data import DataLoader


class Model(object):

    def __init__(self, config):
        self.config = config
        if config["train"]["mixup"]["version"] == "v1":
            import nets.loss.mixup_v1 as mixup
            global mixup
        elif config["train"]["mixup"]["version"] == "v2":
            import nets.loss.mixup_v2 as mixup
            global mixup
        else:
            raise NotImplementedError
        # create dataset
        self._create_dataset()
        # create net
        self._create_net()
        # logger and writer
        self._create_log()
        # create optimizer
        self._create_optimizer()
        # create criterion
        self._create_criterion()
        # load parameters
        if config['eval']['ckpt_path']!="None":
            self.load(config['eval']['ckpt_path'])

    def _create_net(self):
        network_params = self.config["network"]
        # loading network parameters
        self.device = torch.device(network_params['device'])
        self.epochs = self.config['optim']['num_epochs']
        model_name = self.config['network']['model_name']
        num_classes = self.config["data"]["num_classes"]
        color_channels = self.config["data"]["color_channels"]
        if "vgg" in model_name or "resnet" in model_name:
            self.net = nets.__dict__[model_name](
                pretrained=True, num_classes=num_classes, color_channels=color_channels)
        else:
            self.net = nets.__dict__[model_name](
                pretrained=True, num_classes=num_classes, color_channels=color_channels, drop_rate=self.config['network']['drop_prob'])

        if network_params['use_parallel']:
            self.net = nn.DataParallel(self.net)
        else:
            self.net = self.net.to(self.device)

    def _create_log(self):
        self.best_result = {"test": {"epoch": 0, "acc": 0},
                            "val": {"epoch": 0, "acc": 0},
                            "train": {"epoch": 0, "acc": 0}}
        config = self.config
        model_name = config['network']['model_name']
        model_suffix = config['network']['model_suffix']
        seed = "_"+str(config['network']['seed'])

        # loading logging parameters
        logging_params = config['logging']
        timestamp = time.strftime("_%Y-%m-%d_%H-%M-%S", time.localtime())
        self.ckpt_path = os.path.join(
            'results', model_name, model_suffix+seed+timestamp, 'ckpt')
        self.tb_path = os.path.join(
            'results', model_name, model_suffix+seed+timestamp, 'tensorboard')
        self.log_path = os.path.join(
            'results', model_name, model_suffix+seed+timestamp, 'log')

        if logging_params['use_logging']:
            from utils.log import get_logger
            from utils.parse import format_config

            if not os.path.exists(self.ckpt_path):
                os.makedirs(self.ckpt_path)
            self.logger = get_logger(self.log_path)
            # self.logger.info(">>>The net is:")
            # self.logger.info(self.net)
            self.logger.info(">>>The config is:")
            self.logger.info(format_config(config))
        if logging_params['use_tensorboard']:
            from tensorboardX import SummaryWriter

            if not os.path.exists(self.tb_path):
                os.makedirs(self.tb_path)
                self.writer = SummaryWriter(self.tb_path)

    def _create_dataset(self):
        def _init_fn(worker_id):
            """Workers init func for setting random seed."""
            np.random.seed(self.config['network']['seed'])
            random.seed(self.config['network']['seed'])

        data_params = self.config['data']
        # making training dataset and dataloader
        train_params = self.config['train']
        train_trans_seq = self._resolve_transforms(train_params['aug_trans'])
        train_dataset = BIRADS(root_dir=data_params['root_dir'],
                               pkl_file=data_params["pkl_file"],
                               mode="train",
                               transforms=train_trans_seq)
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=train_params['batch_size'],
                                       shuffle=True,
                                       num_workers=train_params['num_workers'],
                                       drop_last=True,
                                       pin_memory=train_params['pin_memory'],
                                       worker_init_fn=_init_fn)

        # making testing dataset and dataloader
        eval_params = self.config['eval']
        eval_trans_seq = self._resolve_transforms(eval_params['aug_trans'])
        eval_dataset = BIRADS(root_dir=data_params['root_dir'],
                              pkl_file=data_params["pkl_file"],
                              mode="test",
                              transforms=eval_trans_seq)
        self.eval_loader = DataLoader(eval_dataset,
                                      batch_size=eval_params['batch_size'],
                                      shuffle=False,
                                      num_workers=eval_params['num_workers'],
                                      pin_memory=eval_params['pin_memory'],
                                      worker_init_fn=_init_fn)
        # making validation dataset and dataloader
        val_dataset = BIRADS(root_dir=data_params['root_dir'],
                              pkl_file=data_params["pkl_file"],
                              mode="val",
                              transforms=eval_trans_seq)
        self.val_loader = DataLoader(val_dataset,
                                      batch_size=eval_params['batch_size'],
                                      shuffle=False,
                                      num_workers=eval_params['num_workers'],
                                      pin_memory=eval_params['pin_memory'],
                                      worker_init_fn=_init_fn)

    def _create_optimizer(self):
        optim_params = self.config['optim']
        if optim_params['optim_method'] == 'sgd':
            sgd_params = optim_params['sgd']
            optimizer = optim.SGD(self.net.parameters(),
                                  momentum=sgd_params['momentum'],
                                  weight_decay=sgd_params['weight_decay'],
                                  nesterov=sgd_params['nesterov'])
        elif optim_params['optim_method'] == 'adam':
            adam_params = optim_params['adam']
            optimizer = optim.Adam(self.net.parameters(),
                                   lr=adam_params['base_lr'],
                                   betas=adam_params['betas'],
                                   weight_decay=adam_params['weight_decay'],
                                   amsgrad=adam_params['amsgrad'])
        elif optim_params['optim_method'] == 'adadelta':
            adadelta_params = optim_params['adadelta']
            optimizer = optim.Adadelta(self.net.parameters(
            ), lr=adadelta_params['base_lr'], weight_decay=adadelta_params['weight_decay'],)

        # choosing whether to use lr_decay and related parameters
        if optim_params['use_lr_decay']:
            if optim_params['lr_decay_method'] == 'cosine':
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, eta_min=0, T_max=self.config['optim']['num_epochs'])
            if optim_params['lr_decay_method'] == 'lambda':
                def lr_lambda(epoch): return (1 - float(epoch) /
                                              self.config['optim']['num_epochs'])**0.9
                lr_scheduler = optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda)
            self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer

    def _create_criterion(self):
        # choosing criterion
        criterion_params = self.config['criterion']
        if criterion_params['criterion_method'] == 'cross_entropy':
            class_weights = torch.FloatTensor(criterion_params['class_weights']).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights).to(self.device)
        elif criterion_params['criterion_method'] == 'ordered_loss':
            from nets.loss.orderedloss import OrderedLoss
            criterion = OrderedLoss(alpha=criterion_params["ordered_loss"]["alpha"], beta=criterion_params["ordered_loss"]["beta"]).to(self.device)

        else:
            raise NotImplementedError
        self.criterion = criterion

    def run(self):
        for epoch_id in range(self.config['optim']['num_epochs']):
            self._train(epoch_id)
            if self.config['optim']['use_lr_decay']:
                self.lr_scheduler.step()
            # self._eval(epoch_id,"train")
            self._eval(epoch_id, "test")
            self._eval(epoch_id, "val")

    def _train(self, epoch_id):
        self.net.train()
        with tqdm(total=len(self.train_loader)) as pbar:
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)
                inputs, targets_a, targets_b, lam = mixup.mixup_data(
                    inputs, targets, self.config["train"]["mixup"]["alpha"])
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss_func = mixup.mixup_criterion(targets_a, targets_b, lam)
                loss = loss_func(self.criterion, outputs)
                loss.backward()
                self.optimizer.step()
                pbar.update(1)

    def _forward(self, data_loader):
        net = self.net
        net.eval()
        total_loss = 0
        total_output = []
        total_target = []
        num_steps = data_loader.__len__()
        with torch.no_grad():
            with tqdm(total=num_steps) as pbar:
                for data, target in data_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = net(data)
                    loss = self.criterion(output, target).mean()
                    output = F.softmax(net(data), dim=1)
                    # convert a 0-dim tensor to a Python number
                    total_loss += loss.item()
                    total_output.extend(output.data.cpu().numpy())
                    total_target.extend(target.data.cpu().numpy())
                    pbar.update(1)
        return total_output, total_target, total_loss

    def _eval(self, epoch, mode="test"):
        logger = self.logger
        if mode == "test":
            data_loader = self.eval_loader
        elif mode == "train":
            data_loader = self.train_loader
        elif mode == "val":
            data_loader = self.val_loader
        output, target, loss = self._forward(data_loader)
        confusion_matrix = metrics.get_confusion_matrix(output, target)
        num_correct = sum(np.argmax(output, 1) == target)
        acc = num_correct / len(target)
        loss = loss / len(target)
        logger.info("[{}] Epoch:{},confusion matrix:\n{}".format(
            mode, epoch, confusion_matrix))
        logger.info("[{0}] Epoch:{1},{0} acc:{2}/{3}={4:.5},{0} loss:{5:.5}".format(
            mode, epoch, num_correct, len(target), acc, loss))
        self.writer.add_scalar('%s_loss' % mode, loss, epoch)
        self.writer.add_scalar('%s_acc' % mode, acc, epoch)
        results = self.best_result[mode]
        if acc > results["acc"]:
            results["acc"] = acc
            results["epoch"] = epoch
            logger.info('[Info] Epochs:%d, %s accuracy improve to %g' %
                        (epoch, mode, acc))
            snapshot_name = '%s_epoch_%d_loss_%.5f_acc_%.5f_lr_%.10f' % (
                mode, epoch, loss, acc, self.optimizer.param_groups[0]['lr'])
            torch.save(self.net.state_dict(), os.path.join(
                self.ckpt_path, snapshot_name + '.pth'))
            # torch.save(self.optimizer.state_dict(), os.path.join(
            #     self.ckpt_path, 'opt_' + snapshot_name + '.pth'))
        else:
            logger.info("[Info] Epochs:%d, %s accuracy didn't improve,\
current best acc is %g, epoch:%g" % (epoch, mode, results["acc"], results["epoch"]))
        return output, target

    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location = torch.device(self.config['network']['device']))
        if self.config['network']['use_parallel']:
            self.net.module.load_state_dict(ckpt)
        else:
            self.net.load_state_dict(ckpt)
        print(">>> Loading model successfully from {}.".format(ckpt_path))

    def save(self, epoch):
        if self.config['network']['use_parallel']:
            state_dict = self.net.module.state_dict()
        else:
            state_dict = self.net.state_dict()
        torch.save(state_dict, os.path.join(
            self.ckpt_path, '{}.pth'.format(epoch)))

    def _resolve_transforms(self, aug_trans_params):
        """
            According to the given parameters, resolving transform methods
        :param aug_trans_params: the json of transform methods used
        :return: the list of augment transform methods
        """
        trans_seq = []
        for trans_name in aug_trans_params['trans_seq']:
            if trans_name == 'fixed_resize':
                resize_params = aug_trans_params['fixed_resize']
                trans_seq.append(transforms.Resize(resize_params['size']))
            elif trans_name == 'to_tensor':
                trans_seq.append(transforms.ToTensor())
            elif trans_name == 'random_horizontal_flip':
                flip_p = aug_trans_params['flip_prob']
                trans_seq.append(transforms.RandomHorizontalFlip(p=flip_p))
            elif trans_name == 'multi_input':
                trans_seq.append(extend_transforms.Multi_Input(
                    aug_trans_params['multi_input']["size"]))
        return trans_seq
