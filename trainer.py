from cmath import isnan
from prompt_clip import PromptCLIP

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import PromptConfig

from evaluator import Evaluator
from solver.optimizer import make_optimizer
from solver.lr_scheduler import WarmupCosineSchedule

import os
import wandb

class Trainer():
    def __init__(self, 
                cfg:PromptConfig, 
                model:PromptCLIP, 
                evaluater:Evaluator, 
                lr:float, 
                weight_decay:float, 
                device:str='cpu') -> None:
        self.cfg = cfg
        self.model = model
        self.evaluater = evaluater
        self.device = device
        self.base_lr = lr
        self.base_wd = weight_decay

        self.optimizer = make_optimizer(self.model, lr, weight_decay, cfg.momenton)
        self.lr_scheduler = WarmupCosineSchedule(optimizer=self.optimizer, warmup_steps=cfg.warmup_epoch, t_total=cfg.total_epoch)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward_one_batch(self, 
                        image_inputs:torch.Tensor, 
                        targets:torch.Tensor, 
                        is_train:bool):
        image_inputs = image_inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        #forward
        with torch.set_grad_enabled(is_train):
            image_features = self.model.encode_image(image_inputs)
            image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True) #(batch_size, dim)
            similarity = (100.0 * image_features_norm @ self.evaluater.text_features.T)
            softmax_similarity = similarity.softmax(dim=-1)
            loss = self.loss_fn(softmax_similarity, targets)

        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss, softmax_similarity
    
    def train_classifier(self, train_loader):
        # save the model prompt if required before training
        #self.model.visual.eval()
        #self.save_prompt(0, '.')

        self.evaluater.evaluate('./clip_prompt_{0}_result_lr{1}_wd{2}.json'.format('deep' if self.model.visual.transformer.deep else 'shallow', 
                                                                                    self.base_lr, 
                                                                                    self.base_wd))

        total_epoch = self.cfg.total_epoch
        best_epoch = -1
        best_metric = 0

        for epoch in range(total_epoch):
            total_loss = []
            self.model.visual.train()
            for idx, (images, targets) in enumerate(train_loader):
                #Debug
                #if idx == 5:
                #    return

                train_loss, _ = self.forward_one_batch(images, targets, True)
                total_loss.append(train_loss)
            print((
                "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                + "average train loss: {:.4f}".format(torch.mean(torch.Tensor(total_loss)))))
            self.lr_scheduler.step()

            #eval
            self.model.visual.eval()

            cur_acc, _ = self.evaluater.evaluate('./clip_prompt_{0}_result_lr{1}_wd{2}.json'.format('deep' if self.model.visual.transformer.deep else 'shallow', 
                                                                                                    self.base_lr, 
                                                                                                    self.base_wd))
            if cur_acc > best_metric:
                best_metric = cur_acc
                best_epoch = epoch + 1
                self.save_prompt(epoch + 1, '.', 'vpt_prompt.pth')
            wandb.log({'loss': torch.mean(torch.Tensor(total_loss)), 
                       'acc': cur_acc, 
                       'lr': self.lr_scheduler.get_lr()[0]})
        print(f'Best_epoch:{best_epoch} Best_metric:{best_metric}')

    @torch.no_grad()
    def save_prompt(self, epoch:int, root:str, path:str=None):
        prompt_embds = self.model.visual.transformer.prompt_embeddings.cpu().numpy()
        prompt_proj = self.model.visual.transformer.prompt_proj.weight.cpu().numpy()
        out = {'shallow_prompt': prompt_embds, 
               'prompt_proj': prompt_proj}
        if self.model.visual.transformer.deep:
            deep_prompt_embeddings = self.model.visual.transformer.deep_prompt_embeddings.cpu().numpy()
            out['deep_shallow'] = deep_prompt_embeddings
        torch.save(out, os.path.join(root, f'prompt_ep{epoch}.pth' if path is None else path))

class BaselineTrainer():
    def __init__(self, cfg: PromptConfig, model: PromptCLIP, evaluater: Evaluator, device: str = 'cpu') -> None:
        self.cfg = cfg
        self.model:PromptCLIP = model
        self.evaluater = evaluater
        self.device = device

        self.optimizer = make_optimizer(model=self.model, 
                                        lr=cfg.lr, 
                                        weight_decky=cfg.wd, 
                                        optim=cfg.optim, 
                                        momentum=cfg.momentum, 
                                        betas=cfg.betas, 
                                        amsgrad=cfg.amsgrad)
        self.lr_scheduler = WarmupCosineSchedule(optimizer=self.optimizer, warmup_steps=cfg.warmup, t_total=cfg.epoch)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward_one_batch(self, data, froze_visual:bool=False, froze_text:bool=False):
        image_inputs:torch.Tensor = data.image_vec.to(self.device)
        targets = data.labels_per_image.to(self.device)
        text_inputs = data.description_vec.to(self.device)

        is_train = not froze_visual or not froze_text

        #forward
        with torch.set_grad_enabled(is_train):
            similarity = self.model(image_inputs, text_inputs, froze_visual=froze_visual, froze_text=froze_text)
            loss = self.loss_fn(similarity, targets)

            indices = torch.argmax(similarity, dim=1)
            acc = torch.sum(targets == indices)

        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss, similarity, acc
    
    def train_classifier(self, 
                         train_loader:DataLoader, 
                         model_path='./model.pth', 
                         result_path='./result.json', 
                         train_visual:bool=True, 
                         train_text:bool=True, 
                         train_classifier:bool=True):
        wandb.watch(self.model.visual.transformer, criterion=self.loss_fn, log='all', log_freq=1)

        self.evaluater.evaluate(result_path)

        total_epoch = self.cfg.epoch
        best_epoch = -1
        best_metric = 0

        for epoch in range(total_epoch):
            train_acc = 0
            total_loss = []
            self.model.train(visual_mode=train_visual, text_mode=train_text, classifier_mode=train_classifier)
            for idx, data in enumerate(train_loader):
                train_loss, _, acc = self.forward_one_batch(data, froze_visual=(not train_visual), froze_text=(not train_text))
                train_acc += acc
                total_loss.append(train_loss)
            train_acc = train_acc / len(train_loader.dataset)
            print((
                "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                + "average train loss: {:.4f}".format(torch.mean(torch.Tensor(total_loss)))
                + 'total train acc: {:.4f}'.format(train_acc)))
            self.lr_scheduler.step()

            #eval
            self.model.eval()

            cur_acc, val_loss, _ = self.evaluater.evaluate(result_path)
            if cur_acc > best_metric:
                best_metric = cur_acc
                best_epoch = epoch + 1
                torch.save(self.model.state_dict(), f=model_path)
            wandb.log({'train_loss': torch.mean(torch.Tensor(total_loss)), 
                       'train_acc': train_acc, 
                       'val_loss': val_loss, 
                       'val_acc': cur_acc, 
                       'lr': self.lr_scheduler.get_lr()[0]})
        print(f'Best_epoch:{best_epoch} Best_metric:{best_metric}')

