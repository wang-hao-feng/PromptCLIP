import sys
sys.path.append('..')

import numpy as np
import random
import wandb
import os

import torch
from torch.utils.data import DataLoader

#config
from parser import get_argparser
from config import INPUT_PROCESS
from config import OBJECT_PROCESS
from config import PromptConfig

#datasets
from dataset import VCRDataset

#clip
import clip

#vpt
from prompt_clip import PromptCLIP

from evaluator import BaselineEvaluater
from trainer import BaselineTrainer

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

parser = get_argparser()
args = parser.parse_args()

cfg = PromptConfig(train_length=args.train_length, 
                   visual_prompt=args.visual_prompt, 
                   text_prompt=args.text_prompt, 
                   classifier=args.classifier, 
                   batch_size=args.batch_size, 
                   learning_rate=args.learning_rate,
                   weight_decay=args.weight_decay, 
                   momenton=args.momenton,  
                   warmup=args.warmup, 
                   epoch=args.epoch, 
                   seed=args.seed)
setup_seed(cfg.seed)

#init clip
device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model, preprocess = clip.load('../ViT-B-16.pt', device)

#init prompt_vit
model = PromptCLIP(clip_model, visual_prompt=cfg.visual_prompt, text_prompt=cfg.text_prompt).to(device)
#model.init_prompt(device)

vcr_train = VCRDataset(split='train', 
                       task='qa', 
                       preprocess=preprocess, 
                       tokenize=clip.tokenize, 
                       obj_preprocess_func=OBJECT_PROCESS.object_with_number, 
                       input_preprocess_func=INPUT_PROCESS.QA_IQA)
vcr_val = VCRDataset(split='val', 
                     task='qa', 
                     preprocess=preprocess, 
                     tokenize=clip.tokenize, 
                     obj_preprocess_func=OBJECT_PROCESS.object_with_number, 
                     input_preprocess_func=INPUT_PROCESS.QA_IQA)

     
vcr_train.limit_len(cfg.train_length)

train_loader = DataLoader(dataset=vcr_train, 
                          batch_size=cfg.batch_size, 
                          shuffle=True, 
                          num_workers=4, 
                          collate_fn=vcr_train.collate_fn, 
                          pin_memory=True)
val_loader = DataLoader(dataset=vcr_val, 
                        batch_size=cfg.batch_size, 
                        shuffle=False, 
                        num_workers=4, 
                        collate_fn=vcr_val.collate_fn, 
                        pin_memory=True)

evaluator = BaselineEvaluater(model=model, 
                 dataloader=val_loader, 
                 device=device)
trainer = BaselineTrainer(cfg=cfg, 
                          model=model, 
                          evaluater=evaluator, 
                          device=device)

name = ''
name += f'vpt:{cfg.visual_prompt}_' if cfg.visual_prompt != 'none' else ''
name += f'tpt:{cfg.text_prompt}_' if cfg.text_prompt != 'none' else ''
name += f'lr:{cfg.lr}_wd:{cfg.wd}_{cfg.train_length}'

print('-' * 30)
print(f'lr:{cfg.lr}   wd:{cfg.wd}   dataset_length:{cfg.train_length}')
model.init_prompt(device)
wandb.init(project=f'prompt_clip', name=name)
wandb.config = {
     'learning_rate': cfg.lr, 
     'epochs': cfg.epoch, 
     'batch_size': cfg.batch_size
}
trainer.train_classifier(train_loader, 
                         result_path=os.path.join(args.output_dir, name+'.json'), 
                         model_path=name+'.pth', 
                         train_visual=(cfg.visual_prompt != 'none'), 
                         train_text=(cfg.text_prompt != 'none'))