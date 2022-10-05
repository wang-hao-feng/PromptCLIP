import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from prompt_clip import PromptCLIP

import os
import json

class Evaluator():
    def __init__(self, model:nn.Module, dataloader:DataLoader, text_features:torch.Tensor, device:str='cpu') -> None:
        self.model = model
        self.dataloader = dataloader
        self.text_features = text_features
        self.device = device
        self.iter = 0
    
    def evaluate(self, result_path='./result.json'):
        if(os.path.exists(result_path) and self.iter == 0):
            os.remove(result_path)
        results = []
        acc = 0
        for idx, (image, class_id) in enumerate(self.dataloader):
            #print('{:.1f}%'.format(idx / len(self.dataloader) * 100))
            image_input = image.to(self.device)
            class_id = class_id.to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_input)

                image_features /= image_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
                indices = torch.argmax(similarity, dim=1)
                for i in range(class_id.shape[0]):
                    results.append(json.dumps({'id':idx * self.dataloader.batch_size + i, 
                                                'label':class_id[i].item(), 
                                                'predict':indices[i].item(), 
                                                'correct':(class_id[i] == indices[i]).item()}) + '\n')
                acc += torch.sum(class_id == indices)
        acc = acc / len(self.dataloader.dataset)
        with open(result_path, 'a') as file:
            file.write('iter:{0} acc:{1}\n'.format(self.iter, acc))
        self.iter += 1
        
        return acc, results

class BaselineEvaluater:
    def __init__(self, model:PromptCLIP, dataloader:DataLoader, device:str='cpu') -> None:
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.iter = 0
    
    def evaluate(self, result_path='result.json'):
        if(os.path.exists(result_path) and self.iter == 0):
            os.remove(result_path)
        results = []
        acc = 0
        total_val_loss = []
        for idx, data in enumerate(self.dataloader):
            image_input = data.image_vec.to(self.device)
            class_id = data.labels_per_image.to(self.device)
            text_input = data.description_vec.to(self.device)

            with torch.no_grad():
                similarity = self.model(image_input, text_input)
                indices = torch.argmax(similarity, dim=1)
                acc += torch.sum(class_id == indices)
                loss = F.cross_entropy(similarity, class_id)
                total_val_loss.append(loss)
            
        val_acc = acc / len(self.dataloader.dataset)
        total_val_loss = torch.mean(torch.Tensor(total_val_loss))
        with open(result_path, 'a') as file:
            file.write('iter:{0} acc:{1}\n'.format(self.iter, val_acc))
        self.iter += 1

        return val_acc, total_val_loss, results