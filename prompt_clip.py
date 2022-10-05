import torch
import torch.nn as nn
from torch.nn import Dropout
from torch.nn.modules.utils import _pair

from clip.model import Transformer, VisionTransformer, CLIP, convert_weights

import math
from functools import reduce
from operator import mul

class PromptTransformer(Transformer):
    def __init__(self, 
                width:int, 
                layers:int, 
                heads:int, 
                patch_size:int, 
                prompt_dim:int, 
                attn_mask:torch.Tensor=None, 
                deep:bool=True, 
                prompt_dropout:float=0.1, 
                prompt_num_tokens:int=10):
        super().__init__(width, layers, heads, attn_mask=attn_mask)

        self.patch_size = _pair(patch_size)

        self.deep = deep
        self.prompt_num_tokens = prompt_num_tokens
        self.prompt_dim = prompt_dim
        self.prompt_dropout_ = prompt_dropout
        self.prompt_proj = nn.Identity()
        self.prompt_dropout = nn.Identity()

    def init_prompt(self, device='cpu'):
        #init prompt project
        if self.prompt_dim > -1:
            self.prompt_proj = nn.Linear(self.prompt_dim, self.width).to(device)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out')
            self.prompt_proj.half()
        else:
            self.prompt_dim=self.width
        
        #init prompt embedding
        val = math.sqrt(6. / float(3 * reduce(mul, self.patch_size, 1) + self.prompt_dim))
        self.prompt_embeddings = nn.Parameter(torch.zeros(self.prompt_num_tokens, 1, self.prompt_dim, device=device).half())
        # xavier_uniform initialization
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        
        if self.deep:
            self.deep_prompt_embeddings = nn.Parameter(torch.zeros(self.prompt_num_tokens, self.layers - 1, self.prompt_dim, device=device).half())
            # xavier_uniform initialization
            nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
            self.deep_prompt_embeddings.data.half()
        
        self.prompt_dropout = Dropout(self.prompt_dropout_).to(device)
    
    def train(self, mode:bool=True):
        if mode:
            self.resblocks.eval()
            self.prompt_proj.train()
            self.prompt_dropout.train()
        else:
            for module in self.children():
                module.train(mode)

    def incorporate_prompt(self, x):
        #x (1 + squence_length, batch_size, embedding_dim)
        batch_size = x.shape[1]
        if self.prompt_embeddings is not None:
            x = torch.cat([x[:1, :, :], 
                           self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(-1, batch_size, -1)), 
                           x[1:, :, :]], dim=0).half()     #(1 + prompt_length + squence_length, batch_size, embedding_dim)
        return x

    def forward_deep(self, x):
        if self.deep_prompt_embeddings is None:
           return self.resblocks(x) 
        batch_size = x.shape[1]
        output = x

        for i, module in enumerate(self.resblocks):
            if 0 < i and i <= self.deep_prompt_embeddings.shape[1]:
                deep_prompt_embedding = self.prompt_dropout(
                                        self.prompt_proj(
                                        self.deep_prompt_embeddings[:, [i-1], :]).expand(-1, batch_size, -1))
                output = torch.cat([output[:1, :, :],
                                    deep_prompt_embedding, 
                                    output[(1+self.prompt_num_tokens):, :, :]], dim=0).half()
            output = module(output)
        
        return torch.cat([output[:1, :, :], output[(1+self.prompt_num_tokens):, :, :]], dim=0)

    def forward(self, x):
        #x (1 + squence_length, batch_size, embedding_dim)
        embedding_output = self.incorporate_prompt(x)

        if self.deep:
            output = self.forward_deep(embedding_output)
        else:
            output = self.resblocks(embedding_output)
            output = torch.cat([output[:1, :, :], output[(1+self.prompt_num_tokens):, :, :]], dim=0)
        
        return output

class PromptVisionTransformer(VisionTransformer):
    def __init__(self, 
                input_resolution:int, 
                patch_size:int, 
                width:int, layers:int, 
                heads:int, 
                output_dim:int, 
                prompt_dim:int, 
                deep:bool=True, 
                prompt_dropout:float=0.1, 
                prompt_num_tokens:int=10,):
        super().__init__(input_resolution, patch_size, width, layers, heads, output_dim)

        self.transformer = PromptTransformer(width, 
                                            layers, 
                                            heads, 
                                            patch_size, 
                                            prompt_dim,
                                            deep=deep, 
                                            prompt_dropout=prompt_dropout, 
                                            prompt_num_tokens=prompt_num_tokens)
    
    def train(self, mode: bool = True):
        for module in self.children():
            module.eval()
        self.transformer.train(mode)

    def init_prompt(self, device='cpu'):
        self.transformer.init_prompt(device=device)

class PromptCLIP(CLIP):
    def __init__(self, clip:CLIP, prompt_num_tokens:int=10, visual_prompt:str='none', text_prompt:str='none', classifier:str='none'):
        clip_state = clip.state_dict()

        embed_dim = clip_state["text_projection"].shape[1]
        vision_patch_size = clip_state["visual.conv1.weight"].shape[-1]
        vision_width = clip_state["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in clip_state.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_heads = vision_width // 64
        image_resolution = vision_patch_size * round((clip_state["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        context_length = clip_state["positional_embedding"].shape[0]
        vocab_size = clip_state["token_embedding.weight"].shape[0]
        transformer_width = clip_state["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state if k.startswith("transformer.resblocks")))

        super().__init__(embed_dim=embed_dim, 
                         image_resolution=image_resolution, 
                         vision_layers=vision_layers, 
                         vision_width=vision_width, 
                         vision_patch_size=vision_patch_size, 
                         context_length=context_length, 
                         vocab_size=vocab_size, 
                         transformer_width=transformer_width, 
                         transformer_heads=transformer_heads, 
                         transformer_layers=transformer_layers)
        convert_weights(self)
        self.load_state_dict(clip_state)
        assert visual_prompt in ['none', 'deep', 'shallow']
        assert text_prompt in ['none', 'deep', 'shallow']

        self.visual_prompt = visual_prompt
        self.text_prompt = text_prompt

        if self.visual_prompt != 'none':
            deep = self._is_deep(self.visual_prompt)
            prompt_vit = PromptVisionTransformer(input_resolution=image_resolution, 
                                                 patch_size=vision_patch_size, 
                                                 width=vision_width, 
                                                 layers=vision_layers, 
                                                 heads=vision_heads, 
                                                 output_dim=embed_dim, 
                                                 prompt_dim=embed_dim, 
                                                 deep=deep, 
                                                 prompt_num_tokens=prompt_num_tokens)
            self.visual = prompt_vit
            convert_weights(self.visual)
        self.visual.load_state_dict(clip.visual.state_dict())

        if self.text_prompt != 'none':
            deep = self._is_deep(self.text_prompt)
            transformer_width = clip_state["ln_final.weight"].shape[0]
            transformer_heads = transformer_width // 64
            transformer_layers = len(set(k.split(".")[2] for k in clip_state if k.startswith("transformer.resblocks")))
            self.context_length += prompt_num_tokens
            prompt_transformer = PromptTransformer(width=transformer_width, 
                                                   layers=transformer_layers, 
                                                   heads=transformer_heads, 
                                                   patch_size=vision_patch_size, 
                                                   prompt_dim=embed_dim, 
                                                   attn_mask=self.build_attention_mask(), 
                                                   deep=deep, 
                                                   prompt_num_tokens=prompt_num_tokens)
            self.transformer = prompt_transformer
            convert_weights(self.transformer)
        self.transformer.load_state_dict(clip.transformer.state_dict())

        self.classifier = nn.Identity()

    def init_prompt(self, device='cpu'):
        if self.visual_prompt != 'none':
            self.visual.init_prompt(device)
        
        if self.text_prompt != 'none':
            self.transformer.init_prompt(device)
    
    def forward(self, image, text, froze_visual:bool=True, froze_text:bool=True):
        with torch.set_grad_enabled(not froze_visual):
            image_features = self.encode_image(image)
            image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True) #(batch_size, dim)
            image_features_norm = image_features_norm.unsqueeze(-1)
        
        with torch.set_grad_enabled(not froze_text):
            text_features = self.encode_text(text)
            text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features_norm = text_features_norm.view(-1, 4, text_features_norm.shape[-1])
        
        with torch.set_grad_enabled(not froze_visual or not froze_text):
            similarity = (100.0 * text_features_norm @ image_features_norm).squeeze(-1).softmax(dim=-1)
        
        return similarity

    def train(self, visual_mode:bool=True, text_mode:bool=True, classifier_mode:bool=True):
        for module in self.children():
            module.eval()
        
        if self.visual_prompt != 'none':
            self.visual.train(visual_mode)
        if self.text_prompt != 'none':
            self.transformer.train(text_mode)
        self.classifier.train(classifier_mode)

    def _is_deep(self, prompt_kind:str):
        if prompt_kind == 'deep':
            return True
        elif prompt_kind == 'shallow':
            return False
