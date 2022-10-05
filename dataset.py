import enum
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter, namedtuple, defaultdict

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 933120000
import random

from torchvision.datasets.folder import DatasetFolder

instance_fields = [
    'annot_id',
    'image_vec',
    'description_vec',
    'labels_per_image',
]
Instance = namedtuple('Instance', field_names=instance_fields,
                      defaults=[None] * len(instance_fields))

batch_fields = [
    'annot_id',
    'image_vec',
    'description_vec',
    'labels_per_image',
    'descriptions'
]
Batch = namedtuple('Batch', field_names=batch_fields,
                   defaults=[None] * len(batch_fields))


class VCRDataset(Dataset):
    def __init__(self, split, task, image_dir='/users7/pyhou/dataset/vcr_images/vcr1images',
                    preprocess=None, tokenize=None, obj_preprocess_func=None, input_preprocess_func=None, refer_file=None, freq_file=None):
        """
        :param path (str): path to the data file.
        :param gpu (bool): use GPU (default=False).
        """
        assert split in ['train', 'val', 'test']
        assert task in ['qa', 'qar']
        
        self.qa_json = f'/users7/pyhou/dataset/vcr_annotation/{split}.jsonl'
        
        self.image_dir = image_dir
        self.task = task
        self.data = []

        self.preprocess = preprocess
        self.tokenize = tokenize
        
        if refer_file:
            with open(refer_file, 'r') as f:
                self.refer_info = json.load(f)
        else:
            self.refer_info = None
        
        if freq_file:
            with open(freq_file, 'r') as f:
                self.freq_info = [l.strip() for l in f.readlines()]
            self.freq_num = 0
        else:
            self.freq_info = None
        
        self.obj_preposs_func = obj_preprocess_func
        self.input_preprocess_func = input_preprocess_func

        # self.vocabs, self.templates = self.load_dict_evt()
        self.load_data()

    def limit_len(self, length):
        if length != 0:
            self.data = self.data[:length]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # image reading
        return self.data[item]

    def shorten_context(self, text):
        text = text.replace('FILE - ', '')
        text = text[:350] # not exceed 350 characters
        return text


    def load_data(self):
        """Load data from file."""

        # load qa pairs
        for line in open(self.qa_json):
            data = json.loads(line)
            annot_id = data['annot_id']
            movie = data['movie']
            objects = data['objects']
            # image path
            img_fn = data['img_fn']
            # bounding boxes
            metadata_fn = data['metadata_fn']
            question = data ['question']
            answer_choices = data['answer_choices']
            answer_label = data['answer_label']
            rationale_choices = data['rationale_choices']
            rationale_label = data['rationale_label']

            inst = dict()
            inst['annot_id'] = annot_id
            # inst['movie'] = movie
            inst['img_fn'] = img_fn
            # inst['answer'] = list()
            # inst['retionale'] = list()
            inst['descriptions'] = list()


            if self.task == 'qar':
                raise NotImplementedError()
                for retionale in rationale_choices:
                    retionale_str = self.fill_name(retionale, objects)
                    inst['descriptions'].append(retionale_str)
                    inst['label'] = rationale_label
            else:
                if self.refer_info and img_fn in self.refer_info:
                    refer = self.refer_info[img_fn]
                else:
                    refer = None
                inst['question'] = self.obj_preposs_func(question, objects, refer)
                if self.freq_info:
                    if inst['question'] in self.freq_info:
                        self.freq_num += 1
                        print(f'{self.freq_num} IN FREQ!!!!!!{inst["question"]}')
                    else:
                        inst["question"] = ''
                for answer in answer_choices:
                    # try:
                    answer_str = self.obj_preposs_func(answer, objects, refer)
                    # except:
                    #     print(inst['annot_id'])
                    inst['descriptions'].append(answer_str)
                    inst['label'] = answer_label
                inst['descriptions'], _ = self.input_preprocess_func(inst['question'], inst['descriptions'], [inst['label']], None, None)
            self.data.append(inst)

            # break
        
        print('Loaded {} instances from {}'.format(len(self), self.qa_json))

    # def fill_name(self, word_list, object_names, refer):
    #     for word_idx, word in enumerate(word_list):
    #         if isinstance(word, list):
    #             word_objnames = [object_names[obj_idx] + f' {obj_idx}' for obj_idx in word]
    #             word_list[word_idx] = ' and '.join(word_objnames)
    #     return ' '.join(word_list)


    def clean_imageid(self, image_id):
        return image_id.replace('.', '_')

    def collate_fn(self, batch): #, preprocess, tokenize):
        # print('batch', batch[0]['image_id'])    

        # image_ids = [self.clean_imageid(inst['image_id']) for inst in batch]
        annot_ids = list()
        # movies = list()
        images = list()
        image_vecs = list()
        descriptions = list()
        description_vecs = list()
        
        for inst in batch:
            annot_ids.append(inst['annot_id'])
            # movies.append(inst['movie'])
            images.append(inst['img_fn'])
            description_vecs.append(self.tokenize(inst['descriptions'], truncate=True))

            image_path = os.path.join(self.image_dir, inst['img_fn'])
            image_vec = self.preprocess(Image.open(image_path))
            image_vecs.append(image_vec)
            descriptions.append(inst['descriptions'])

        image_vecs = torch.stack(image_vecs, dim=0)
        
        description_vecs = torch.stack(description_vecs, dim=0)
        description_vecs = description_vecs.view(len(batch)*4, -1)

        labels_per_image_keepshape = [inst['label'] for inst in batch]
        labels_per_image_keepshape = torch.LongTensor(labels_per_image_keepshape)

        return Batch(
            annot_id=annot_ids,
            image_vec=image_vecs,
            description_vec=description_vecs,
            labels_per_image=labels_per_image_keepshape,
            descriptions=descriptions,
        )

