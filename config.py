from numpy import argmax

class PromptConfig():
    def __init__(self, 
                 train_length:int=0, 
                 visual_prompt:str='none', 
                 text_prompt:str='none', 
                 classifier:str='none', 
                 optim:str='SGD', 
                 batch_size:int=128, 
                 learning_rate:float=5e-5, 
                 weight_decay:float=1e-3, 
                 momentum:float=0.9, 
                 betas:tuple=(0.9, 0.999), 
                 amsgrad:bool=False, 
                 warmup:int=10, 
                 epoch:int=75, 
                 seed:int=2022) -> None:
        self.train_length = train_length
        self.visual_prompt = visual_prompt
        self.text_prompt = text_prompt
        self.classifier = classifier
        self.optim = optim
        self.batch_size = batch_size
        self.lr = learning_rate
        self.wd = weight_decay
        self.momentum = momentum
        self.betas = betas
        self.amsgrad = amsgrad
        self.warmup = warmup
        self.epoch = epoch
        self.seed = seed

        #grid search
        self.train_lengths = [100, 300, 500, 1000, 3000, 5000, 10000, 0] 
        self.lrs = [1e-4, 5e-5, 1e-5]
        self.wds = [1e-3, 5e-4, 1e-4]
        self.seeds = [2022]

class DeepPromptConfig(PromptConfig):
    def __init__(self) -> None:
        super().__init__()
        self.deep = 'deep'

class ShallowPromptConfig(PromptConfig):
    def __init__(self) -> None:
        super().__init__()
        self.deep = False

class OBJECT_PROCESS:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def only_object(word_list, object_names, refer):
        for word_idx, word in enumerate(word_list):
            if isinstance(word, list):
                word_objnames = [object_names[obj_idx] for obj_idx in word]
                word_list[word_idx] = ' and '.join(word_objnames)
        return ' '.join(word_list)

    @staticmethod
    def only_number(word_list, object_names, refer):
        for word_idx, word in enumerate(word_list):
            if isinstance(word, list):
                word_objnames = [f' {obj_idx}' for obj_idx in word]
                word_list[word_idx] = ' and '.join(word_objnames)
        return ' '.join(word_list)

    @staticmethod
    def object_with_number(word_list, object_names, refer):
        assert type(word_list) == list
        for word_idx, word in enumerate(word_list):
            if isinstance(word, list):
                word_objnames = [object_names[obj_idx] + f' {obj_idx}' for obj_idx in word]
                word_list[word_idx] = ' and '.join(word_objnames)
        return ' '.join(word_list)
    
    @staticmethod
    def replace_by_refer(word_list, object_names, refer):
        if refer is not None:
            for word_idx, word in enumerate(word_list):
                if isinstance(word, list):
                    word_objnames = []
                    for obj_idx in word:
                        if str(obj_idx) in refer:
                            word_objnames.append(refer[str(obj_idx)])
                        else:
                            word_objnames.append(object_names[obj_idx])
                    # word_objnames = [refer[str(obj_idx)] for obj_idx in word]
                    word_list[word_idx] = ' and '.join(word_objnames)
            return ' '.join(word_list)
        else:
            return OBJECT_PROCESS.only_object(word_list, object_names, refer)   

class INPUT_PROCESS:
    def __init__(self) -> None:
        pass

    @staticmethod
    def QA_IQA(question, answers, answer_labels, rationale_answers, rationale_labels):
        return [question + ' ' + a for a in answers], answer_labels
    
    @staticmethod
    def QA_IA(question, answers, answer_labels, rationale_answers, rationale_labels):
        return answers, answer_labels
    
    @staticmethod
    def QA_IAQ(question, answers, answer_labels, rationale_answers, rationale_labels):
        return [a + ' ' + question for a in answers], answer_labels

    @staticmethod
    def QA_IMaskQA(question, answers, answer_labels, rationale_answers, rationale_labels):
        mask_question = ' '.join(['a' for _ in range(len(question.split(' ')))])
        return [mask_question + ' ' + a for a in answers], answer_labels

    @staticmethod
    def QA_IMaskAQ(question, answers, answer_labels, rationale_answers, rationale_labels):
        mask_question = ' '.join(['a' for _ in range(len(question.split(' ')))])
        return [a + ' ' + mask_question for a in answers], answer_labels

    @staticmethod
    def QA_APhotoOf_IQA(question, answers, answer_labels, rationale_answers, rationale_labels):
        return ['a photo of ' + question + ' ' + a for a in answers], answer_labels
    
    @staticmethod
    def QA_APhotoOf_IA(question, answers, answer_labels, rationale_answers, rationale_labels):
        return ['a photo of' + a for a in answers], answer_labels

    @staticmethod
    def QA_APhoto_IA(question, answers, answer_labels, rationale_answers, rationale_labels):
        return ['a photo' + a for a in answers], answer_labels

    @staticmethod
    def QA_APhotoOf_IAQ(question, answers, answer_labels, rationale_answers, rationale_labels):
        return ['a photo of ' + a + ' ' + question for a in answers], answer_labels

    @staticmethod
    def QA_APhotoOfA_IQA(question, answers, answer_labels, rationale_answers, rationale_labels):
        return ['a photo of a ' + question + ' ' + a for a in answers], answer_labels
    
    @staticmethod
    def QA_APhotoOfA_IA(question, answers, answer_labels, rationale_answers, rationale_labels):
        return ['a photo of a ' + a for a in answers], answer_labels

    @staticmethod
    def QA_APhotoOfA_IAQ(question, answers, answer_labels, rationale_answers, rationale_labels):
        return ['a photo of a ' + a + ' ' + question for a in answers], answer_labels
    
    @staticmethod
    def QA_QATemplate_IQA(question, answers, answer_labels, rationale_answers, rationale_labels):
        return ['question : ' + question + ' answer : ' + a for a in answers], answer_labels

    @staticmethod
    def QAR_IQA(question, answers, answer_labels, rationale_answers, rationale_labels):
        # 这里answer labels用的有问题
        question += (' ' + answers[argmax(answer_labels)])
        return [question + ' ' + a for a in rationale_answers], rationale_labels

    @staticmethod
    def QA_APhotoOfA_ICQA(question, answers, answer_labels, rationale_answers, rationale_labels, caption):
        return ['a photo of a ' + caption + question + ' ' + a for a in answers], answer_labels
    
    @staticmethod
    def QA_APhotoOfA_ICA(question, answers, answer_labels, rationale_answers, rationale_labels, caption):
        return ['a photo of a ' + caption + a for a in answers], answer_labels