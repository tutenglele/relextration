import os
import torch
import os
import json
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from transformers import BertTokenizer
# from transformers import RobertaTokenizer
from dataloader_utils import read_examples, convert_examples_to_features

class FeatureDataset(Dataset):
    """Pytorch Dataset for InputFeatures
        数据集特征，该类继承torch.dataset
    """
    def __init__(self, features):
        self.features = features

    def __len__(self) -> int: #返回数据集大小
        return len(self.features)

    def __getitem__(self, index): #支持从 0 到 len(self)的索引
        return self.features[index]

class CustomDataLoader(object):
    def __init__(self, params):
        self.params = params
        self.train_batch_size = params.train_batch_size
        self.val_batch_size = params.val_batch_size
        self.test_batch_size = params.test_batch_size
        self.data_dir = params.data_dir
        self.max_seq_length = params.max_seq_length #最大句子长度
        self.tokenizer = BertTokenizer(vocab_file=os.path.join(params.bert_model_dir, 'vocab.txt'),
                                       do_lower_case=False)
        # self.tokenizer = RobertaTokenizer.from_pretrained(os.path.join(params.bert_model_dir / 'roberta'))
        self.data_cache = params.data_cache

    @staticmethod
    def collate_fn_train(features):
        """将InputFeatures转换为Tensor
        Args:
            features (List[InputFeatures]) train的feature(inputid, attmask, seqtags, p_rel, corr, rel_tags)
        Returns:
            tensors (List[Tensors])
        """
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        sub_labels = torch.tensor([f.sub_labels for f in features], dtype=torch.long)
        s2o_loc = torch.tensor([f.s2o_loc for f in features], dtype=torch.long)
        obj_labels = torch.tensor([f.obj_labels for f in features], dtype=torch.long)
        sub_id = torch.tensor([f.sub_id for f in features], dtype=torch.long)
        obj_id = torch.tensor([f.obj_id for f in features], dtype=torch.long)
        s_mask = torch.tensor([f.s_mask for f in features], dtype=torch.long)
        p_r = torch.tensor([f.p_r for f in features], dtype=torch.long)
        rel = torch.tensor([f.r for f in features], dtype=torch.long)
        so_mask = torch.tensor([f.so_mask for f in features], dtype=torch.long)
        tensors = [input_ids, attention_mask, sub_labels, s_mask, s2o_loc, obj_labels, sub_id, obj_id, p_r, rel, so_mask]
        return tensors

    @staticmethod
    def collate_fn_test(features):
        """将InputFeatures转换为Tensor
        Args:
            features (List[InputFeatures]) test的feature(inputid,attmask,triples,inputtoken)
        Returns:
            tensors (List[Tensors])
        """
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        triples = [f.triples for f in features]
        input_tokens = [f.input_tokens for f in features]
        tensors = [input_tokens, input_ids, attention_mask, triples]
        return tensors

    def get_features(self, data_sign, ex_params):
        """convert InputExamples to InputFeatures
        :param data_sign: 'train', 'val' or 'test'
        """
        print("=*=" * 10)
        print("Loading {} data...".format(data_sign))
        # get features
        # os.path.join连接路径
        cache_path = os.path.join(self.data_dir, "{}.cache.{}".format(data_sign, str(self.max_seq_length)))
        if os.path.exists(cache_path) and self.data_cache: #如果之前已经有缓存就不用特征了
            features = torch.load(cache_path)
        else:
            # get relation to idx
            with open(self.data_dir / f'rel2id.json', 'r', encoding='utf-8') as f_re:
                rel2idx = json.load(f_re)[-1] #获取关系id对应表
            # get examples
            if data_sign in ("train", "val", "test", "pseudo", 'EPO', 'SEO', 'SOO', 'Normal', '1', '2', '3', '4', '5'):
                examples = read_examples(self.data_dir, data_sign=data_sign, rel2idx=rel2idx)
            else:
                raise ValueError("please notice that the data can only be train/val/test!!")
            features = convert_examples_to_features(self.params, examples, self.tokenizer, rel2idx, data_sign,
                                                    ex_params)
            # save data
            if self.data_cache:
                torch.save(features, cache_path)
        return features

    def get_dataloader(self, data_sign="train", ex_params=None):
        """construct dataloader
        :param data_sign: 'train', 'val' or 'test'
        """
        # InputExamples to InputFeatures
        features = self.get_features(data_sign=data_sign, ex_params=ex_params)
        dataset = FeatureDataset(features)
        print(f"{len(features)} {data_sign} data loaded!")
        print("=*=" * 10)
        # construct dataloader
        # RandomSampler(dataset) or SequentialSampler(dataset)
        if data_sign == "train":
            datasampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.train_batch_size,
                                    collate_fn=self.collate_fn_train)
            #collate_fn将一个list的sample组成一个mini-batch的函数
            # Dataloader的处理逻辑是先通过Dataset类里面的
            # __getitem__
            # 函数获取单个的数据，然后组合成batch，再使用collate_fn所指定的函数对这个batch做一些操作，比如padding啊之类的。
        elif data_sign == "val":
            #按顺序对数据集采样
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.val_batch_size,
                                    collate_fn=self.collate_fn_test)
        elif data_sign in ("test", "pseudo", 'EPO', 'SEO', 'SOO', 'Normal', '1', '2', '3', '4', '5'):
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.test_batch_size,
                                    collate_fn=self.collate_fn_test)
        else:
            raise ValueError("please notice that the data can only be train/val/test !!")
        return dataloader


if __name__ == '__main__':
    from utils import Params
    params = Params(corpus_type='NYT')
    ex_params = {
       'ensure_rel': True
    }
    dataloader = CustomDataLoader(params)
    feats = dataloader.get_features(ex_params=ex_params, data_sign='train')

    # val_data = dataloader.get_dataloader("train", ex_params)

    # print(len(val_data))
    print(len(feats))
    # for i, batch_val_data in enumerate(val_data):
    #     batch_input_token, batch_input_ids, batch_attention_mask, batch_triples = batch_val_data
    #     if i==0 :
    #         for token, ids, mask, triple in zip(batch_input_token, batch_input_ids, batch_attention_mask, batch_triples):
    #             print(token)
    #             print(mask)
    #             print(triple)
    #             t = ids.unsqueeze(0)
    #             print(t.shape)
    #             print(ids.shape)
    #             print(ids)
    #             print(t)
    #             print("*****************************")
    print(feats[0].input_tokens)
    print(feats[0].input_ids)
    print(feats[0].triples)
    print(feats[0].attention_mask)
    print(feats[0].sub_labels)
    print(feats[0].s_mask)
    print(feats[0].s2o_loc)
    print(feats[0].obj_labels)
    print(feats[0].sub_id)
    print(feats[0].obj_id)
    print(feats[9].r)
    print(feats[9].p_r)
    print("p_r.size  ", feats[0].p_r.size)
    print(feats[0].so_mask)

