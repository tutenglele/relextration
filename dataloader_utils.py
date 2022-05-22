# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import random
from multiprocessing import Pool
import functools
import numpy as np
from collections import defaultdict
from itertools import chain

from typing import Dict, List, Optional, Tuple


class InputExample(object):
    """a single set of samples of data
        包含文本，实体对，关系列表，实体对对应关系
    """
    def __init__(self, text, en_pair_list, re_list, s2orel):
        self.text = text
        self.en_pair_list = en_pair_list #实体对
        self.re_list = re_list
        self.s2orel = s2orel


class InputFeatures(object):
    """
    Desc:
        a single set of features of data
        一组数据特征 包含
        输入token，ids，mask seqtag corres_tag relation triples, rel_tag
    """
    def __init__(self,
                 input_tokens,
                 input_ids,
                 attention_mask,
                 triples=None,
                 sub_labels=None,
                 s_mask=None,
                 obj_labels=None,
                 p_r = None,
                 r = None,
                 so_mask = None
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.triples = triples
        self.sub_labels = sub_labels
        self.obj_labels = obj_labels
        self.s_mask = s_mask
        self.p_r = p_r
        self.r = r
        self.so_mask = so_mask


def read_examples(data_dir, data_sign, rel2idx):
    """load data to InputExamples
        加载输入到INputExamples 传入参数为路径，和文件名
        返回INputexample，包含文本，实体对列，关系列，关系对应实体对的字典
    """
    examples = []
    # read src data
    with open(data_dir / f'{data_sign}.json', "r", encoding='utf-8') as f:
        data = json.load(f)
        for sample in data:
            text = sample['text'] #句子
            en_pair_list = [] #实体对
            re_list = [] #关系
            s2orel = defaultdict(list)
            for triple in sample['triple_list']:
                en_pair_list.append([triple[0], triple[-1]]) #添加实体对
                re_list.append(rel2idx[triple[1]]) #添加关系
                s2orel[triple[0]].append((triple[-1], rel2idx[triple[1]]))
            example = InputExample(text=text, en_pair_list=en_pair_list, re_list=re_list, s2orel=s2orel)
            examples.append(example)
    print("InputExamples:", len(examples))
    return examples

def convert(example, max_text_len, tokenizer, rel2idx, data_sign, ex_params):
    """convert function
    """
    text_tokens = tokenizer.tokenize(example.text) #把一句话进行分词变成一个一个字符
    # cut off 截断句子
    if len(text_tokens) > max_text_len:
        text_tokens = text_tokens[:max_text_len]

    # token to id
    input_ids = tokenizer.convert_tokens_to_ids(text_tokens) #把tokens映射为数字id
    attention_mask = [1] * len(input_ids)
    # zero-padding up to the sequence length
    if len(input_ids) < max_text_len: #填充
        pad_len = max_text_len - len(input_ids)
        # token_pad_id=0
        input_ids += [0] * pad_len
        attention_mask += [0] * pad_len

    # train data
    if data_sign == 'train':
        sub_feats = []
        s2op_map = {}
        p_r = np.zeros(len(rel2idx))
        for en_pair, rel in zip(example.en_pair_list, example.re_list):
            sub_token = tokenizer.tokenize(en_pair[0])
            obj_token = tokenizer.tokenize(en_pair[1])
            # get sub and obj head
            sub_head_id = find_head_idx(text_tokens, sub_token)
            obj_head_id = find_head_idx(text_tokens, obj_token)
            p_r[rel] = 1
            if sub_head_id != -1 and obj_head_id != -1: #关系存在
                sub = (sub_head_id, sub_head_id + len(sub_token) - 1)
                objrel = (obj_head_id, obj_head_id + len(obj_token) - 1, rel)
                if sub not in s2op_map: #key ((start,end), )
                    s2op_map[sub] = []
                s2op_map[sub].append(objrel) # 建立了主体匹配到客体的map匹配
        if s2op_map:
            sub_labels = np.zeros((len(input_ids), 2)) #seqlen,2
            for s in s2op_map: #s(start, end)
                sub_labels[s[0], 0] = 1
                sub_labels[s[1], 1] = 1
            # s_mask 和 object_labels :o_pred阶段使用
            start, end = np.array(list(s2op_map.keys())).T
            for s_start in start:
                e_end = np.random.choice(end[end >= s_start])
                s2o_loc = (s_start, e_end)
                s_mask = np.zeros(len(input_ids))
                s_mask[s_start: e_end+1] = 1
                obj_labels = np.zeros((len(input_ids), 2))

                if s2o_loc in s2op_map: #如果随机选择主语存在
                    for item in s2op_map[s2o_loc]:
                        o1, o2, _ = item
                        obj_labels[o1, 0] = 1
                        obj_labels[o2, 1] = 1
                        # print("obj_labels : ", obj_labels)
                #改为随机抽取 o1, o2
                sub_id = random.choice(list(s2op_map.keys()))
                os, oe, _ = random.choice(s2op_map[sub_id])
                obj_id = (os, oe)
                r = np.zeros(len(rel2idx))
                if sub_id in s2op_map:
                    for o1_, o2_, the_r in s2op_map[sub_id]:
                        if o1_ == obj_id[0] and o2_ == obj_id[1]:
                            r[the_r] = 1
                so_mask = np.zeros((len(input_ids), 2)) #分别存储sub, obj
                so_mask[sub_id[0]:sub_id[1] + 1, 0] = 1
                so_mask[obj_id[0]:obj_id[1] + 1, 1] = 1
                sub_feats.append(InputFeatures(
                    input_tokens=text_tokens,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    sub_labels=sub_labels,  # (seqlen, 2)
                    s_mask=s_mask,
                    obj_labels=obj_labels,  # (seqlen, 2)
                    p_r=p_r,
                    r=r,
                    so_mask=so_mask.T
                ))
                # 两个独立的任务 ->obj  ->r 采用负样本的话，先用set集合得到sub集合， obj集合
                #正样本，1，sub -> obj 2，sub,obj->rel
                #负样本 1 no-sub -> 0 2.no_sub,obj->0 3,no_sub,no_obj ->0, 4  sub,no_obj->0
                #负样本集合 negset

    # val and test data
    else:
        triples = []
        for rel, en in zip(example.re_list, example.en_pair_list):
            triple = (tokenizer.tokenize(en[0]), rel, tokenizer.tokenize(en[1]))
            sub_head_idx = find_head_idx(text_tokens, triple[0])
            obj_head_idx = find_head_idx(text_tokens, triple[2])
            if sub_head_idx != -1 and obj_head_idx != -1:
                sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1) #存了sub的头和尾位置
                obj = (obj_head_idx, obj_head_idx + len(triple[2]) - 1)
                triples.append((sub, rel, obj))
        sub_feats = [
            InputFeatures(
                input_tokens=text_tokens,
                input_ids=input_ids,
                attention_mask=attention_mask,
                triples=triples
            )
        ]

    # get sub-feats
    return sub_feats

def find_head_idx(source: List[int], target: List[int]) -> int:
    '''
    从句子中source找到实体target
    :param source:
    :param target:
    :return:
    '''
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1
def convert_examples_to_features(params, examples, tokenizer, rel2idx, data_sign, ex_params):
    """convert examples to features.
    :param examples (List[InputExamples])
    features：输入token，ids，mask seqtag corres_tag relation triples, rel_tag
    """
    max_text_len = params.max_seq_length
    # multi-process
    with Pool(10) as p: # 建立进程池p，其大小为20
        #使用 functools.partial 来包装一个函数，其中参数example，由线程来提供
        convert_func = functools.partial(convert, max_text_len=max_text_len, tokenizer=tokenizer, rel2idx=rel2idx,
                                         data_sign=data_sign, ex_params=ex_params)
        features = p.map(func=convert_func, iterable=examples)
        #将f函数投入进程池p中进行执行，每个进程取一个in_argv作为shell的参数，然后开始执行。所有的f返回值都会被返回到一个list中

    return list(chain(*features))
