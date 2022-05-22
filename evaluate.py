import argparse
import json
import logging
import os
import random
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
from transformers import RobertaTokenizer
import utils
from dataloader import CustomDataLoader
# load args 加载参数

# load args
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2022, help="random seed for initialization")
parser.add_argument('--ex_index', type=str, default=7)
parser.add_argument('--corpus_type', type=str, default="WebNLG", help="NYT, WebNLG, NYT*, WebNLG*")
parser.add_argument('--mode', type=str, default="val")
parser.add_argument('--device_id', type=int, default=0, help="GPU index")
parser.add_argument('--restore_file', default='last', help="name of the file containing weights to reload")
parser.add_argument('--corres_threshold', type=float, default=0.5, help="threshold of global correspondence")
parser.add_argument('--rel_threshold', type=float, default=0.5, help="threshold of relation judgement")
parser.add_argument('--ensure_corres', action='store_true', help="correspondence ablation")
parser.add_argument('--ensure_rel', action='store_true', help="relation judgement ablation")
parser.add_argument('--emb_fusion', type=str, default="concat", help="way to embedding")

def get_metrics(correct_num, predict_num, gold_num):
    """
    计算精确度，放回p， recall， F1
    :param correct_num: 预测正确的
    :param predict_num: 预测的
    :param gold_num:  样本的
    :return:
    """
    p = correct_num / predict_num if predict_num > 0 else 0 #精确度
    r = correct_num / gold_num if gold_num > 0 else 0 #召回
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0 #F1
    return {
        'correct_num': correct_num,
        'predict_num': predict_num,
        'gold_num': gold_num,
        'precision': p,
        'recall': r,
        'f1': f1
    }


def _concat(token_list):
    """
    返回token对应的头实体或者尾实体
    """
    result = ''
    for idx, t in enumerate(token_list):
        if idx == 0:  # 第一个直接加
            result = t
        elif t.startswith('##'):  # 如果以开头
            result += t.lstrip('##')  # 去掉##加
        else:
            result += ' ' + t  # 表示第二个单词，加空格
    return result

def span2str(triple, tokens):
    output = []
    rel = triple[1]
    sub_tokens = tokens[triple[0][0] : triple[0][-1]+1] #要
    obj_tokens = tokens[triple[-1][0] : triple[-1][-1]+1]
    sub = _concat(sub_tokens)
    obj = _concat(obj_tokens)
    output.append((sub, obj, rel))
    return output
def extractspobymodel(input_token, input_id, attention_mask, model, params, ex_params, entity_start=0.5,entity_end=0.5,p_num=0.5):
    def get_entity(entity_pred, input_id):
        '''
        从预测得到的sub得到具体的sub
        :param entity_pred:
        :return: 具体的实体的头部和尾部位置（entitynum， 2）
        '''
        start = np.where(entity_pred[0, :, 0] > entity_start)[0] #实体头部标注位置
        end = np.where(entity_pred[0, :, 1] > entity_end)[0] #实体尾部标注位置
        entity = []
        for i in start: #按照最近匹配原则匹配出实体
            if input_id[i] == 0 :
                continue
            j = end[end >= i]
            if len(j) > 0:
                j = j[0]
                entity.append((i, j))
        return entity
    triples=[]
    model.eval()
    with torch.no_grad():
        head, tail, rel, cls = model.get_embed(input_id.unsqueeze(0).to("cuda"), attention_mask.unsqueeze(0).to("cuda"))
        head = head.cpu().detach().numpy()  # [1,L,H]
        tail = tail.cpu().detach().numpy()
        rel = rel.cpu().detach().numpy()
        cls = cls.cpu().detach().numpy()
        s_preds = model.s_pred(torch.tensor(head).to("cuda"), torch.tensor(cls).to("cuda"))
        s_preds = s_preds.cpu().detach().numpy()  # [1,L,2]
    s = get_entity(s_preds, input_id) #s （snum, 2）
    # print("s: ", s)
    s_loc = []
    o_loc = []
    so_mask=[]
    for i, sub in enumerate(s): #针对到一个句子的所有sub，来分别进行抽取obj s (,L,2)
        # s:(start,end)
        s_mask = np.zeros(input_id.shape[0]).astype(np.int64)
        s_mask[sub[0]: sub[1] + 1] = 1 #基s抽取o
        model.eval()
        with torch.no_grad():
            o_pred = model.o_pred_from_s(torch.tensor(np.array(s_mask)).to("cuda"), torch.tensor(head).to("cuda"),
                                         torch.tensor(tail).to("cuda"),torch.tensor(cls).to("cuda"))
            o_pred = o_pred.cpu().detach().numpy()  # [1,L,2]
        obj = get_entity(o_pred, input_id)
        # print("obj: ", obj)
        if obj:
            for o in obj:
                o_mask = np.zeros(input_id.shape[0]).astype(np.int64)
                o_mask[o[0]: o[1] + 1] = 1  # 基s抽取o
                so_mask.append([s_mask, o_mask])
                s_loc.append(sub)  # s[start, end]
                o_loc.append(o) # o[start, end]
    model.eval()
    with torch.no_grad():
        p_r = model.p_r_pred(torch.tensor(rel).to("cuda"), torch.tensor(cls).to("cuda"))
        p_r = p_r.cpu().detach().numpy()
        p_r_label = np.where(p_r>0.5, np.ones(p_r.shape), np.zeros(p_r.shape))
        # print("p_r: ", p_r_label)
    if s_loc and o_loc: # 一个句子中的sub, obj 的entity‘
        # 复制多份head
        head = np.repeat(head, len(s_loc), 0)
        tail = np.repeat(tail, len(s_loc), 0)
        p_r_label = np.repeat(p_r_label, len(s_loc), 0)
        # 传入subject，object 抽取predicate
        model.eval()
        with torch.no_grad():
            p_pred = model.r_pred_from_so(entiypair=torch.Tensor(np.array(so_mask)).to("cuda").long(),#entiypair=[torch.tensor(s_loc).to("cuda").long(),torch.tensor(o_loc).to("cuda").long()]
                                          p_r_label=torch.Tensor(p_r_label).to("cuda"),
                                          head=torch.tensor(head).to("cuda"),
                                          tail=torch.tensor(tail).to("cuda"),
                                          rel=torch.tensor(rel).to("cuda")
                                  ) #br
            p_pred = p_pred.cpu().detach().numpy()  # BR

        index, p_index = np.where(p_pred > p_num)
        for j, p in zip(index, p_index):
            subject = s_loc[j]
            object = o_loc[j]
            triple = (subject, p, object)
            triple1 = span2str(triple, input_token)
            triples.extend(triple1)
    return triples, p_r_label
def evaluate(model, dataloader, params, ex_params, mark='Val'):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.to("cuda")
    model.eval()
    rel_num = params.rel_num
    correct_num, predict_num, gold_num = 0, 0, 0 #初始化

    with tqdm(total=dataloader.__len__(), desc="eval", ncols=100) as t:
        for i, batch in enumerate(dataloader):
            # to device
            batch = tuple(t.to(params.device) if isinstance(t, torch.Tensor) else t for t in batch)
            batch_input_token, batch_input_ids, batch_attention_mask, batch_triples = batch

            for input_token, input_ids, attention_mask, triples in zip(batch_input_token, batch_input_ids,
                                                                       batch_attention_mask, batch_triples):
                triple = []
                for triple_one in triples:
                    triple.extend(span2str(triple_one, input_token))
                pretriples, p_r_label = extractspobymodel(input_token, input_ids, attention_mask, model, params,
                                               ex_params)
                # print("true: ", triple)
                # print("pri: ", pretriples)

                correct_num += len(set(pretriples) & set(triple))
                predict_num += len(set(pretriples)) #预测的
                gold_num += len(set(triple)) #真实
            t.set_postfix(predict_num="%.1lf" % (predict_num),
                          gold_num="%.1lf" % (gold_num),
                          correct_num="%.1lf" % (correct_num))
            t.update(1)
    metrics = get_metrics(correct_num, predict_num, gold_num) #包含传入三个参数，还包含精确度，召回，F1
    metrics_str = "; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics.items())
    print("- {} metrics:\n".format(mark) + metrics_str)
    return metrics

if __name__ == '__main__':
    args = parser.parse_args()
    params = utils.Params(ex_index=args.ex_index, corpus_type=args.corpus_type)
    ex_params = {
        'corres_threshold': args.corres_threshold,
        'rel_threshold': args.rel_threshold,
        'ensure_corres': args.ensure_corres,
        'ensure_rel': args.ensure_rel,
        'emb_fusion': args.emb_fusion
    }

    torch.cuda.set_device(args.device_id)
    print('current device:', torch.cuda.current_device())
    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params.seed = args.seed
    mode = args.mode
    # Set the logger
    utils.set_logger()

    # get dataloader
    dataloader = CustomDataLoader(params)

    # Define the model
    logging.info('Loading the model...')
    logging.info(f'Path: {os.path.join(params.model_dir, args.restore_file)}.pth.tar')
    # Reload weights from the saved file
    model, optimizer = utils.load_checkpoint(os.path.join(params.model_dir, args.restore_file + '.pth.tar'))
    model.to(params.device)
    logging.info('- done.')

    logging.info("Loading the dataset...")
    loader = dataloader.get_dataloader(data_sign=mode, ex_params=ex_params)
    logging.info('-done')

    logging.info("Starting prediction...")
    me = evaluate(model, loader, params, ex_params, mark=mode)
    correct_num, predict_num, gold_num, p, r, f1 = me
    print(f1)
    # with open(params.data_dir / f'{mode}_triples.json', 'r', encoding='utf-8') as f_src:
    #     src = json.load(f_src)
    #     df = pd.DataFrame(
    #         {
    #             'text': [sample['text'] for sample in src],
    #             'pre': predictions,
    #             'truth': ground_truths
    #         }
    #     )
    #     df.to_csv(params.ex_dir / f'{mode}_result.csv')
    # logging.info('-done')