# /usr/bin/env python
# coding=utf-8
"""model"""
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from transformers import BertPreTrainedModel, BertModel
from opt_einsum import contract
from math import sqrt

class Attention_net(nn.Module):
    def __init__(self, relinput, subinput, output):
        super(Attention_net, self).__init__()
        self.relinput = relinput
        self.subinput = subinput
        self.output = output
        self.Wr = nn.Linear(self.relinput, self.output)
        self.Ws = nn.Linear(self.subinput, self.output)
        self.Wv = nn.Linear(self.output, self.output)
    def forward(self, rel, sub):
        rel_ = self.Wr(rel) # bs h
        sub_ = self.Ws(sub) # bs h
        alpha = torch.tanh(rel_ + sub_) # bs h
        att = self.Wv(alpha) # bs h
        weight = torch.softmax(att, dim=-1) #得到注意力分布
        res = weight * sub
        return res

class Rel_embedding(nn.Module): #就是一个全连接+激活函数sigmoid
    def __init__(self, input_size, output_size, dropout_rate): #
        super(Rel_embedding, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, int(output_size // 2), bias=False) #inputdim  outputdim
        self.rel2hidden = nn.Linear(int(output_size // 2), self.output_size) #
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.LeakyReLU()

    def forward(self, input_features):
        features_tmp = self.linear(input_features) # bs, h
        features_tmp = self.relu(features_tmp)
        # features_tmp = self.dropout(features_tmp)
        features_output = self.rel2hidden(features_tmp)
        return features_output

class Biaffine(nn.Module):
    def __init__(self, in1_features, in2_features, out_features, bias=(True, True, True)):
        super(Biaffine, self).__init__()
        self.in1_features = in1_features #h
        self.in2_features = in2_features #h
        self.out_features = out_features #R
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1])) # 3-dim -> 2-dim #(h+1)*R
        self.linear = nn.Linear(in_features=self.linear_input_size,
                                out_features=self.linear_output_size, #h+1 -> (h+1)*R
                                bias=False)
    def forward(self, input1, input2): #input bs*h
        input1=input1.unsqueeze(dim=1) #bs * 1 * h
        input2=input2.unsqueeze(dim=1)
        batch_size, _, dim1 = input1.size()
        _, _, dim2 = input2.size()
        if self.bias[0]:
            ones = input1.data.new(batch_size, 1,1).zero_().fill_(1)
            input1 = torch.cat((input1, Variable(ones)), dim=2)# input bs,1,h -> bs, 1, h+1, 新的一维都是1
            dim1 += 1
        if self.bias[1]:
            ones = input2.data.new(batch_size, 1,1).zero_().fill_(1)
            input2 = torch.cat((input2, Variable(ones)), dim=2)
            dim2 += 1
        affine = self.linear(input1) # bs,1,h+1 -> bs,1,（h+1)*r
        affine = affine.view(batch_size, self.out_features, dim2) #  bs,r,h+1
        input2 = torch.transpose(input2, 1, 2) #将input2的1维和2维进行转置 bs,1,h+1 -> bs,h+1,1
        biaffine = torch.transpose(torch.bmm(affine, input2), 1, 2) #做矩阵乘法 bs, r, 1 -> bs, 1, r
        biaffine = biaffine.contiguous().view(batch_size, 1, 1, self.out_features) #contiguous方法改变了多维数组在内存中的存储顺序，以便配合view方法使用
        # biaffine = torch.einsum('bin,anm,bjm->bija', input1, self.weight, input2)
        return biaffine.squeeze(dim=1).squeeze(dim=1) #bs r
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'in1_features=' + str(self.in1_features) \
               + ', in2_features=' + str(self.in2_features) \
               + ', out_features=' + str(self.out_features) + ')'

class BertForRE(BertPreTrainedModel):
    """
    Bert-based的模型，包括预训练模型和下游任务模型都是基于BertPreTrainedModel类，用于初始化权重参数和加载预训练描述。
    """
    def __init__(self, config, params):
        super().__init__(config)
        self.max_seq_len = params.max_seq_length
        self.rel_num = params.rel_num # rel数量
        # pretrain model 预训练模型
        self.bert = BertModel(config) #实例化bert模型
        # self.bert = RobertaModel(config)
        self.dropout =nn.Dropout(params.drop_prob)
        # 分别得到sor的隐藏表
        self.w1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.w2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.w3 = nn.Linear(config.hidden_size, config.hidden_size)
        self.w4 = nn.Linear(config.hidden_size, config.hidden_size)
        # self.p_r_embedding = nn.Embedding(params.rel_num, config.hidden_size) #建立潜在关系查询表
        self.p_r_embedding = Rel_embedding(params.rel_num, config.hidden_size, params.drop_prob)
        # s
        self.s_classier = nn.Linear(config.hidden_size, 2)
        #o
        self.o_classier_from_s = nn.Linear(config.hidden_size, 2)
        # 潜在关系注意力融合 潜在关系注意力机制 r
        # self.self_attention = Self_attention(config.hidden_size, config.hidden_size, config.hidden_size)
        self.p_classier = nn.Linear(config.hidden_size, params.rel_num)
        #基于主体和o，做双仿射关系分类
        #p
        self.biaffine=Biaffine(config.hidden_size, config.hidden_size, params.rel_num)
        self.lastnet = nn.Linear(config.hidden_size, params.rel_num)
        self.sigmoid = nn.Sigmoid()

        # 设计一个gate网络整合两个向量
        self.w5 = nn.Linear(config.hidden_size*2, 1)
        self.w6 = nn.Linear(config.hidden_size*2, 1)
        self.attention_net = Attention_net(config.hidden_size, config.hidden_size, config.hidden_size)
        self.init_weights()


    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            s2o_loc=None, #[batch, 2] #改为使用s mask
            so_mask = None,
            p_r_label = None
    ):
        head, tail, p_rel, rel, cls = self.get_embed(input_ids, attention_mask)
        s_pred = self.s_pred(head, cls) #bs seqlen 2
        o_pred = self.o_pred_from_s(s2o_loc, head, tail, cls) #s2o_loc bs,seqlen,2
        p_r_pred = self.p_r_pred(p_rel, attention_mask, cls)
        r_pred= self.r_pred_from_so(so_mask, p_r_label, head, tail, rel, attention_mask)
        return s_pred, o_pred, p_r_pred, r_pred

    def extract_entity(self, input, mask): # input :bs,seqlen,h  mask:bs,seqlen
        _, _, dim = input.shape
        entity = input * mask.unsqueeze(dim=-1) #bs seqlen, h
        entity = entity.sum(dim=1) / mask.sum(dim=-1, keepdim=True)  # BH/B1
        return entity #bs h
    def get_p_r_embedding(self, p_r):
        #p_r 为bs，rel
        # em = torch.arange(0, self.rel_num).to("cuda")
        # em = self.p_r_embedding(em) #r, h   bs r * r h bs h
        # _, hidden = em.shape
        # batch, _ = p_r.shape
        # em = torch.stack([em]*batch, dim=0) # bs, r, h
        # m = torch.stack([p_r]*hidden, dim=2) #bs, r, h
        # m = m.float()
        # p_r_ = torch.where(m != 0, em, m) #bs, r, h
        # #做pooling
        # p_r = p_r_.sum(dim=1) / p_r.sum(dim=1, keepdim=True)
        # 直接对潜在关系采用全连接的方式， bs,r -> bs h 需要考虑到维度
        p_r = self.p_r_embedding(p_r.float())
        return self.dropout(p_r) # bs, h
    def p_r_pred(self, p_r, mask, cls):
        #p_r:bs,seqlen,h
        # p_r = p_r.mean(dim=1)+cls
        p_r_ = self.masked_mean(p_r, mask) + cls
        p_r_ = self.p_classier(p_r_)
        return self.sigmoid(p_r_)

    # def p_r_attention(self, p_rel, rel, sub, entity_mask):
    #     # sub :bs,seqlen,h   rel:bs, seqle
    #     p_r = self.get_p_r_embedding(p_rel)
    #     entity = self.extract_entity(rel, entity_mask)
    #     p_r_entity = p_r + entity # bs h + bs h #实体和关系融合 # 将融合向量的方法换成*precision: precision: 0.868; recall: 0.829; f1: 0.848
    #     p_r_entity = p_r_entity.unsqueeze(dim=1)
    #     p_entity = self.self_attention(sub + p_r_entity) # self attention提高句子表征能力
    #     return p_entity

    def s_pred(self, head, cls):
        #暂时不考虑融合句子信息cls
        sub = self.s_classier(head + cls.unsqueeze(dim=1))
        return self.sigmoid(sub) # bs, seqlen, 2 #维度2一个表示实体头，一个表示实体尾

    def o_pred_from_s(self, sub, head, tail, cls):
        # 考虑  使用sub_mask
        sub = self.extract_entity(head, sub) #bs h
        tail = tail * sub.unsqueeze(dim=1) #bs seq h
        obj = self.o_classier_from_s(tail+cls.unsqueeze(dim=1))
        return self.sigmoid(obj)

    def masked_mean(self, sent, mask):
        mask_ = mask.masked_fill(mask==0, -1e9).float()
        score = torch.softmax(mask_, -1)
        return torch.matmul(score.unsqueeze(1), sent).squeeze(1)
    def r_pred_from_so(self, entiypair, p_r_label, head, tail, rel, mask):
        # s_entity = self.p_r_attention(p_r_label, rel, head, entity_mask=entiypair[:, 0])
        # o_entity = self.p_r_attention(p_r_label, rel, tail, entity_mask=entiypair[:, 1]) #bs seqlen h
        # s_entity = s_entity.mean(dim=1)
        # o_entity = o_entity.mean(dim=1) # precision: 0.884; recall: 0.848; f1: 0.866
        # # s_entity = self.extract_entity(s_entity, entiypair[:, 0]) #precision: 0.870; recall: 0.854; f1: 0.862
        # # o_entity = self.extract_entity(o_entity, entiypair[:, 1])

        s_entity = self.extract_entity(rel, entiypair[:, 0])  # 换成rel的时候是0.84
        o_entity = self.extract_entity(rel, entiypair[:, 1])
        p_r = self.get_p_r_embedding(p_r_label)

        ## p_r_sub = self.attention_net(p_r, head.sum(dim=1))#p_r + head.sum(dim=1)
        ## p_r_obj = self.attention_net(p_r, tail.sum(dim=1))#p_r + tail.sum(dim=1)
        p_r_sub = self.attention_net(p_r, self.extract_entity(head, entiypair[:, 0]))
        p_r_obj = self.attention_net(p_r, self.extract_entity(tail, entiypair[:, 1]))
        s_r = self.sigmoid(self.w5(torch.cat([p_r_sub, s_entity], dim=-1)))
        o_r = self.sigmoid(self.w6(torch.cat([p_r_obj, o_entity], dim=-1)))  # precision: 0.887; recall: 0.858; f1: 0.873  0.903; recall: 0.861; f1: 0.881    precision: 0.904; recall: 0.875; f1: 0.889
        s_entity = s_r * p_r_sub + (1-s_r) * s_entity
        o_entity = o_r * p_r_obj + (1-o_r) * o_entity
        s_entity = s_entity + p_r
        o_entity = o_entity + p_r
        # logist=self.biaffine(s_entity, o_entity) #BR
        logist = self.lastnet(s_entity + o_entity)
        r_pred=self.sigmoid(logist)
        return r_pred #BR



    def get_embed(self, input_ids, attention_mask):
        # pre-train model
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )  # sequence_output, pooled_output就是cls的输出代表整个句子，但往往用sequ_output来pooling, (hidden_states), (attentions)
        sequence_output = outputs[0]
        head = self.w1(sequence_output) # bs, seqlen, h
        tail = self.w2(sequence_output)
        # #rel 通过sequence_output
        p_rel = self.w3(sequence_output)
        rel = self.w4(sequence_output)
        cls = outputs[1] # bs, h
        head = head + tail[:, 0, :].unsqueeze(dim=1) #tail[:,0,:]维度为bs,h, unsqueeze bs,1,h
        tail = tail + head[:, 0, :].unsqueeze(dim=1) #s,o都融合cls句子信息
        head, tail, p_rel, rel, cls = self.dropout(head), self.dropout(tail), self.dropout(p_rel), self.dropout(rel), self.dropout(cls)
        return head, tail, p_rel, rel, cls


if __name__ == '__main__':
    from transformers import BertConfig
    import utils
    import os

    params = utils.Params()
    # Prepare model
    bert_config = BertConfig.from_json_file(os.path.join(params.bert_model_dir, 'config.json'))
    model = BertForRE.from_pretrained(config=bert_config,
                                      pretrained_model_name_or_path=params.bert_model_dir,
                                      params=params)

    model.to(params.device) #将模型加载到相应的设备

    for n, _ in model.named_parameters():
        print(n)
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    para2 = sum([np.prod(list(p.size())) for n,p in model.named_parameters() if 'biaffine.' in n])
    print(para)
    print(para2)
    type_size = 4
    print('Model {} : params:{:4f}M'.format(model._get_name(), para * type_size /1000 /1000))
    print('Model {} : params:{:4f}M'.format(model._get_name(), para2 * type_size /1000 /1000))






