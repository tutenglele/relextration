import torch
from torch import nn
p_r_embedding = nn.Embedding(4, 6)
class CE():
    def __call__(self,targets, pred, from_logist=False):
        '''
        计算二分类交叉熵
        :param targets: [batch,seq,2]
        :param pred: [batch,seq,2]
        :param from_logist:是否没有经过softmax/sigmoid
        :return: loss.shape==targets.shape==pred.shape
        '''
        if not from_logist: # 默认这里是true
            '''返回到没有经过softmax/sigmoid得张量'''
            # 截取pred，防止趋近于0或1,保持在[min_num,1-min_num]
            pred = torch.where(pred < 1 - 1e-7, pred, torch.ones(pred.shape).to("cuda") * 1 - 1e-7).to("cuda")
            pred = torch.where(pred > 1e-7, pred, torch.ones(pred.shape).to("cuda") * 1e-7).to("cuda")
            pred = torch.log(pred / (1 - pred))
        relu = nn.ReLU()
        # 计算传统的交叉熵loss
        loss = relu(pred) - pred * targets + torch.log(1 + torch.exp(-1 * torch.abs(pred).to("cuda"))).to("cuda")
        return loss
def get_p_r_embedding(p_r):
    #p_r 为bs，rel
    em = torch.arange(0, 4).to("cpu")
    em = p_r_embedding(em) #r, h
    _, hidden = em.shape # ,20, 32
    batch, _ = p_r.shape # bs, 20
    em = torch.stack([em]*batch, dim=0) # bs, r, h
    m = torch.stack([p_r]*hidden, dim=2) #bs, r, h
    m = m.float()
    p_r_ = torch.where(m != 0, em, m) #bs, r, h
    print(p_r_)
    #做pooling
    p_r = p_r_.sum(dim=1) / p_r.sum(dim=1, keepdim=True)
    return p_r # bs, h
if __name__ == '__main__':
    p_r = torch.Tensor([[1,1,1,0],[1,0,1,0],[1,0,1,0]])
    p_r.to("cpu")
    r = get_p_r_embedding(p_r)

    print(p_r_embedding(torch.arange(0,3)))
    print(r)