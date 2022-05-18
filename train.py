import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import random
from tqdm import tqdm
import torch
import argparse
from torch import nn
import utils
from dataloader import CustomDataLoader
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from evaluate import evaluate
from model import BertForRE
# load args 加载参数
parser = argparse.ArgumentParser()

# load args 加载参数
parser = argparse.ArgumentParser()
parser.add_argument('--min_num', default=1e-7, type=float)
parser.add_argument('--seed', type=int, default=2022, help="random seed for initialization")
parser.add_argument('--ex_index', type=str, default=7)
parser.add_argument('--corpus_type', type=str, default="WebNLG", help="NYT, WebNLG, NYT*, WebNLG*")
parser.add_argument('--device_id', type=int, default=0, help="GPU index")
parser.add_argument('--epoch_num',  type=int, default=100, help="number of epochs") #required=True,
parser.add_argument('--multi_gpu', action='store_true', help="ensure multi-gpu training")
parser.add_argument('--restore_file', default=None, help="name of the file containing weights to reload")
parser.add_argument('--corres_threshold', type=float, default=0.5, help="threshold of global correspondence")
parser.add_argument('--rel_threshold', type=float, default=0.5, help="threshold of relation judgement")
parser.add_argument('--ensure_corres', action='store_true', help="correspondence ablation")
parser.add_argument('--ensure_rel', action='store_true', help="relation judgement ablation")
parser.add_argument('--emb_fusion', type=str, default="concat", help="way to embedding")
parser.add_argument('--num_negs', type=int, default=4,
                    help="number of negative sample when ablate relation judgement")
class CE():
    def __call__(self,args,targets, pred, from_logist=False):
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
            pred = torch.where(pred < 1 - args.min_num, pred, torch.ones(pred.shape).to("cuda") * 1 - args.min_num).to("cuda")
            pred = torch.where(pred > args.min_num, pred, torch.ones(pred.shape).to("cuda") * args.min_num).to("cuda")
            pred = torch.log(pred / (1 - pred))
        relu = nn.ReLU()
        # 计算传统的交叉熵loss
        loss = relu(pred) - pred * targets + torch.log(1 + torch.exp(-1 * torch.abs(pred).to("cuda"))).to("cuda")
        return loss

def train(model, dataloader, optimizer, scheduler,losstor, p_r_lossor, params, ex_params):
    '''
    one epoch train
    :param model:
    :param data_iterator:
    :param optimizer:
    :param params:
    :param ex_params:
    :return:
    '''
    model.train()
    epoch_loss = 0
    with tqdm(total=dataloader.__len__(), desc="train", ncols=150) as t:
        for i, batch in enumerate(dataloader):
            batch = [d.to("cuda") for d in batch]
            batch_input_ids, batch_attention_mask, batch_sub_labels, batch_s_mask, batch_s2o_loc, batch_obj_labels, batch_sub_ids, \
            batch_obj_ids, batch_p_r, batch_rel, batch_so_mask = batch #句子id，attentionmask，

            s_pred, o_pred, p_r_pred, r_pred = model(batch_input_ids, batch_attention_mask,  # bs,seqlen,2   r_pred bs,r
                  batch_s_mask, entiypair=[batch_sub_ids,batch_obj_ids], so_mask=batch_so_mask, p_r_label=batch_p_r)

            # 计算损失
            def get_loss(target, pred, mask):  # target 的维度【bs, seq_len, h】
                '''
                传入三个参数 目标，预测，遮盖
                '''
                loss = losstor(args, targets=target, pred=pred)  # BL2
                loss = torch.mean(loss, dim=2).to("cuda")  # BL
                loss = torch.sum(loss * mask).to("cuda") / torch.sum(mask).to("cuda")
                return loss

            s_loss = get_loss(target=batch_sub_labels, pred=s_pred, mask=batch_attention_mask)
            o_loss = get_loss(target=batch_obj_labels, pred=o_pred, mask=batch_attention_mask)

            # p_r_loss = p_r_lossor(p_r_pred, batch_p_r.float())
            p_r_loss = losstor(args, targets=batch_p_r, pred=p_r_pred)
            p_r_loss = p_r_loss.mean()
            r_loss = losstor(args, targets=batch_rel, pred=r_pred)  # batch_r
            r_loss = r_loss.mean()


            # loss = s_loss/(s_loss.detach()+params.eps) + o_loss/(o_loss.detach()+params.eps) + p_r_loss/(p_r_loss.detach()+params.eps) + r_loss/(r_loss.detach()+params.eps)
            loss = s_loss + o_loss + r_loss + p_r_loss
            loss.backward()
            epoch_loss += loss.item()  # bs每个loss加，最终得到整个epoch
            torch.nn.utils.clip_grad_norm_(model.parameters(), params.max_grad_norm)  # 梯度裁剪 根据参数的范数来衡量的
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

            t.set_postfix(loss="%.4lf" % (loss.cpu().item()),
                          s_loss="%.4lf" % (s_loss.cpu().item()),
                          o_loss="%.4lf" % (o_loss.cpu().item()),
                          p_r_loss="%.4lf" % (p_r_loss.cpu().item()),
                          r_loss="%.4lf" % (r_loss.cpu().item()))
            t.update(1)

def train_and_evaluate(model, params, ex_params, restore_file=None):
    """Train the model and evaluate every epoch.训练及评估"""
    print("进入训练和评估")
    # Load training data and val data
    dataloader = CustomDataLoader(params)
    train_loader = dataloader.get_dataloader(data_sign='train', ex_params=ex_params)
    val_loader = dataloader.get_dataloader(data_sign='val', ex_params=ex_params)
    test_loader = dataloader.get_dataloader(data_sign='test', ex_params=ex_params)
    print("加载数据完毕")
    # reload weights from restore_file if specified
    if restore_file is not None: #加载模型参数，如果存在
        restore_path = os.path.join(params.model_dir, args.restore_file + '.pth.tar')
        print("Restoring parameters from {}".format(restore_path))
        # 读取checkpoint
        model, optimizer = utils.load_checkpoint(restore_path)
    print("模型加载gpu")
    model.to(params.device) #模型加载到GPU
    print("模型加载完毕")
    # parallel model
    if params.n_gpu > 1 and args.multi_gpu:
        model = torch.nn.DataParallel(model) #多gpu

    # Prepare optimizer
    # fine-tuning
    """ 优化器准备 """
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "bert." in n],
            "weight_decay": 0.0,
            "lr": params.fin_tuning_lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if "bert." not in n],
            "weight_decay": 0.0,
            "lr": params.downs_en_lr,
        }
    ]
    #len(train_loader)为数据集大小/batchsize, num_train_optimization_steps更新次数
    num_train_optimization_steps = len(train_loader) * args.epoch_num #//除法返回整数部分
    optimizer = AdamW(optimizer_grouped_parameters, lr=params.downs_en_lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup( # 初始预热步数，整个过程总训练步数
        optimizer, num_warmup_steps=params.warmup_prop * num_train_optimization_steps, num_training_steps=num_train_optimization_steps
    )
    print("优化器准备完成")
    # patience stage
    best_val_f1 = 0.0
    patience_counter = 0
    losstor = CE()
    p_r_loss = nn.BCELoss(reduction='mean')
    for epoch in range(1, args.epoch_num + 1):
        print("Epoch {}/{}".format(epoch, args.epoch_num))
        # Train for one epoch on training set
        train(model, train_loader, optimizer, scheduler, losstor, p_r_loss, params, ex_params) #train
        val_metrics = evaluate(model, val_loader, params, ex_params, mark='Val') #evaluate
        test_metrics = evaluate(model, test_loader, params, ex_params, mark='test')
        val_f1 = val_metrics['f1'] # 得到F1分数
        improve_f1 = val_f1 - best_val_f1

        # Save weights of the network
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self model有module这个attr
        optimizer_to_save = optimizer
        utils.save_checkpoint({'epoch': epoch + 1,
                               'model': model_to_save,
                               'optim': optimizer_to_save},
                              is_best=improve_f1 > 0,
                              checkpoint=params.model_dir)
        params.save(params.ex_dir / 'params.json') #保存

        # stop training based params.patience
        if improve_f1 > 0:
            print("- Found new best F1")
            best_val_f1 = val_f1
            if improve_f1 < params.patience: #提高小于期待的多少或者没有提高
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping and logging best f1
        if (patience_counter > params.patience_num and epoch > params.min_epoch_num) or epoch == args.epoch_num:
            print("Best val f1: {:05.2f}".format(best_val_f1))
            break


if __name__ == '__main__':
    args = parser.parse_args()
    params = utils.Params(args.ex_index, args.corpus_type)
    ex_params = {
        'ensure_corres': args.ensure_corres,
        'ensure_rel': args.ensure_rel,
        'num_negs': args.num_negs,
        'emb_fusion': args.emb_fusion
    }

    if args.multi_gpu:
        params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("使用多gpu")
        n_gpu = torch.cuda.device_count()
        print("gpu数量{}".format(n_gpu))
        params.n_gpu = n_gpu
    else:
        torch.cuda.set_device(args.device_id)
        print('current device:', torch.cuda.current_device())
        params.n_gpu = n_gpu = 1

    # Set the random seed for reproducible experiments 为可重复实验设置随机种子
    # 传入的数值用于指定随机数生成时所用算法开始时所选定的整数值，如果使用相同的seed()值，则每次生成的随机数都相同；
    random.seed(args.seed)
    torch.manual_seed(args.seed)  # 设置随机生成种子
    params.seed = args.seed
    # if n_gpu > 0:
    #     torch.cuda.manual_seed_all(args.seed)

    #
    print(f"Model type:")
    print("device: {}".format(params.device))

    print('Load pre-train model weights...')

    model = BertForRE.from_pretrained(config=os.path.join(params.bert_model_dir, 'config.json'),
                                      pretrained_model_name_or_path=params.bert_model_dir,
                                      params=params)
    print('-done')

    # Train and evaluate the model
    print("Starting training for {} epoch(s)".format(args.epoch_num))
    train_and_evaluate(model, params, ex_params, args.restore_file)