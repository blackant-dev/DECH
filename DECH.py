
from CONFIG import args
import torch
from models import *
from LOSS import *
import CONFIG
import os

model_save_path = os.path.join(CONFIG.model_save_dirnme,f'{args.data_name}_{args.bit}bit_last_{args.L3idx}')

import torch

import logging
os.makedirs('/media/hdd4/liy/DECH__NEW/LogFiles',exist_ok=True)
LogFilesPath = '/media/hdd4/liy/DECH__NEW/LogFiles'
# 创建第一个logger和handler
logger1 = logging.getLogger("map_all")
logger1.setLevel(logging.INFO)
handler1 = logging.FileHandler(f'{LogFilesPath}/map_all.log')
formatter1 = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler1.setFormatter(formatter1)
logger1.addHandler(handler1)

# 创建第二个logger和handler
logger2 = logging.getLogger("map_500")
logger2.setLevel(logging.INFO)
handler2 = logging.FileHandler(f'{LogFilesPath}/map_500.log')
formatter2 = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler2.setFormatter(formatter2)
logger2.addHandler(handler2)

import numpy as np 
def i2t(npts, sims, return_ranks=False, mode='coco'):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        # rank = np.where((labels[inds].mm(labels[index].t()) > 0).sum(-1))[0][0]
        rank = np.where(inds == index)[0][0]
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)
    
def generate_data_laoder():
    from DATASET import CMDataset
    train_dataset = CMDataset(
        data_name=args.data_name,
        partition='train'
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    retrieval_dataset = CMDataset(
        data_name=args.data_name,
        partition='retrieval'
    )
    retrieval_loader = torch.utils.data.DataLoader(
        retrieval_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
    )

    test_dataset = CMDataset(
        data_name=args.data_name,
        partition='test'
    )
    query_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader,query_loader,retrieval_loader,4096,retrieval_dataset.text_dim

def generate_hash_code(data_loader,image_model,text_model):
    imgs, txts, labs = [], [], []
    # imgs_fea,txts_fea = [],[]
    with torch.no_grad():
        for batch_idx, (images, texts, targets) in enumerate(data_loader):
            images_outputs = [image_model(images.cuda().float())]
            texts_outputs = [text_model(texts.cuda().float())]
            imgs += images_outputs
            txts += texts_outputs
            labs.append(targets.float())

        imgs = torch.cat(imgs).sign_()#.cpu().numpy()
        txts = torch.cat(txts).sign_()#.cpu().numpy()
        labs = torch.cat(labs)#.cpu().numpy()
        
    return imgs,txts,labs

def getbdu(evidence):
    L = (2+evidence[:,0]+evidence[:,1])
    b = evidence[:,0] / L
    d = evidence[:,1] / L
    u = 2 / L
    return b,d,u
def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH
from tqdm import tqdm

def fx_calc_map_multilabel_k(retrieval, retrieval_labels, query, query_label, k=None, metric='hamming'):
    retrieval, retrieval_labels, query, query_label = [i.cuda().to(dtype=torch.float64) for i in [retrieval, retrieval_labels, query, query_label]]
    # qB, rB, query_label, retrieval_label = [i.cuda().to(dtype=torch.float64) for i in [qB, rB, query_label, retrieval_label]]
    import scipy.spatial
    # dist = scipy.spatial.distance.cdist(query, retrieval, metric)
    # ord = dist.argsort()
    # numcases = dist.shape[0]
    
    numcases = query.shape[0]
    if k == None:
        k = retrieval_labels.shape[0]
    res = []
    for i in (range(numcases)):
        hamm = calc_hamming_dist(query[i,:],retrieval).cuda()
        _, ord = torch.sort(hamm)
        order = ord.reshape(-1)[0: k]
        tmp_label =  (query_label[i,:]@retrieval_labels[order].T)>0
        tmp_label =tmp_label.float().cuda()
        if tmp_label.sum() > 0:
            prec = torch.cumsum(tmp_label,-1) / torch.arange(1.0, 1 + tmp_label.shape[0]).cuda()
            total_pos = float(tmp_label.sum())
            if total_pos > 0:
                res += [tmp_label@prec/total_pos]
    return torch.mean(torch.tensor(res))

def test(query_loader,retrieval_loader,image_model,text_model,evidence_model,epoch):
    global pre_val,no_update_count,state_dict
    # global state_dict
    qX,qY,qL = generate_hash_code(query_loader,image_model,text_model)
    rX,rY,rL = generate_hash_code(retrieval_loader,image_model,text_model)
    MAPi2t = fx_calc_map_multilabel_k(rY,rL,qX,qL,None)
    MAPt2i = fx_calc_map_multilabel_k(rX,rL,qY,qL,None)
    print(MAPi2t,MAPt2i)
def main():
    if args.data_name == 'nus_wide_deep':
        args.hiden_layer = 2
    train_loader,query_loader,retrieval_loader,image_dim,text_dim = generate_data_laoder()
    evidence_model = EvidenceNet(args.bit,args.tau).cuda()
    image_model = ImageNet(image_dim,args.bit,hiden_layer=args.hiden_layer).cuda()
    text_model = TextNet(text_dim,args.bit,hiden_layer=args.hiden_layer).cuda()
    
    parameters = list(image_model.parameters()) + list(text_model.parameters())  + list(evidence_model.parameters())
    optimizerTxt = torch.optim.Adam(image_model.parameters(), 1e-4)
    optimizerImg = torch.optim.Adam(text_model.parameters(), 1e-4)
    optimizerEvi = torch.optim.Adam(evidence_model.parameters(),1e-4)
    if args.continue_eval == 1:
        status_dict = torch.load(model_save_path)
        image_model.load_state_dict(status_dict['image_model_state_dict'])
        text_model.load_state_dict(status_dict['text_model_state_dict'])
        evidence_model.load_state_dict(status_dict['evidence_model_state_dict'])
      
        
    # optimizer = Lion(parameters, lr=1e-5)
    for epoch in range(0, args.max_epochs):
        train_loss=0
        print(f'当前{epoch} / {args.max_epochs}')
        for batch_idx, (images, texts, label) in (enumerate(train_loader)):
            
                
                
                images_outputs = image_model(images.cuda().float())
                texts_outputs = text_model(texts.cuda().float())
                label = label.cuda().float()
                evidencei2t = evidence_model(images_outputs,texts_outputs,'i2t')
                evidencet2i = evidence_model(images_outputs,texts_outputs,'t2i')
                GND = (label@label.T>0).float().cuda().view(-1, 1)
                
                lossI2t = edl_log_loss(evidencei2t,torch.cat([GND,1-GND],dim=1),epoch,2,42)
                lossT2i =edl_log_loss(evidencet2i,torch.cat([GND,1-GND],dim=1),epoch,2,42)
                
                loss = lossI2t+lossT2i
                optimizerTxt.zero_grad()
                optimizerImg.zero_grad()
                optimizerEvi.zero_grad()
                loss.backward()
                from torch.nn.utils.clip_grad import clip_grad_norm_
                
                clip_grad_norm_(parameters, max_norm=1.0)

                
                optimizerTxt.step()
                optimizerImg.step()
                optimizerEvi.step()
                
                
                train_loss += loss.item()


        test(query_loader,retrieval_loader,image_model,text_model,evidence_model,epoch)
    eval_res()
def eval_res():
    train_loader,query_loader,retrieval_loader,image_dim,text_dim = generate_data_laoder()
    evidence_model = EvidenceNet(args.bit,args.tau).cuda().eval()
    image_model = ImageNet(image_dim,args.bit).cuda().eval()

    text_model = TextNet(text_dim,args.bit).cuda().eval()
    status_dict = torch.load(model_save_path)
    image_model.load_state_dict(status_dict['image_model_state_dict'])
    text_model.load_state_dict(status_dict['text_model_state_dict'])
    evidence_model.load_state_dict(status_dict['evidence_model_state_dict'])
    qX,qY,qL = generate_hash_code(query_loader,image_model,text_model)
    rX,rY,rL = generate_hash_code(retrieval_loader,image_model,text_model)
    epoch=status_dict['epoch']
    
    MAPi2t =fx_calc_map_multilabel_k(rY,rL,qX,qL,2000)
    MAPt2i =fx_calc_map_multilabel_k(rX,rL,qY,qL,2000)
    logger2.info(f'i2t:{MAPi2t},t2i:{MAPt2i},data_name = {args.data_name}, bit = {args.bit},epoch={epoch},L3idx={args.L3idx},tau = {args.tau}')

    hash_code = {"rI":rX, "rT":rY, "rL":rL,"qI":qX, "qT":qY, "qL":qL}
    MAPi2t = fx_calc_map_multilabel_k(rY,rL,qX,qL,None)
    MAPt2i = fx_calc_map_multilabel_k(rX,rL,qY,qL,None)
    logger1.info(f'i2t:{MAPi2t},t2i:{MAPt2i},data_name = {args.data_name}, bit = {args.bit},epoch={epoch},L3idx={args.L3idx},tau = {args.tau}')
    file_path = os.path.join(os.path.dirname(__file__),f'checkpoints', f'MYMETHOD_{str(args.bit)}_{args.data_name}_{args.L3idx}')
    os.makedirs(os.path.join(os.path.dirname(__file__),f'checkpoints'),exist_ok=True)
    with open(file_path,'wb') as f:
        import pickle
        pickle.dump(hash_code,f)
    
    
if __name__ == '__main__':
    # print('ds')
    if args.is_eval == 1:
        eval_res()
    else:
        main()