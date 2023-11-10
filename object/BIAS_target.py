import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
#torch.cuda.current_device()
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from loss import CrossEntropyFeatureAug, CrossEntropyOn,CrossEntropyFeatureAugWeight
import torch.nn.functional as F

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load_t(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args._true_test_dset_path).readlines()
    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_tar = new_tar.copy()

    dsets["test"] = ImageList_idx(txt_tar, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)
    return dset_loaders


def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders


def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

def train_target(args):
    dset_loaders = data_load(args)
    
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda() 
    elif args.net[0:3] == 'ale':
        netF = network.AlexBase().cuda() 

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'    
    netC.load_state_dict(torch.load(modelpath))
    

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
   
    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0
    best_acc = 0
    

    while iter_num < max_iter:

        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0:
            netF.eval()
            netB.eval()
            netC.eval()
            if iter_num /interval_iter == 0:
                mem_label,loc_label,true_label,mem_weight,confi_loc,confi_label,fea_bank,flag,sel_label= obtain_label(dset_loaders['test'], netF, netB, netC, args)
            else:
                mem_label,mem_weight,confi_loc,confi_label = obtain_label_rectify(dset_loaders['test'], netF, netB, netC, args,flag,sel_label)
            mem_label = torch.from_numpy(np.array(mem_label)).cuda()
            mem_weight = torch.from_numpy(np.array(mem_weight)).cuda()
            netF.train()
            netB.train()
            netC.train()

        
        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)

        inputs_mix_hard = copy.deepcopy(inputs_test.cpu().detach())
        weight_mix = torch.zeros(len(tar_idx))
        mix_pred = []
        if iter_num / interval_iter >= 0:
            for index in range(len(tar_idx)):
                if tar_idx[index] in confi_loc:
                    weight_mix[index] = 1.0
                else:
                    weight_mix[index] = 0.0
            for index in range(len(tar_idx)):
                label = np.zeros(args.class_num)
                if weight_mix[index] == 1.0:
                    lam = np.random.beta(args.alpha,args.alpha)
                    idx_confi_label = confi_label[confi_loc.index(tar_idx[index])]
                    sel_id = np.random.randint(0,len(loc_label))
                    inputs_mix_hard[index] = lam *inputs_test[index] + (1-lam) * fea_bank[loc_label[sel_id]].cuda()
                    label[idx_confi_label] = lam
                    label[true_label[sel_id]] = 1-lam
                    mix_pred.append(label)
                else:
                    mix_pred.append(label)
                    
            outputs_test_mix = netC(netB(netF(inputs_mix_hard.cuda())))
            classifier_mix_loss = CrossEntropyFeatureAugWeight(num_classes=args.class_num)(outputs_test_mix, torch.tensor(mix_pred).cuda(),weight_mix.double().cuda()).float() 

        pred = mem_label[tar_idx]
        weight = mem_weight[tar_idx]
        classifier_loss = CrossEntropyFeatureAugWeight(num_classes=args.class_num)(outputs_test, pred.cuda(),weight.cuda()).float()    
        
        if iter_num / interval_iter >= 0:
            # print(classifier_loss)
            classifier_loss += classifier_mix_loss
            # print(classifier_mix_loss)


        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss = classifier_loss + im_loss 

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()
        


        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            if args.dset=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            if best_acc <= acc_s_te:
                best_acc = acc_s_te
            netF.train()
            netB.train()
            netC.train()     
    print(best_acc) 
    return netF, netB, netC

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s
def cal_ratio(n_a):
    true_cls = int(n_a[0])
    cls_filter  = n_a == true_cls 
    cls_filter = cls_filter.tolist()
    list_loc =  [i for i,x in enumerate(cls_filter ) if x ==1 ]
    
    return len(list_loc)/ len(n_a)

def obtain_label(loader, netF, netB, netC, args):
    h_dict = {}
    loc_dict = {}
    fea_sel_dict = {}
    label_sel_dict = {}
    for cls in range(args.class_num):
        h_dict[cls] = []
        loc_dict[cls] = []
        fea_sel_dict[cls] = []
        label_sel_dict[cls] = []

    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        # sel_path = iter_test.dataset.imgs
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            feas_uniform = F.normalize(feas)
            outputs = netC(feas)
            if start_test:
                all_inputs = inputs.float().cpu()
                all_fea = feas_uniform.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_inputs = torch.cat((all_inputs, inputs.float().cpu()), 0)
                all_fea = torch.cat((all_fea, feas_uniform.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    fir_max, predict = torch.max(all_output, 1)
    # more smaller, more consideration
    max2 = torch.topk(all_output, k=2, dim=1, largest=True).values
    BVSB_uncertainty = max2[:, 0] - max2[:, 1]
    # more higher, more consideration
    con = 1 - fir_max


    accuracy_ini = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])


    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > args.threshold)
    labelset = labelset[0]
    

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]
    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]
    before_pred_label = copy.deepcopy(pred_label)
    accuracy = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    if(len(labelset) < args.class_num):
        print("missing classes") 

    distance = torch.tensor(all_fea) @  torch.tensor(all_fea).t()
    _, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.KK)
    _, idx_near_c = torch.topk(distance, dim=-1, largest=True, k=2)

    if args.standard == 'con':
        stand = (-con)
    elif args.standard == 'bvsb':
        stand = BVSB_uncertainty
    elif args.standard == 'ent':
        stand = (-ent)

    loc_label = []
    true_label = []
    sor = np.argsort(stand)
    index = 0
    index_v = 0
    args.SSN =  int(len(pred_label) * (args.ratio*2))
    can_loc_label = []
    for index in range(args.SSN):
        r_i = sor[index]
        can_loc_label.append(r_i)
        
    flag =  np.zeros(len(pred_label)) - 1
    sel_label = np.zeros(len(pred_label)) - 1
    set_sel = set()
    #analysis first
    near_label_h_all = np.array(all_label[np.array(idx_near)[can_loc_label]])
    print("The accuracy based on nearest neighbors")
    print((np.sum(near_label_h_all[:,0] == near_label_h_all[:,1]) )/ len(near_label_h_all[:,0]))
    print(np.sum(before_pred_label[can_loc_label] == all_label[can_loc_label].numpy())/ len(all_label[can_loc_label]))
    
    
    init_neighbor_set = []
    idx_near_all = np.array(idx_near_c)[can_loc_label]
    idx_near_all_data = np.array(idx_near_c)
    for index in range(len(can_loc_label)):
        anchor_index = can_loc_label[index].tolist()
        if anchor_index in set_sel:
            continue
        near_label_same =[]
        near_label_same.append(anchor_index)
        idx_near_anchor_all = idx_near_all[index]
        for index_near in range(len(idx_near_anchor_all)):
            if anchor_index != idx_near_all[index][index_near]:
                idx_near_i = idx_near_all[index][index_near]
            else:
                continue
            if anchor_index in idx_near_all_data[idx_near_i]:
                near_label_same.append(idx_near_i)
        set_sel = set_sel.union(np.array(near_label_same))
        if (len(near_label_same)) > 1:
            init_neighbor_set.append(near_label_same)
            flag[near_label_same] = 0

    
    ratio_list =[]
    for index in range(len(init_neighbor_set)):
        group_loc = init_neighbor_set[index]
        true_g = np.array(all_label[group_loc].int())
        ratio_g = cal_ratio(true_g)
        ratio_list.append(ratio_g)
    print(ratio_list)
    print(np.mean(np.array(ratio_list)))
    print(len(init_neighbor_set))



    args.SSN = (int(len(pred_label) * (args.ratio))) 
    for index in range(len(init_neighbor_set)):
        if index < args.SSN:
            com_data = init_neighbor_set[index]
            flag[com_data[0]] = 1.0
            flag[com_data[1:]] = 2.0
            sel_label[com_data] = int(all_label[com_data[0]])  

    sor = np.argsort(stand)
    args.SSN = (int(len(pred_label) * (args.ratio))) - len(init_neighbor_set)
    index = 0
    index_v = 0
    while index < args.SSN:
        r_i = sor[index_v]
        if flag[r_i] == 1.0 or flag[r_i] == 2.0:
            index_v += 1
            continue
        else:
            flag[r_i] = 1.0
            sel_label[r_i] = int(all_label[r_i])
            pred_label[r_i] = all_label[r_i]
            index += 1
            
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    for index in range(len(pred_label)):
        if flag[index] == 1.0 or flag[index] == 2.0:
            pred_label[index] = sel_label[index]
            loc_label.append(index)
            true_label.append(int(sel_label[index]))
    print(len(loc_label))
    acc1 = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}% -> {:.2f}%-> {:.2f}%'.format(accuracy_ini * 100, accuracy * 100, acc * 100,acc1 * 100)
    print(log_str +'\n')
    args.out_file.write(log_str+'\n')
    args.out_file.flush()
    count = flag > 0
    count = count.tolist()
    count = [i for i,x in enumerate(count) if x ==1 ]
    print("The number of active samples after RAS")
    print(len(count))
    count = flag == 1
    count = count.tolist()
    count = [i for i,x in enumerate(count) if x ==1 ]
    print("The number of strict active samples")
    print(len(count))
    #select confi data
    sor = np.argsort(-stand)
    confi_loc = []
    confi_label = []
    for index in range(int(len(pred_label) * args.sel_cal_ratio)):
        r_i = sor[index]
        if r_i in loc_label:
            continue
        else:
            confi_loc.append(r_i)
            confi_label.append(pred_label[r_i])
    print(" the accracy of selected confident label:")
    print(np.sum(pred_label[confi_loc] == all_label[confi_loc].float().numpy()) / len(all_fea[confi_loc]))

    pred_true = []
    weight = []
    

    for index in range(len(pred_label)):
        label = np.zeros(args.class_num)
        if flag[index] == 1.0:
            label[int(sel_label[index])] = 1.0
            weight.append(1.0)
        else:
            label[pred_label[index]] = 1.0
            weight.append(args.cls_par)
        pred_true.append(label)

    return pred_true,loc_label,true_label,weight,confi_loc,confi_label,all_inputs,flag,sel_label

def obtain_label_rectify(loader, netF, netB, netC, args,flag,sel_label):
    h_dict = {}
    loc_dict = {}
    fea_sel_dict = {}
    label_sel_dict = {}
    for cls in range(args.class_num):
        h_dict[cls] = []
        loc_dict[cls] = []
        fea_sel_dict[cls] = []
        label_sel_dict[cls] = []

    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            feas_uniform = F.normalize(feas)
            outputs = netC(feas)
            if start_test:
                all_fea = feas_uniform.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas_uniform.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    fir_max, predict = torch.max(all_output, 1)
    # more smaller, more consideration
    max2 = torch.topk(all_output, k=2, dim=1, largest=True).values
    BVSB_uncertainty = max2[:, 0] - max2[:, 1]
    # more higher, more consideration
    con = 1 - fir_max

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        
    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]
    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)


    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format( accuracy * 100, acc * 100)
    print(log_str+'\n')
    args.out_file.write(log_str+'\n')
    args.out_file.flush()

    if args.standard == 'con':
        stand = con
    elif args.standard == 'bvsb':
        stand = -BVSB_uncertainty
    elif args.standard == 'ent':
        stand = ent

    sor = np.argsort(stand)
    confi_loc = []
    confi_label = []
    for index in range(int(len(pred_label) * args.sel_cal_ratio)):
        r_i = sor[index]
        if flag[r_i] == 1.0 or flag[r_i] == 2.0:
            continue
        else:
            confi_loc.append(r_i)
            confi_label.append(pred_label[r_i])
    print(" the accracy of selected confident label:")
    print(np.sum(pred_label[confi_loc] == all_label[confi_loc].float().numpy()) / len(all_fea[confi_loc]))



    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format( accuracy * 100,acc * 100)
    print(log_str+'\n')
    args.out_file.write(log_str+'\n')
    args.out_file.flush()
    count = flag == 1 
    count = count.tolist()
    count = [i for i,x in enumerate(count) if x ==1 ]
    print(len(count))
    pred_true = []
    weight = []
    for index in range(len(pred_label)):
        label = np.zeros(args.class_num)
        if flag[index] == 1.0:
            label[int(sel_label[index])] = 1.0
            weight.append(1.0)
        else:
            label[pred_label[index]] = 1.0
            weight.append(args.cls_par_af)
        pred_true.append(label) 
    return pred_true,weight,confi_loc,confi_label

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BIAS')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=1, help="source")
    parser.add_argument('--t', type=int, default=0, help="target")
    parser.add_argument('--max_epoch', type=int, default=20, help="max iterations")
    parser.add_argument('--interval', type=int, default=20)
    parser.add_argument('--KK', type=int, default=20)
    parser.add_argument('--KK_E', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--SSN', type=int, default=200)
 
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--cls_par_af', type=float, default=0.0)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--ratio', type=float, default=0.05)
    parser.add_argument('--sel_cal_ratio', type=float, default=0.4)
    parser.add_argument('--alpha', type=float, default=5)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='./Result/Office-Home/Resnet50/')
    parser.add_argument('--output_src', type=str, default='./object/meckps50')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--standard', type=str, default='con', choices=['con', 'ent','bvsb'])
    parser.add_argument('--issave', type=bool, default=False)
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'Real_world']
        args.class_num = 65 
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    print(args.gpu_id)
    # print(args.overratio)

    folder = './datasets/data/'
    args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
    args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
    args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'



    args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
    args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
    args.name = names[args.s][0].upper()+names[args.t][0].upper()

    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)


    args.savename = 'ASFDA_standard'+ args.standard +'_ratio'+ str(args.ratio) + '_beta' +str(args.cls_par)+'alpha_'+str(args.alpha)+'_bias_'+str(args.sel_cal_ratio)
    args.out_file = open(osp.join(args.output_dir, args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    train_target(args)