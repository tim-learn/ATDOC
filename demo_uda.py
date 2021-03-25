import argparse
import os
import os.path as osp
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import loss
import random, pdb, math
import sys, copy
from tqdm import tqdm
import utils
import scipy.io as sio

def data_load(args):   
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.RandomCrop((224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.CenterCrop((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    source_set = utils.ObjectImage('', args.s_dset_path, train_transform)
    target_set = utils.ObjectImage_mul('', args.t_dset_path, train_transform)
    test_set = utils.ObjectImage('', args.test_dset_path, test_transform)

    dset_loaders = {}
    dset_loaders["source"] = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.worker, drop_last=True)
    dset_loaders["target"] = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.worker, drop_last=True)
    dset_loaders["test"] = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size*3,
        shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def data_load_y(args, labels):   
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.RandomCrop((224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.CenterCrop((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # pdb.set_trace()
    source_set = utils.ObjectImage_y('', args.t_dset_path, train_transform, labels)
    target_set = utils.ObjectImage_mul('', args.s_dset_path, train_transform)
    test_set = utils.ObjectImage('', args.s_dset_path, test_transform)

    dset_loaders = {}
    dset_loaders["source"] = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.worker, drop_last=True)
    dset_loaders["target"] = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.worker, drop_last=True)
    dset_loaders["test"] = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size*3,
        shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def lr_scheduler(optimizer, init_lr, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def bsp_loss(feature):
    train_bs = feature.size(0) // 2
    feature_s = feature.narrow(0, 0, train_bs)
    feature_t = feature.narrow(0, train_bs, train_bs)
    _, s_s, _ = torch.svd(feature_s)
    _, s_t, _ = torch.svd(feature_t)
    sigma = torch.pow(s_s[0], 2) + torch.pow(s_t[0], 2)
    sigma *= 0.0001
    return sigma

def train(args, validate=False, label=None):
    ## set pre-process
    if validate:
        dset_loaders = data_load_y(args, label)
    else:
        dset_loaders = data_load(args)
    class_num = args.class_num
    class_weight_src = torch.ones(class_num, ).cuda()
    ##################################################################################################

    ## set base network
    if args.net == 'resnet101':
        netG = utils.ResBase101().cuda()
    elif args.net == 'resnet50':
        netG = utils.ResBase50().cuda()  

    netF = utils.ResClassifier(class_num=class_num, feature_dim=netG.in_features, 
        bottleneck_dim=args.bottleneck_dim).cuda()

    max_len = max(len(dset_loaders["source"]), len(dset_loaders["target"]))
    args.max_iter = args.max_epoch * max_len

    ad_flag = False
    if args.method in {'DANN', 'DANNE'}:
        ad_net = utils.AdversarialNetwork(args.bottleneck_dim, 1024, max_iter=args.max_iter).cuda()
        ad_flag = True
    if args.method in {'CDAN', 'CDANE'}:
        ad_net = utils.AdversarialNetwork(args.bottleneck_dim*class_num, 1024, max_iter=args.max_iter).cuda() 
        random_layer = None
        ad_flag = True  

    optimizer_g = optim.SGD(netG.parameters(), lr = args.lr * 0.1)
    optimizer_f = optim.SGD(netF.parameters(), lr = args.lr)
    if ad_flag:
        optimizer_d = optim.SGD(ad_net.parameters(), lr = args.lr)
   
    base_network = nn.Sequential(netG, netF)

    if args.pl.startswith('atdoc_na'):
        mem_fea = torch.rand(len(dset_loaders["target"].dataset), args.bottleneck_dim).cuda()
        mem_fea = mem_fea / torch.norm(mem_fea, p=2, dim=1, keepdim=True)
        mem_cls = torch.ones(len(dset_loaders["target"].dataset), class_num).cuda() / class_num

    if args.pl == 'atdoc_nc':
        mem_fea = torch.rand(args.class_num, args.bottleneck_dim).cuda()
        mem_fea = mem_fea / torch.norm(mem_fea, p=2, dim=1, keepdim=True)
        
    source_loader_iter = iter(dset_loaders["source"])
    target_loader_iter = iter(dset_loaders["target"])

    ####
    list_acc = []
    best_ent = 100

    for iter_num in range(1, args.max_iter + 1):
        base_network.train()
        lr_scheduler(optimizer_g, init_lr=args.lr * 0.1, iter_num=iter_num, max_iter=args.max_iter)
        lr_scheduler(optimizer_f, init_lr=args.lr, iter_num=iter_num, max_iter=args.max_iter)
        if ad_flag:
            lr_scheduler(optimizer_d, init_lr=args.lr, iter_num=iter_num, max_iter=args.max_iter)

        try:
            inputs_source, labels_source = source_loader_iter.next()
        except:
            source_loader_iter = iter(dset_loaders["source"])
            inputs_source, labels_source = source_loader_iter.next()
        try:
            inputs_target, _, idx = target_loader_iter.next()
        except:
            target_loader_iter = iter(dset_loaders["target"])
            inputs_target, _, idx = target_loader_iter.next()
        
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()

        if args.method == 'srconly' and args.pl == 'none':
            features_source, outputs_source = base_network(inputs_source)
        else:
            features_source, outputs_source = base_network(inputs_source)
            features_target, outputs_target = base_network(inputs_target)
            features = torch.cat((features_source, features_target), dim=0)
            outputs = torch.cat((outputs_source, outputs_target), dim=0)
            softmax_out = nn.Softmax(dim=1)(outputs)

        eff = utils.calc_coeff(iter_num, max_iter=args.max_iter)
        if args.method[-1] == 'E':
            entropy = loss.Entropy(softmax_out)
        else:
            entropy = None

        if args.method in {'CDAN', 'CDANE'}:           
            transfer_loss = loss.CDAN([features, softmax_out], ad_net, entropy, eff, random_layer)

        elif args.method in {'DANN', 'DANNE'}:  
            transfer_loss = loss.DANN(features, ad_net, entropy, eff)

        elif args.method == 'DAN':
            transfer_loss = eff * loss.DAN(features_source, features_target)
        elif args.method == 'DAN_Linear':
            transfer_loss = eff * loss.DAN_Linear(features_source, features_target)
        
        elif args.method == 'JAN':
            transfer_loss = eff * loss.JAN([features_source, softmax_out[0:args.batch_size,:]], [features_target, softmax_out[args.batch_size::,:]])
        elif args.method == 'JAN_Linear':
            transfer_loss = eff * loss.JAN_Linear([features_source, softmax_out[0:args.batch_size,:]], [features_target, softmax_out[args.batch_size::,:]])
        
        elif args.method == 'CORAL':
            transfer_loss = eff * loss.CORAL(features_source, features_target)        
        elif args.method == 'DDC':
            transfer_loss = loss.MMD_loss()(features_source, features_target)

        elif args.method == 'srconly':
            transfer_loss = torch.tensor(0.0).cuda()
        else:
            raise ValueError('Method cannot be recognized.')

        src_ = loss.CrossEntropyLabelSmooth(reduction='none',num_classes=class_num, epsilon=args.smooth)(outputs_source, labels_source)
        weight_src = class_weight_src[labels_source].unsqueeze(0)
        classifier_loss = torch.sum(weight_src * src_) / (torch.sum(weight_src).item())
        total_loss = transfer_loss + classifier_loss

        eff = iter_num / args.max_iter

        if args.pl == 'none':
            pass

        elif args.pl == 'square':
            softmax_out = nn.Softmax(dim=1)(outputs_target)
            square_loss = - torch.sqrt((softmax_out**2).sum(dim=1)).mean()
            total_loss += args.tar_par * eff * square_loss

        elif args.pl == 'bsp':
            sigma_loss = bsp_loss(features)
            total_loss += args.tar_par * sigma_loss

        elif args.pl == 'bnm':
            softmax_out = nn.Softmax(dim=1)(outputs_target)
            bnm_loss = -torch.norm(softmax_out, 'nuc') 
            cof = torch.tensor(np.sqrt(np.min(softmax_out.size())) / softmax_out.size(0))
            bnm_loss *= cof 
            total_loss += args.tar_par * eff * bnm_loss

        elif args.pl == "mcc":
            softmax_out = nn.Softmax(dim=1)(outputs_target)
            ent_weight = 1 + torch.exp(-loss.Entropy(softmax_out)).detach()
            ent_weight /= ent_weight.sum()
            cov_tar = softmax_out.t().mm(torch.diag(softmax_out.size(0)*ent_weight)).mm(softmax_out)
            mcc_loss = (torch.diag(cov_tar)/ cov_tar.sum(dim=1)).mean()
            total_loss -= args.tar_par * eff * mcc_loss

        elif args.pl == 'ent':
            softmax_out = nn.Softmax(dim=1)(outputs_target)
            ent_loss = torch.mean(loss.Entropy(softmax_out))
            ent_loss /= torch.log(torch.tensor(class_num+0.0))
            total_loss += args.tar_par * eff * ent_loss

        elif args.pl[0:3] == 'npl':
            softmax_out = nn.Softmax(dim=1)(outputs_target)
            softmax_out = softmax_out**2 / ((softmax_out**2).sum(dim=0))

            weight_, pred = torch.max(softmax_out, 1)
            loss_ = nn.CrossEntropyLoss(reduction='none')(outputs_target, pred)
            classifier_loss = torch.sum(weight_ * loss_) / (torch.sum(weight_).item())
            total_loss += args.tar_par * eff * classifier_loss

        elif args.pl == 'atdoc_nc':
            mem_fea_norm = mem_fea / torch.norm(mem_fea, p=2, dim=1, keepdim=True)
            dis = torch.mm(features_target.detach(), mem_fea_norm.t())
            _, pred = torch.max(dis, dim=1)
            classifier_loss = nn.CrossEntropyLoss()(outputs_target, pred) 
            total_loss += args.tar_par * eff * classifier_loss
        
        elif args.pl.startswith('atdoc_na'):

            dis = -torch.mm(features_target.detach(), mem_fea.t())
            for di in range(dis.size(0)):
                dis[di, idx[di]] = torch.max(dis)
            _, p1 = torch.sort(dis, dim=1)

            w = torch.zeros(features_target.size(0), mem_fea.size(0)).cuda()
            for wi in range(w.size(0)):
                for wj in range(args.K):
                    w[wi][p1[wi, wj]] = 1/ args.K

            weight_, pred = torch.max(w.mm(mem_cls), 1)

            if args.pl == 'atdoc_na_now':
                classifier_loss = nn.CrossEntropyLoss()(outputs_target, pred) 
            else:
                loss_ = nn.CrossEntropyLoss(reduction='none')(outputs_target, pred)
                classifier_loss = torch.sum(weight_ * loss_) / (torch.sum(weight_).item())   
            total_loss += args.tar_par * eff * classifier_loss

        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        if ad_flag:
            optimizer_d.zero_grad()
        total_loss.backward()
        optimizer_g.step()
        optimizer_f.step()
        if ad_flag:
            optimizer_d.step()

        if args.pl.startswith('atdoc_na'):
            base_network.eval() 
            with torch.no_grad():
                features_target, outputs_target = base_network(inputs_target)
                features_target = features_target / torch.norm(features_target, p=2, dim=1, keepdim=True)
                softmax_out = nn.Softmax(dim=1)(outputs_target)
                if args.pl == 'atdoc_na_nos':
                    outputs_target = softmax_out
                else:
                    outputs_target = softmax_out**2 / ((softmax_out**2).sum(dim=0))

            mem_fea[idx] = (1.0 - args.momentum) * mem_fea[idx] + args.momentum * features_target.clone()
            mem_cls[idx] = (1.0 - args.momentum) * mem_cls[idx] + args.momentum * outputs_target.clone()

        if args.pl == 'atdoc_nc':
            base_network.eval() 
            with torch.no_grad():
                features_target, outputs_target = base_network(inputs_target)
                softmax_t = nn.Softmax(dim=1)(outputs_target)
                _, pred_t = torch.max(softmax_t, 1)
                onehot_t = torch.eye(args.class_num)[pred_t].cuda()
                center_t = torch.mm(features_target.t(), onehot_t) / (onehot_t.sum(dim=0) + 1e-8)

            mem_fea = (1.0 - args.momentum) * mem_fea + args.momentum * center_t.t().clone()

        if iter_num % int(args.eval_epoch*max_len) == 0:
            base_network.eval()
            if args.dset == 'VISDA-C':
                acc, py, score, y, tacc = utils.cal_acc_visda(dset_loaders["test"], base_network)
                args.out_file.write(tacc + '\n')
                args.out_file.flush()

                _ent = loss.Entropy(score)
                mean_ent = 0
                for ci in range(args.class_num):
                    mean_ent += _ent[py==ci].mean()
                mean_ent /= args.class_num

            else:
                acc, py, score, y = utils.cal_acc(dset_loaders["test"], base_network)
                mean_ent = torch.mean(loss.Entropy(score))

            list_acc.append(acc * 100)
            if best_ent > mean_ent:
                best_ent = mean_ent
                val_acc = acc * 100
                best_y = y
                best_py = py
                best_score = score

            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%; Mean Ent = {:.4f}'.format(args.name, iter_num, args.max_iter, acc*100, mean_ent)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')            

    idx = np.argmax(np.array(list_acc))
    max_acc = list_acc[idx]
    final_acc = list_acc[-1]

    log_str = '\n==========================================\n'
    log_str += '\nVal Acc = {:.2f}\nMax Acc = {:.2f}\nFin Acc = {:.2f}\n'.format(val_acc, max_acc, final_acc)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()

    # torch.save(base_network.state_dict(), osp.join(args.output_dir, args.log + ".pt"))
    # sio.savemat(osp.join(args.output_dir, args.log + ".mat"), {'y':best_y.cpu().numpy(), 
    #     'py':best_py.cpu().numpy(), 'score':best_score.cpu().numpy()})

    return best_y.cpu().numpy().astype(np.int64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain Adaptation Methods')
    parser.add_argument('--method', type=str, default='srconly', choices=['srconly', 'CDAN', 'CDANE', 'DANN',
     'DANNE', 'JAN_Linear', 'JAN', 'DAN_Linear', 'DAN', 'CORAL', 'DDC'])
    parser.add_argument('--pl', type=str, default='none', choices=['none', 'square', 'npl', 'bnm', 'mcc', 'ent', 'bsp',
        'atdoc_na', 'atdoc_nc', 'atdoc_na_now', 'atdoc_na_nos'])
    # atdoc_na_now: atdoc_na without instance weights
    # atdoc_na_nos: atdoc_na without predictions sharpening

    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--batch_size', type=int, default=36, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--bottleneck_dim', type=int, default=256)

    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--momentum', type=float, default=1.0)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--tar_par', type=float, default=1.0)
    parser.add_argument('--validate', type=bool, default=False)
    
    parser.add_argument('--net', type=str, default='resnet50', choices=["resnet50", "resnet101"])
    parser.add_argument('--dset', type=str, default='office-home', choices=['DomainNet126', 'VISDA-C', 'office', 'office-home'], help="The dataset or source dataset used")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")

    args = parser.parse_args()
    args.output = args.output.strip()

    if args.pl.startswith('atdoc_na'):
        args.pl += str(args.K) 
        args.momentum = 1.0
    if args.pl == 'atdoc_nc':
        args.momentum = 0.1

    args.eval_epoch = args.max_epoch / 10

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'DomainNet126':
        names = ['clipart', 'painting', 'real', 'sketch']
        args.class_num = 126
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    args.s_dset_path = './data/' + args.dset + '/' + names[args.s] + '_list.txt'
    args.t_dset_path = './data/' + args.dset + '/' + names[args.t] + '_list.txt'
    args.test_dset_path = args.t_dset_path

    if args.pl == 'none':
        args.output_dir = osp.join(args.output, args.pl, args.dset, 
            names[args.s][0].upper() + names[args.t][0].upper())
    else:
        args.output_dir = osp.join(args.output, args.pl + '_' + str(args.tar_par), args.dset, 
            names[args.s][0].upper() + names[args.t][0].upper())

    args.name = names[args.s][0].upper() + names[args.t][0].upper()
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.log = args.method
    args.out_file = open(osp.join(args.output_dir, "{:}.txt".format(args.log)), "w")

    utils.print_args(args)
    label = train(args)
    if args.validate:
        train(args, validate=True, label=label)