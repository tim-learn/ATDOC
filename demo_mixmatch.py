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
import utils, pickle
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
    target_set = utils.ObjectImage_mul('', args.t_dset_path, [train_transform, train_transform])
    test_set = utils.ObjectImage('', args.test_dset_path, test_transform)

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

def train(args):
    ## set pre-process
    dset_loaders = data_load(args)

    max_len = max(len(dset_loaders["source"]), len(dset_loaders["target"]))
    args.max_iter = args.max_epoch*max_len
    
    ## set base network
    if args.net == 'resnet101':
        netG = utils.ResBase101().cuda()
    elif args.net == 'resnet50':
        netG = utils.ResBase50().cuda()  
    
    netF = utils.ResClassifier(class_num=args.class_num, feature_dim=netG.in_features, 
        bottleneck_dim=args.bottleneck_dim).cuda()
    
    if len(args.gpu_id.split(',')) > 1:
        netG = nn.DataParallel(netG)
    
    optimizer_g = optim.SGD(netG.parameters(), lr = args.lr * 0.1)
    optimizer_f = optim.SGD(netF.parameters(), lr = args.lr)

    base_network = nn.Sequential(netG, netF)
    source_loader_iter = iter(dset_loaders["source"])
    target_loader_iter = iter(dset_loaders["target"])

    if args.pl == 'atdoc_na':
        mem_fea = torch.rand(1*len(dset_loaders["target"].dataset), args.bottleneck_dim).cuda()
        mem_fea = mem_fea / torch.norm(mem_fea, p=2, dim=1, keepdim=True)
        mem_cls = torch.ones(1*len(dset_loaders["target"].dataset), args.class_num).cuda() / args.class_num

    if args.pl == 'atdoc_nc':
        mem_fea = torch.rand(args.class_num, args.bottleneck_dim).cuda()
        mem_fea = mem_fea / torch.norm(mem_fea, p=2, dim=1, keepdim=True)

    list_acc = []
    best_ent = 100

    for iter_num in range(1, args.max_iter + 1):
        base_network.train()
        lr_scheduler(optimizer_g, init_lr=args.lr * 0.1, iter_num=iter_num, max_iter=args.max_iter)
        lr_scheduler(optimizer_f, init_lr=args.lr, iter_num=iter_num, max_iter=args.max_iter)

        try:
            inputs_source, labels_source = source_loader_iter.next()
        except:
            source_loader_iter = iter(dset_loaders["source"])
            inputs_source, labels_source = source_loader_iter.next()
        try:
            inputs_target, _, target_idx = target_loader_iter.next()
        except:
            target_loader_iter = iter(dset_loaders["target"])
            inputs_target, _, target_idx = target_loader_iter.next()
        
        targets_s = torch.zeros(args.batch_size, args.class_num).scatter_(1, labels_source.view(-1,1), 1)
        inputs_s = inputs_source.cuda()
        targets_s = targets_s.cuda()
        inputs_t = inputs_target[0].cuda()
        inputs_t2 = inputs_target[1].cuda()

        if args.pl == 'atdoc_na':

            targets_u = 0
            for inp in [inputs_t, inputs_t2]:
                with torch.no_grad():
                    features_target, outputs_u = base_network(inp)

                dis = -torch.mm(features_target.detach(), mem_fea.t())
                for di in range(dis.size(0)):
                    dis[di, target_idx[di]] = torch.max(dis)
                    # dis[di, target_idx[di]+len(dset_loaders["target"].dataset)] = torch.max(dis)

                _, p1 = torch.sort(dis, dim=1)
                w = torch.zeros(features_target.size(0), mem_fea.size(0)).cuda()
                for wi in range(w.size(0)):
                    for wj in range(args.K):
                        w[wi][p1[wi, wj]] = 1/ args.K

                _, pred = torch.max(w.mm(mem_cls), 1)

                targets_u += 0.5*torch.eye(outputs_u.size(1))[pred].cuda()

        elif args.pl == 'atdoc_nc':

            targets_u = 0
            mem_fea_norm = mem_fea / torch.norm(mem_fea, p=2, dim=1, keepdim=True)
            for inp in [inputs_t, inputs_t2]:
                with torch.no_grad():
                    features_target, outputs_u = base_network(inp)
                dis = torch.mm(features_target.detach(), mem_fea_norm.t())
                _, pred = torch.max(dis, dim=1)
                targets_u += 0.5*torch.eye(outputs_u.size(1))[pred].cuda()

        elif args.pl == 'npl':

            targets_u = 0
            for inp in [inputs_t, inputs_t2]:
                with torch.no_grad():
                    _, outputs_u = base_network(inp)
                _, pred = torch.max(outputs_u.detach(), 1)
                targets_u += 0.5*torch.eye(outputs_u.size(1))[pred].cuda()

        else:
            with torch.no_grad():
                # compute guessed labels of unlabel samples
                _, outputs_u = base_network(inputs_t)
                _, outputs_u2 = base_network(inputs_t2)
                p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
                pt = p**(1/args.T)
                targets_u = pt / pt.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()

        ####################################################################
        all_inputs = torch.cat([inputs_s, inputs_t, inputs_t2], dim=0)
        all_targets = torch.cat([targets_s, targets_u, targets_u], dim=0)
        if args.alpha > 0:
            l = np.random.beta(args.alpha, args.alpha)
            l = max(l, 1-l)
        else:
            l = 1
        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
        mixed_input = list(torch.split(mixed_input, args.batch_size))
        mixed_input = utils.interleave(mixed_input, args.batch_size)  
        # s = [sa, sb, sc]
        # t1 = [t1a, t1b, t1c]
        # t2 = [t2a, t2b, t2c]
        # => s' = [sa, t1b, t2c]   t1' = [t1a, sb, t1c]   t2' = [t2a, t2b, sc]

        # _, logits = base_network(mixed_input[0])
        features, logits = base_network(mixed_input[0])
        logits = [logits]
        for input in mixed_input[1:]:
            _, temp = base_network(input)
            logits.append(temp)

        # put interleaved samples back
        # [i[:,0] for i in aa]
        logits = utils.interleave(logits, args.batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        train_criterion = utils.SemiLoss()

        Lx, Lu, w = train_criterion(logits_x, mixed_target[:args.batch_size], logits_u, mixed_target[args.batch_size:], 
            iter_num, args.max_iter, args.lambda_u)
        total_loss = Lx + w * Lu

        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        total_loss.backward()
        optimizer_g.step()
        optimizer_f.step()

        if args.pl == 'atdoc_na':
            base_network.eval()
            with torch.no_grad():
                fea1, outputs1 = base_network(inputs_t)
                fea2, outputs2 = base_network(inputs_t2)
                feat = 0.5 * (fea1 + fea2)
                feat = feat / torch.norm(feat, p=2, dim=1, keepdim=True)
                softmax_out = 0.5*(nn.Softmax(dim=1)(outputs1) + nn.Softmax(dim=1)(outputs2))
                softmax_out = softmax_out**2 / ((softmax_out**2).sum(dim=0))

            mem_fea[target_idx] = (1.0 - args.momentum)*mem_fea[target_idx] + args.momentum*feat
            mem_cls[target_idx] = (1.0 - args.momentum)*mem_cls[target_idx] + args.momentum*softmax_out

        if args.pl == 'atdoc_nc':
            base_network.eval() 
            with torch.no_grad():
                fea1, outputs1 = base_network(inputs_t)
                fea2, outputs2 = base_network(inputs_t2)
                feat = 0.5*(fea1 + fea2)
                softmax_t = 0.5*(nn.Softmax(dim=1)(outputs1) + nn.Softmax(dim=1)(outputs2))
                _, pred_t = torch.max(softmax_t, 1)
                onehot_t = torch.eye(args.class_num)[pred_t].cuda()
                center_t = torch.mm(feat.t(), onehot_t) / (onehot_t.sum(dim=0) + 1e-8)

            mem_fea = (1.0 - args.momentum) * mem_fea + args.momentum * center_t.t().clone()

        if iter_num % int(args.eval_epoch * max_len) == 0:
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
                val_acc = acc * 100
                best_ent = mean_ent
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
    
    return base_network, py

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mixmatch for Domain Adaptation')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=36, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--bottleneck_dim', type=int, default=256)

    parser.add_argument('--net', type=str, default='resnet50', choices=["resnet50", "resnet101"])
    parser.add_argument('--dset', type=str, default='office-home', choices=['DomainNet126', 'VISDA-C', 'office', 'office-home'], help="The dataset or source dataset used")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--pl', type=str, default='none', choices=['none', 'npl', 'atdoc_na', 'atdoc_nc'])
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--momentum', type=float, default=1.0)

    parser.add_argument('--alpha', default=0.75, type=float)
    parser.add_argument('--lambda_u', default=100, type=float)
    parser.add_argument('--T', default=0.5, type=float)
    parser.add_argument('--ema_decay', default=0.999, type=float)

    args = parser.parse_args()
    if args.pl == 'atdoc_na':
        args.pl += args.pl + str(args.K) 
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

    args.output_dir = osp.join(args.output, 'mixmatch', args.dset, 
        names[args.s][0].upper() + names[args.t][0].upper())

    args.name = names[args.s][0].upper() + names[args.t][0].upper()
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.log = 'mixmatch_' + args.pl
    args.out_file = open(osp.join(args.output_dir, "{:}.txt".format(args.log)), "w")

    utils.print_args(args)

    train(args)