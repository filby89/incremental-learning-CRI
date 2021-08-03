import os
import time
import torch
import argparse
import importlib
import numpy as np
from functools import reduce

import utils
import approach
from loggers.exp_logger import MultiLogger
from datasets.data_loader import get_loaders, get_loaders_baby
from datasets.dataset_config import dataset_config
from last_layer_analysis import last_layer_analysis
from networks import tvmodels, allmodels, set_tvmodel_head_var
from networks.models import TSN
import numpy as np
import torchvision
from tsn_pytorch.transforms import *

def main(split, seed, argv=None):
    tstart = time.time()
    # Arguments
    parser = argparse.ArgumentParser(description='FACIL - Framework for Analysis of Class Incremental Learning')

    # miscellaneous args
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU (default=%(default)s)')
    parser.add_argument('--results-path', type=str, default='../results',
                        help='Results path (default=%(default)s)')
    parser.add_argument('--exp-name', default=None, type=str,
                        help='Experiment name (default=%(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default=%(default)s)')
    parser.add_argument('--log', default=['disk'], type=str, choices=['disk', 'tensorboard'],
                        help='Loggers used (disk, tensorboard) (default=%(default)s)', nargs='*', metavar="LOGGER")
    parser.add_argument('--save-models', action='store_true',
                        help='Save trained models (default=%(default)s)')
    parser.add_argument('--last-layer-analysis', action='store_true',
                        help='Plot last layer analysis (default=%(default)s)')
    parser.add_argument('--no-cudnn-deterministic', action='store_true',
                        help='Disable CUDNN deterministic (default=%(default)s)')
    # dataset args
    parser.add_argument('--datasets', default=['babyaction'], type=str, choices=list(dataset_config.keys()),
                        help='Dataset or datasets used (default=%(default)s)', nargs='+', metavar="DATASET")
    parser.add_argument('--num-workers', default=4, type=int, required=False,
                        help='Number of subprocesses to use for dataloader (default=%(default)s)')
    parser.add_argument('--pin-memory', default=False, type=bool, required=False,
                        help='Copy Tensors into CUDA pinned memory before returning them (default=%(default)s)')
    parser.add_argument('--batch-size', default=8, type=int, required=False,
                        help='Number of samples per batch to load (default=%(default)s)')
    parser.add_argument('--num-tasks', default=4, type=int, required=False,
                        help='Number of tasks per dataset (default=%(default)s)')
    parser.add_argument('--nc-first-task', default=None, type=int, required=False,
                        help='Number of classes of the first task (default=%(default)s)')
    parser.add_argument('--use-valid-only', action='store_true',
                        help='Use validation split instead of test (default=%(default)s)')
    parser.add_argument('--stop-at-task', default=0, type=int, required=False,
                        help='Stop training after specified task (default=%(default)s)')
    # model args
    parser.add_argument('--network', default='resnet32', type=str, choices=allmodels,
                        help='Network architecture used (default=%(default)s)', metavar="NETWORK")
    parser.add_argument('--keep-existing-head', action='store_true',
                        help='Disable removing classifier last layer (default=%(default)s)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained backbone (default=%(default)s)')
    # training args
    parser.add_argument('--approach', default='icarl', type=str, choices=approach.__all__,
                        help='Learning approach used (default=%(default)s)', metavar="APPROACH")
    parser.add_argument('--nepochs', default=1, type=int, required=False,
                        help='Number of epochs per training session (default=%(default)s)')
    parser.add_argument('--lr', default=1e-4, type=float, required=False,
                        help='Starting learning rate (default=%(default)s)')
    parser.add_argument('--lr-min', default=1e-7, type=float, required=False,
                        help='Minimum learning rate (default=%(default)s)')
    parser.add_argument('--lr-factor', default=10, type=float, required=False,
                        help='Learning rate decreasing factor (default=%(default)s)')
    parser.add_argument('--lr-patience', default=10, type=int, required=False,
                        help='Maximum patience to wait before decreasing learning rate (default=%(default)s)')
    parser.add_argument('--clipping', default=10000, type=float, required=False,
                        help='Clip gradient norm (default=%(default)s)')
    parser.add_argument('--momentum', default=0.9, type=float, required=False,
                        help='Momentum factor (default=%(default)s)')
    parser.add_argument('--weight-decay', default=0.0, type=float, required=False,
                        help='Weight decay (L2 penalty) (default=%(default)s)')
    parser.add_argument('--warmup-nepochs', default=0, type=int, required=False,
                        help='Number of warm-up epochs (default=%(default)s)')
    parser.add_argument('--warmup-lr-factor', default=1.0, type=float, required=False,
                        help='Warm-up learning rate factor (default=%(default)s)')
    parser.add_argument('--multi-softmax', action='store_true',
                        help='Apply separate softmax for each task (default=%(default)s)')
    parser.add_argument('--fix-bn', action='store_true',
                        help='Fix batch normalization after first task (default=%(default)s)')
    parser.add_argument('--eval-on-train', action='store_true',
                        help='Show train loss and accuracy (default=%(default)s)')
    parser.add_argument('--num-classes', default=20, type=int, help='Show train loss and accuracy (default=%(default)s)')

    # gridsearch args
    parser.add_argument('--gridsearch-tasks', default=0, type=int,
                        help='Number of tasks to apply GridSearch (-1: all tasks) (default=%(default)s)')

    # Args -- Incremental Learning Framework
    args, extra_args = parser.parse_known_args(argv)
    args.results_path = os.path.expanduser(args.results_path)
    base_kwargs = dict(nepochs=args.nepochs, lr=args.lr, lr_min=args.lr_min, lr_factor=args.lr_factor,
                       lr_patience=args.lr_patience, clipgrad=args.clipping, momentum=args.momentum,
                       wd=args.weight_decay, multi_softmax=args.multi_softmax, wu_nepochs=args.warmup_nepochs,
                       wu_lr_factor=args.warmup_lr_factor, fix_bn=args.fix_bn, eval_on_train=args.eval_on_train)

    args.seed = seed
    print('Seed is', args.seed)

    if args.no_cudnn_deterministic:
        print('WARNING: CUDNN Deterministic will be disabled.')
        utils.cudnn_deterministic = False

    utils.seed_everything(seed=args.seed)
    print('=' * 108)
    print('Arguments =')
    for arg in np.sort(list(vars(args).keys())):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 108)

    # Args -- CUDA
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = 'cuda'
    else:
        print('WARNING: [CUDA unavailable] Using CPU instead!')
        device = 'cpu'
    # Multiple gpus
    # if torch.cuda.device_count() > 1:
    #     self.C = torch.nn.DataParallel(C)
    #     self.C.to(self.device)
    ####################################################################################################################

    # Args -- Continual Learning Approach
    from approach.incremental_learning import Inc_Learning_Appr
    Appr = getattr(importlib.import_module(name='approach.' + args.approach), 'Appr')
    assert issubclass(Appr, Inc_Learning_Appr)
    appr_args, extra_args = Appr.extra_parser(extra_args)
    print('Approach arguments =')
    for arg in np.sort(list(vars(appr_args).keys())):
        print('\t' + arg + ':', getattr(appr_args, arg))
    print('=' * 108)

    # Args -- Exemplars Management
    from datasets.exemplars_dataset import ExemplarsDataset, ExemplarsDatasetTSN
    Appr_ExemplarsDataset = Appr.exemplars_dataset_class()
    if Appr_ExemplarsDataset:
        Appr_ExemplarsDataset = ExemplarsDatasetTSN
        # assert issubclass(Appr_ExemplarsDataset, ExemplarsDatasetTSN)
        appr_exemplars_dataset_args, extra_args = Appr_ExemplarsDataset.extra_parser(extra_args)
        print('Exemplars dataset arguments =')
        for arg in np.sort(list(vars(appr_exemplars_dataset_args).keys())):
            print('\t' + arg + ':', getattr(appr_exemplars_dataset_args, arg))
        print('=' * 108)
    else:
        appr_exemplars_dataset_args = argparse.Namespace()

    ####################################################################################################################

    # Log all arguments
    # full_exp_name = reduce((lambda x, y: x[0] + y[0]), args.datasets) if len(args.datasets) > 0 else args.datasets[0]
    # full_exp_name += '_' + args.approach

    full_exp_name = f"{args.approach}_exemplars-{appr_exemplars_dataset_args.num_exemplars}_exemplars_per_class-{appr_exemplars_dataset_args.num_exemplars_per_class}_tasks-{args.num_tasks}_epochs-{args.nepochs}_{args.num_classes}-classes_1split_re_MSLR_TRUE-60epochs"

    if args.exp_name is not None:
        full_exp_name += '_' + args.exp_name

    logger = MultiLogger(args.results_path, full_exp_name, loggers=args.log, save_models=args.save_models)
    logger.log_args(argparse.Namespace(**args.__dict__, **appr_args.__dict__, **appr_exemplars_dataset_args.__dict__))

    # Loaders
    utils.seed_everything(seed=args.seed)

    trn_loader, val_loader, tst_loader, taskcla = get_loaders_baby('BabyAction', args.num_tasks, args.nc_first_task,
                                                              args.batch_size, num_workers=args.num_workers,
                                                              pin_memory=args.pin_memory, validation=0, split=split, num_classes=args.num_classes)
    utils.seed_everything(seed=args.seed)

    # Apply arguments for loaders
    # if args.use_valid_only:
        # tst_loader = val_loader
    # val_loader = tst_loader

    max_task = len(taskcla) if args.stop_at_task == 0 else args.stop_at_task

    outputs = {}
    targets = {}

    acc_taw = np.zeros((3, max_task, max_task))
    acc_tag = np.zeros((3, max_task, max_task))
    forg_taw = np.zeros((3, max_task, max_task))
    forg_tag = np.zeros((3, max_task, max_task))

    times = np.zeros((3, max_task))

    modalities = ["Flow"]
    for m, modality in enumerate(modalities):
        # Args -- Network
        from networks.network import LLL_Net

        arch = 'BNInception'

        num_segments = 5

        if modality == 'RGB':
            data_length = 1
            input_mean = [0.485, 0.456, 0.406]
            input_std = [0.229, 0.224, 0.225]
        else:
            input_std = [0.229, 0.224, 0.225]
            input_mean = [0.5]
            input_std = [np.mean(input_std)]
            data_length = 5

        utils.seed_everything(seed=args.seed)

        init_model = TSN(12, num_segments, modality, num_feats=2048,
                  base_model=arch, new_length=data_length,
                  consensus_type='avg', dropout=0.5, partial_bn=True, categorical=True, continuous=False)
        init_model.head_var = 'fc'

        # Network and Approach instances
        utils.seed_everything(seed=args.seed)
        net = LLL_Net(init_model, remove_existing_head=not args.keep_existing_head)
        utils.seed_everything(seed=args.seed)


        # taking transformations and class indices from first train dataset
        first_train_ds = trn_loader[0].dataset
        transform, class_indices = first_train_ds.transform, first_train_ds.class_indices
        print('Class order', class_indices)
        appr_kwargs = {**base_kwargs, **dict(logger=logger, **appr_args.__dict__)}

        # ------- Exemplar ------ #
        arch = 'BNInception'

        utils.seed_everything(seed=args.seed)

        trn_transform = torchvision.transforms.Compose([
            GroupScale((256, 256)),
            GroupRandomHorizontalFlip(),
            GroupRandomCrop(224),
            Stack(roll=arch == 'BNInception'),
            ToTorchFormatTensor(div=arch != 'BNInception'),
            GroupNormalize(input_mean, input_std)
        ])

        utils.seed_everything(seed=args.seed)

        if Appr_ExemplarsDataset:
            appr_kwargs['exemplars_dataset'] = Appr_ExemplarsDataset(trn_transform, class_indices, num_segments=num_segments,
                                                                     new_length=data_length, modality=modality, image_tmpl='{}_{:05d}.jpg',
                                                                     **appr_exemplars_dataset_args.__dict__)
        utils.seed_everything(seed=args.seed)
        appr = Appr(net, device, **appr_kwargs)

        utils.seed_everything(seed=args.seed)

        # Loop tasks
        print(taskcla)

        for t, (_, ncla) in enumerate(taskcla):
            # Early stop tasks if flag
            if t >= max_task:
                continue

            print('*' * 108)
            print('Task {:2d}'.format(t))
            print('*' * 108)

            # Add head for current task
            net.add_head(taskcla[t][1])
            net.to(device)
            #

            # change modality for loaders
            trn_loader[t].dataset.modality=modality
            val_loader[t].dataset.modality=modality
            tst_loader[t].dataset.modality=modality

            trn_loader[t].dataset.new_length=data_length
            val_loader[t].dataset.new_length=data_length
            tst_loader[t].dataset.new_length=data_length

            print("Train dataset:",len(trn_loader[t].dataset))
            print("Val dataset:",len(val_loader[t].dataset))
            print("Test dataset:",len(tst_loader[t].dataset))

            # Train
            s = time.time()
            appr.train(t, trn_loader[t], val_loader[t])
            print('-' * 108)
            times[m, t] = time.time()-s

            # Test
            for u in range(t + 1):
                net.model.num_segments=10
                test_loss, acc_taw[m, t, u], acc_tag[m, t, u], o, tar = appr.eval(u, tst_loader[u], save_outputs=True)
                key = "{}{}{}".format(m,t,u)
                outputs[key] = o.detach()
                targets[key] = tar.detach()

                net.model.num_segments=5

                if u < t:
                    forg_taw[m, t, u] = acc_taw[m, :t, u].max(0) - acc_taw[m, t, u]
                    forg_tag[m, t, u] = acc_tag[m, :t, u].max(0) - acc_tag[m, t, u]
                print('>>> Test on task {:2d} : loss={:.3f} | TAw acc={:5.1f}%, forg={:5.1f}%'
                      '| TAg acc={:5.1f}%, forg={:5.1f}% <<<'.format(u, test_loss,
                                                                     100 * acc_taw[m, t, u], 100 * forg_taw[m, t, u],
                                                                     100 * acc_tag[m, t, u], 100 * forg_tag[m, t, u]))
    utils.print_summary(acc_taw[m], acc_tag[m], forg_taw[m], forg_tag[m])


    if len(modalities) > 1:
        # finally do fusion
        for t, (_, ncla) in enumerate(taskcla):
            for u in range(t + 1):
                m = 2
                # o = (torch.nn.functional.softmax(outputs["{}{}{}".format(0, t, u)].cpu(),dim=1) + 3*torch.nn.functional.softmax(outputs["{}{}{}".format(1, t, u)].cpu(),dim=1))/2
                o = (outputs["{}{}{}".format(0, t, u)].cpu() + 3*outputs["{}{}{}".format(1, t, u)].cpu())/2
                tar = targets["{}{}{}".format(0, t, u)].cpu()
                acc_taw[m, t, u], acc_tag[m, t, u] = appr.eval_met(o,tar)
                if u < t:
                    forg_taw[m, t, u] = acc_taw[m, :t, u].max(0) - acc_taw[m, t, u]
                    forg_tag[m, t, u] = acc_tag[m, :t, u].max(0) - acc_tag[m, t, u]
                print('>>> Test on task {:2d} | TAw acc={:5.1f}%, forg={:5.1f}%'
                      '| TAg acc={:5.1f}%, forg={:5.1f}% <<<'.format(u,
                                                                     100 * acc_taw[m, t, u], 100 * forg_taw[m, t, u],
                                                                     100 * acc_tag[m, t, u], 100 * forg_tag[m, t, u]))

        for m in range(3):
            # Print Summary
            if m == 0:
                print('=' * 108)
                print('RGB')
            if m == 1:
                print('=' * 108)
                print('Flow')
            if m == 2:
                print('=' * 108)
                print('Fusion')

            utils.print_summary(acc_taw[m], acc_tag[m], forg_taw[m], forg_tag[m])


    print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
    print('Done!')

    return acc_taw, acc_tag, forg_taw, forg_tag, times, logger.exp_path, outputs, targets
    ####################################################################################################################


if __name__ == '__main__':
    acc_taw_split = []
    acc_tag_split = []
    forg_taw_split = []
    forg_tag_split = []
    times_split = []

    # seeds = [4852, 2919, 4843, 8480, 5474]
    # seeds = [8480, 5474]
    # seeds = [1571, 3217, 9199, 2941, 8835, 4852, 2919, 4843, 8480, 5474]
    seeds = [4852, 2919, 4843, 8480, 5474]

    import pickle
    import sys
    for seed in seeds:
        print("Seed:", seed)

        for split in range(1):
            acc_taw, acc_tag, forg_taw, forg_tag, times, exp_path, outputs, targets = main(split,seed)
            acc_taw_split.append(acc_taw)
            acc_tag_split.append(acc_tag)
            forg_taw_split.append(forg_taw)
            forg_tag_split.append(forg_tag)
            times_split.append(times)
            np.save(f'{exp_path}/acc_taw_split_{split}_{seed}.npy',acc_taw)
            np.save(f'{exp_path}/acc_tag_split_{split}_{seed}.npy',acc_tag)
            np.save(f'{exp_path}/forg_taw_split_{split}_{seed}.npy',forg_taw)
            np.save(f'{exp_path}/forg_tag_split_{split}_{seed}.npy',forg_tag)
            np.save(f'{exp_path}/times_split_{split}_{seed}.npy',times)

            pickle.dump( outputs, open( f'{exp_path}/outputs_split_{split}_{seed}.pkl', "wb" ) )
            pickle.dump( targets, open( f'{exp_path}/targets_split_{split}_{seed}.pkl', "wb" ) )

    acc_taw_split = np.stack(acc_taw_split,axis=0)
    acc_tag_split = np.stack(acc_tag_split,axis=0)
    forg_taw_split = np.stack(forg_taw_split,axis=0)
    forg_tag_split = np.stack(forg_tag_split,axis=0)

    times_split = np.stack(times_split,axis=0)

    # times_split = np.sum(times_split,axis=0)

    np.save(f'{exp_path}/acc_taw_split_all.npy',acc_taw_split)
    np.save(f'{exp_path}/acc_tag_split_all.npy',acc_tag_split)
    np.save(f'{exp_path}/forg_taw_split_all.npy',forg_taw_split)
    np.save(f'{exp_path}/forg_tag_split_all.npy',forg_tag_split)
    np.save(f'{exp_path}/times_split_all.npy',times_split)

    acc_taw_split = np.mean(acc_taw_split,axis=0)
    acc_tag_split = np.mean(acc_tag_split,axis=0)
    forg_taw_split = np.mean(forg_taw_split,axis=0)
    forg_tag_split = np.mean(forg_tag_split,axis=0)

    for m in range(3):
        # Print Summary
        if m == 0:
            print('=' * 108)
            print('RGB')
        if m == 1:
            print('=' * 108)
            print('Flow')
        if m == 2:
            print('=' * 108)
            print('Fusion')

        utils.print_summary(acc_taw_split[m], acc_tag_split[m], forg_taw_split[m], forg_tag_split[m])
        print(times_split)