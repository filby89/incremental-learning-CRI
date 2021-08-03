import argparse
import collections
import torch
import numpy as np
import model.loss as module_loss
import model.metric as module_metric
from parse_config import ConfigParser
from tsn_pytorch.transforms import *
from logger import setup_logging
from model import loss
from BabyAction.dataset import TSNDataSet
from trainer.trainer_action import Trainer
from BabyAction.models import TSN
import pandas as pd
from pathlib import Path

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(args, config):
    # setup data_loader instances
    # data_loader = config.init_obj('data_loader', module_data, 'train')

    # valid_data_loader = config.init_obj('data_loader', module_data, 'val')
    # args = config
    if args.modality == 'RGB':
        data_length = 1
    elif args.modality == "depth":
        data_length = 5
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    # CROSS VALIDATION
    from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, LeaveOneGroupOut

    df = pd.read_csv("BabyAction/all_videos.csv")

    df['subject'] = [x.split("_")[0] for x in df['video']]

    logo = LeaveOneGroupOut().split(df['video'], df['label'], df['subject'])

    log_dir = str(config._log_dir) + "_split_{it}"
    save_dir = str(config._log_dir) + "_split_{it}"

    results = []
    results_25 = []
    results_10 = []
    results_nseg = []
    result_times = []
    for it, (train_idx, test_idx) in enumerate(logo):
      np.save("BabyAction/%02d_train_idx"%(it+1), train_idx)
      np.save("BabyAction/%02d_test_idx"%(it+1), test_idx)


      from sklearn.model_selection import train_test_split
      tr_idx, val_idx = train_test_split(train_idx, test_size=0.1)
      print(len(tr_idx), len(val_idx), len(test_idx))


      config._log_dir = Path(log_dir.format(it=it+1))
      config._save_dir = Path(save_dir.format(it=it+1))

      model = TSN(13, args.num_segments, args.modality, num_feats=args.num_feats,
                  base_model=args.arch, new_length=data_length, 
                  consensus_type=args.consensus_type, dropout=args.dropout, partial_bn=not args.no_partialbn)

      crop_size = model.crop_size
      scale_size = model.scale_size
      input_mean = model.input_mean
      input_std = model.input_std
      policies = model.get_optim_policies()
      train_augmentation = model.get_augmentation()

      # Data loading code
      if args.modality != 'RGBDiff':
          normalize = GroupNormalize(input_mean, input_std)
      else:
          normalize = IdentityTransform()


      dataset = TSNDataSet(df.iloc[tr_idx], num_segments=args.num_segments,
                     new_length=data_length,        
                     modality=args.modality,
                     image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff", "depth"] else args.flow_prefix+"{}_{:05d}.jpg",
                     transform=torchvision.transforms.Compose([
                     GroupScale((256)),
                     GroupRandomHorizontalFlip(),
                     GroupRandomCrop(224),
                         Stack(roll=args.arch == 'BNInception'),
                         ToTorchFormatTensor(div=args.arch != 'BNInception'),
                         normalize,
                     ]))

      def collate(batch):
          images, embedding, categorical, continuous, path = zip(*batch)
          lengths = []
          for image in images:
              lengths.append(image.size(0)//(args.num_segments*3))

          images = torch.cat(images,dim=0)
          categorical = torch.cat(categorical,dim=0)
          continuous = torch.cat(continuous,dim=0)

          return images, embedding, categorical, continuous, lengths


      collate_fn = None
      sampler = None
      

      train_loader = torch.utils.data.DataLoader(
          dataset,
          sampler=sampler,
          batch_size=args.batch_size, shuffle=True,
          num_workers=args.workers, pin_memory=True, collate_fn=collate_fn, drop_last=False)

      val_loader = torch.utils.data.DataLoader(
          TSNDataSet(df.iloc[val_idx], num_segments=args.num_segments,
                     new_length=data_length,
                     modality=args.modality,
                      image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff", "depth"] else args.flow_prefix+"{}_{:05d}.jpg",
                     random_shift=False,
                     transform=torchvision.transforms.Compose([
                         GroupScale((int(256))),
                         GroupCenterCrop(crop_size),
                         Stack(roll=args.arch == 'BNInception'),
                         ToTorchFormatTensor(div=args.arch != 'BNInception'),
                         normalize,
                     ])),
          batch_size=args.batch_size, shuffle=False,
          num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)


      logger = config.get_logger('train')
      logger.info(model)

      criterion_categorical = getattr(module_loss, config['loss'])
      criterion_continuous = getattr(module_loss, config['loss_continuous'])

      metrics = [getattr(module_metric, met) for met in config['metrics']]
      metrics_continuous = [getattr(module_metric, met) for met in config['metrics_continuous']]

      optimizer = torch.optim.SGD(policies,
                                  args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)

      lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)


      for param_group in optimizer.param_groups:
          print(param_group['lr'])
      trainer = Trainer(model, criterion_categorical, metrics, optimizer,
                        config=config,
                        data_loader=train_loader,
                        valid_data_loader=val_loader,
                        lr_scheduler=lr_scheduler)

      result = trainer.train()
      result_times.append(result)

      # for n_segments in [args.num_segments,10]:
      n_segments = args.num_segments
      test_loader = torch.utils.data.DataLoader(
          TSNDataSet(df.iloc[test_idx], num_segments=n_segments,
                     new_length=data_length,
                     modality=args.modality,

                      image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff", "depth"] else args.flow_prefix+"{}_{:05d}.jpg",
                     test_mode=True,
                     transform=torchvision.transforms.Compose([
                         GroupScale((int(256))),
                         GroupCenterCrop(crop_size),
                         Stack(roll=args.arch == 'BNInception'),
                         ToTorchFormatTensor(div=args.arch != 'BNInception'),
                         normalize,
                     ])),
          batch_size=args.batch_size, shuffle=False,
          num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
      model.num_segments = n_segments

      cp = torch.load(str(trainer.checkpoint_dir / 'model_best.pth'))

      model.load_state_dict(cp['state_dict'],strict=True)
      print('loaded', str(trainer.checkpoint_dir / 'model_best.pth'), 'best_epoch', cp['epoch'])

      trainer = Trainer(model, criterion_categorical, metrics, optimizer,
                        config=config,
                        data_loader=train_loader,
                        valid_data_loader=test_loader,
                        lr_scheduler=lr_scheduler)
      result = trainer.test()
      result.update(**{f'test_{n_segments}_' + k: v for k, v in result.items()})
      results.append(result)

      # # ----------- 10 -------------- #
      # n_segments = 10
      # test_loader = torch.utils.data.DataLoader(
      #     TSNDataSet(df.iloc[test_idx], num_segments=n_segments,
      #                new_length=data_length,
      #                modality=args.modality,

      #                 image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff", "depth"] else args.flow_prefix+"{}_{:05d}.jpg",
      #                test_mode=True,
      #                transform=torchvision.transforms.Compose([
      #                    GroupScale((int(256))),
      #                    GroupCenterCrop(crop_size),
      #                    Stack(roll=args.arch == 'BNInception'),
      #                    ToTorchFormatTensor(div=args.arch != 'BNInception'),
      #                    normalize,
      #                ])),
      #     batch_size=args.batch_size, shuffle=False,
      #     num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
      # model.num_segments = n_segments

      # cp = torch.load(str(trainer.checkpoint_dir / 'model_best.pth'))

      # model.load_state_dict(cp['state_dict'],strict=True)
      # print('loaded', str(trainer.checkpoint_dir / 'model_best.pth'), 'best_epoch', cp['epoch'])

      # trainer = Trainer(model, criterion_categorical, metrics, optimizer,
      #                   config=config,
      #                   data_loader=train_loader,
      #                   valid_data_loader=test_loader,
      #                   lr_scheduler=lr_scheduler)
      # result = trainer.test()
      # result.update(**{f'test_{n_segments}_' + k: v for k, v in result.items()})
      # results_10.append(result)

      # # ----------- 25 -------------- #
      # n_segments = 25
      # test_loader = torch.utils.data.DataLoader(
      #     TSNDataSet(df.iloc[test_idx], num_segments=n_segments,
      #                new_length=data_length,
      #                modality=args.modality,

      #                 image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff", "depth"] else args.flow_prefix+"{}_{:05d}.jpg",
      #                test_mode=True,
      #                transform=torchvision.transforms.Compose([
      #                    GroupScale((int(256))),
      #                    GroupCenterCrop(crop_size),
      #                    Stack(roll=args.arch == 'BNInception'),
      #                    ToTorchFormatTensor(div=args.arch != 'BNInception'),
      #                    normalize,
      #                ])),
      #     batch_size=args.batch_size, shuffle=False,
      #     num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
      # model.num_segments = n_segments

      # cp = torch.load(str(trainer.checkpoint_dir / 'model_best.pth'))

      # model.load_state_dict(cp['state_dict'],strict=True)
      # print('loaded', str(trainer.checkpoint_dir / 'model_best.pth'), 'best_epoch', cp['epoch'])

      # trainer = Trainer(model, criterion_categorical, metrics, optimizer,
      #                   config=config,
      #                   data_loader=train_loader,
      #                   valid_data_loader=test_loader,
      #                   lr_scheduler=lr_scheduler)
      # result = trainer.test()
      # result.update(**{f'test_{n_segments}_' + k: v for k, v in result.items()})
      # results_25.append(result)


      # results.append(result)

      # accs_25 = np.mean([x['test_25_balanced_accuracy'] for x in results_25])
      # accs_10 = np.mean([x[f'test_10_balanced_accuracy'] for x in results_10])
      accs = np.mean([x[f'test_{args.num_segments}_balanced_accuracy'] for x in results])

      test_times = np.mean([x[f'test_{args.num_segments}_time'] for x in results])
      # test_times_25 = np.mean([x[f'test_25_time'] for x in results_25])
      # test_times_10 = np.mean([x[f'test_10_time'] for x in results_10])


      val_times = np.mean([x['val_time'] for x in result_times])
      train_times = np.mean([x['time'] for x in result_times])
      print("Current means:", accs, val_times, train_times, test_times)
      # print("Current means:", accs, accs_10, accs_25, val_times, train_times, test_times, test_times_25, test_times_10)
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # parser.add_argument('--exp_name', type=str)

    parser.add_argument('--modality', type=str, choices=['RGB', 'Flow', 'RGBDiff', 'depth'])

    # ========================= Model Configs ==========================
    parser.add_argument('--arch', type=str, default="resnet50")
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--num_segments', type=int, default=5)
    parser.add_argument('--consensus_type', type=str, default='avg',
                        choices=['avg', 'max', 'topk', 'identity', 'rnn', 'cnn'])
    parser.add_argument('--k', type=int, default=3)

    parser.add_argument('--modalities_fusion', type=str, default='cat')
    parser.add_argument('--lossembed', type=str, default='mse')



    parser.add_argument('--dropout', '--do', default=0.5, type=float,
                        metavar='DO', help='dropout ratio (default: 0.5)')

    # ========================= Learning Configs ==========================
    # parser.add_argument('--epochs', default=45, type=int, metavar='N',
    #                     help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                        metavar='W', help='gradient norm clipping (default: disabled)')
    parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")
    parser.add_argument('--categorical', default=False, action="store_true")
    parser.add_argument('--continuous', default=False, action="store_true")
    parser.add_argument('--embed', default=False, action="store_true")
    parser.add_argument('--num_feats', default=2048, type=int)

    # ========================= Monitor Configs ==========================
    parser.add_argument('--print-freq', '-p', default=20, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--eval-freq', '-ef', default=5, type=int,
                        metavar='N', help='evaluation frequency (default: 5)')


    # ========================= Runtime Configs ==========================
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--flow_prefix', default="", type=str)



    # print(parser)

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--exp_name'], type=str, target='name'),
    ]
    config = ConfigParser.from_args(parser, options)
    print(config)

    args = parser.parse_args()

    main(args, config)
