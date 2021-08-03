from torch import nn
import torch.nn.functional
import torch
from .ops.basic_ops import ConsensusModule, Identity
# from .transforms import *
from torch.nn.init import normal, constant
from torch.nn import Parameter


class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet18', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8, modalities_fusion='cat', embed=False,
                 crop_num=1, partial_bn=True, categorical=True, continuous=True, num_feats=2048):
        super(TSN, self).__init__()
        self.num_feats = num_feats
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.categorical = categorical
        self.continuous = continuous
        self.threshold = torch.nn.Parameter(torch.Tensor([0.5])) # torch.Tensor([0.5], requires_grad=True)  # threshold
        self.threshold.requires_grad = True
        self.modalities_fusion = modalities_fusion
        self.embed = embed

        self.name_base = base_model
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length

        print(("""
Initializing TSN with base model: {}.
TSN Configurations:
    input_modality:     {}
    num_segments:       {}
    new_length:         {}
    consensus_module:   {}
    dropout_ratio:      {}
        """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout)))
        self.embed = embed

        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class)


        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)

            cp = torch.load("../../kinetics_tsn_flow.pth.tar")
            for k, v in cp.items():
                cp[k] = torch.squeeze(v, dim=0)

            self.base_model.load_state_dict(cp, strict=False)
            print("Loaded! Flow")

            print("Done. Flow model ready...")

        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")
        elif self.modality == "RGB" and self.new_length > 1 and "resnet" in self.name_base:
            self.base_model = self._construct_rgb_model(self.base_model)
            print("Done. RGB with more length model ready.")

        self.consensus = ConsensusModule(consensus_type)
        self.consensus_cont = ConsensusModule(consensus_type)
        self.consensus_embed = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        std = 0.001
        # print(type(self.base_model))
        if isinstance(self.base_model, torch.nn.modules.container.Sequential):
            feature_dim = 2048
        else:
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
            if self.dropout == 0:
                setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
                self.fc = None
            else:
                setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))

        self.fc = nn.Linear(feature_dim, num_class)
        normal(self.fc.weight, 0, std)
        constant(self.fc.bias, 0)


        return feature_dim


    def _prepare_base_model(self, base_model):
        import torchvision, torchvision.models
        if "r3d" in base_model:
            self.base_model = torchvision.models.video.r3d_18(True)
            # self.base_model = self._prepare_context_model()
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

        elif 'resnet' in base_model or 'vgg' in base_model or 'resnext' in base_model or 'densenet' in base_model:

            self.base_model = getattr(torchvision.models, "resnet50")(True)
            modules = list(self.base_model.children())[:-1]  # delete the last fc layer.

            self.base_model = nn.Sequential(*modules)

            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225
                              ]

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif base_model == 'BNInception':
            # from tsn_pytorch import tf_model_zoo
            from .tsn_pytorch import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)()


            # # if modality == "RGB":
            cp = torch.load("../../kinetics_tsn_rgb.pth.tar")
            for k, v in cp.items():
                cp[k] = torch.squeeze(v, dim=0)

            self.base_model.load_state_dict(cp, strict=False)
            print("Loaded!")


            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)

        elif 'inception' in base_model:
            from tsn_pytorch import tf_model_zoo

            self.base_model = getattr(tf_model_zoo, base_model)()
            self.base_model.last_layer_name = 'classif'
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn:
            # print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
           


    def partialBN(self, enable):
        self._enable_pbn = enable


    def forward(self, input):
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
        # sample_len = 1

        if self.modality == "depth":
            sample_len = 1 * self.new_length

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        body = input.view((-1, sample_len) + input.size()[-2:])
        base_out = self.base_model(body).squeeze(-1).squeeze(-1)

        # if self.categorical:
        #     base_out_cat = self.fc(base_out)
        #
        # if self.continuous:
        #     base_out_cont = self.fc_1(base_out)
        #
        #
        # base_out = base_out_cat
        #
        # if self.categorical:
        #     base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
        # if self.continuous:
        #     base_out_cont = base_out_cont.view((-1, self.num_segments) + base_out_cont.size()[1:])
        #
        #
        # if self.categorical:
        #     output = self.consensus(base_out)
        #     outputs['categorical'] = output.squeeze(1)
        # if self.continuous:
        #     output_cont = self.consensus_cont(base_out_cont)
        #     outputs['continuous'] = output_cont.squeeze(1)

        return base_out

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data

    def _construct_rgb_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (3 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(3 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model
