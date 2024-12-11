from collections import namedtuple
from models.ResNet import BasicBlock, Bottleneck

ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels','cardinality', 'base_width'])

ViTConfig = namedtuple('ViTConfig', ['tokens_type', 'embed_dim', 'depth', 'num_heads', 'mlp_ratio'])

vgg11_config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

vgg13_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512,
                512, 'M']

vgg16_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
                'M', 512, 512, 512, 'M']

vgg19_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512,
                512, 512, 'M', 512, 512, 512, 512, 'M']

resnet18_config = ResNetConfig(block = BasicBlock,
                               n_blocks = [2,2,2,2],
                               channels = [64, 128, 256, 512],
                                cardinality = 1,
                                base_width = 64)

resnet34_config = ResNetConfig(block = BasicBlock,
                               n_blocks = [3,4,6,3],
                               channels = [64, 128, 256, 512],
                                cardinality = 1,
                                base_width = 64)

resnet50_config = ResNetConfig(block = Bottleneck,
                               n_blocks = [3, 4, 6, 3],
                               channels = [64, 128, 256, 512],
                                cardinality = 1,
                                base_width = 64)

resnet101_config = ResNetConfig(block = Bottleneck,
                                n_blocks = [3, 4, 23, 3],
                                channels = [64, 128, 256, 512],
                                cardinality = 1,
                                base_width = 64)

resnext50_32x4d_config = ResNetConfig(block = Bottleneck,
                                        n_blocks = [3, 4, 6, 3],
                                        channels = [64, 128, 256, 512],
                                        cardinality = 32,
                                        base_width = 4)

resnext101_32x4d_config = ResNetConfig(block = Bottleneck,
                                        n_blocks = [3, 4, 23, 3],
                                        channels = [64, 128, 256, 512],
                                        cardinality = 32,
                                        base_width = 4)

t2t_vit_t_12_config = ViTConfig(tokens_type = 'transformer',
                                embed_dim = 256,
                                depth = 12,
                                num_heads = 4,
                                mlp_ratio = 2.0)

t2t_vit_t_14_config = ViTConfig(tokens_type = 'transformer',
                                embed_dim = 384,
                                depth = 14,
                                num_heads = 6,
                                mlp_ratio = 3.0)
t2t_vit_14_config = ViTConfig(tokens_type = 'performer',
                                embed_dim = 384,
                                depth = 14,
                                num_heads = 6,
                                mlp_ratio = 3.0)        