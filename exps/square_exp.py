import torch
import torch.nn as nn
import conf
import util

from nets.classic_net import get_resnet18
from exps.trainer import ToyboxTrainer

torch.backends.cudnn.benchmark = True


def main():
    ratio = '100'
    net_name = 'resnet18'
    net = get_resnet18(pretrained=False)
    net.cuda()
    print(f'{sum(x.numel() for x in net.parameters())} parameters in total')
    print(f'{sum(x.numel() for x in net.fc.parameters())} parameters in the last layer')

    optimizer = torch.optim.Adam(net.parameters(), lr=5e-05, weight_decay=0.0)
    loss_func = nn.CrossEntropyLoss()
    hyper_p = util.HyperP(
        lr=0.5,
        batch_size=64,
        num_workers=1,
        epochs=300,
    )
    tb_trainer = ToyboxTrainer(ratio, net, net_name, optimizer, loss_func, hyper_p)
    print(f'=== {tb_trainer.exp_name} ===')
    tb_trainer.train_test_save()


if __name__ == '__main__':
    main()
