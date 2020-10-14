import torch
import torch.nn as nn
import conf
import util

from nets.mvcnn_net import SVCNN, MVCNN
from exps.trainer import ToyboxTrainer

torch.backends.cudnn.benchmark = True


def exp_main(ratio):
    # STAGE 1
    print('=== Stage 1 ===')
    net_name = 'svcnn'
    net = SVCNN(net_name, nclasses=12, pretraining=False, cnn_name='resnet18')
    net.cuda()
    print(f'{sum(x.numel() for x in net.net.parameters())} parameters in total')
    print(f'{sum(x.numel() for x in net.net.fc.parameters())} parameters in the last layer')

    optimizer = torch.optim.Adam(net.parameters(), lr=5e-05, weight_decay=0.0)
    loss_func = nn.CrossEntropyLoss()
    hyper_p = util.HyperP(
        lr=5e-05,
        batch_size=64,
        num_workers=1,
        epochs=300,
    )
    tb_trainer = ToyboxTrainer(
        tr=['rzplus', 'rzminus'],
        nview=12,
        ratio=[ratio],
        mode='sv',
        net=net,
        net_name=net_name,
        optimizer=optimizer,
        loss_func=loss_func,
        hyper_p=hyper_p
    )
    print(f'=== {tb_trainer.exp_name} ===')
    tb_trainer.train_test_save()

    # STAGE 2
    print('=== Stage 2 ===')
    net_name = 'mvcnn'
    net_2 = MVCNN(net_name, net, nclasses=12, cnn_name='resnet18', num_views=12)
    net_2.cuda()
    del net
    print(f'{sum(x.numel() for x in net_2.net_1.parameters()) + sum(x.numel() for x in net_2.net_2.parameters())} parameters in total')
    print(f'{sum(x.numel() for x in net_2.net_2.parameters())} parameters in the last layer')

    optimizer = torch.optim.Adam(net_2.parameters(), lr=5e-05, weight_decay=0.0, betas=(0.9, 0.999))
    loss_func = nn.CrossEntropyLoss()
    hyper_p = util.HyperP(
        lr=5e-05,
        batch_size=4,
        num_workers=1,
        epochs=300,
    )
    tb_trainer = ToyboxTrainer(
        tr=['rzplus', 'rzminus'],
        nview=12,
        ratio=[ratio],
        mode='mv',
        net=net_2,
        net_name=net_name,
        optimizer=optimizer,
        loss_func=loss_func,
        hyper_p=hyper_p
    )
    print(f'=== {tb_trainer.exp_name} ===')
    tb_trainer.train_test_save()


def main():
    ratios = [100, 75, 50, 25]
    for ratio in ratios:
        exp_main(ratio)


if __name__ == '__main__':
    main()
