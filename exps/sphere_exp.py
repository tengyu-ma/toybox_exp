import torch
import torch.nn as nn
import util

from nets.sphere_net import SphereNet
from exps.trainer import ToyboxTrainer

torch.backends.cudnn.benchmark = True


def exp_main(ratios, trs):
    net_name = 's2cnn'
    net = SphereNet(12)
    net.cuda()
    print(f'{sum(x.numel() for x in net.parameters())} parameters in total')
    print(f'{sum(x.numel() for x in net.out_layer.parameters())} parameters in the last layer')

    optimizer = torch.optim.SGD(net.parameters(), lr=0, momentum=0.9)
    loss_func = nn.NLLLoss()
    hyper_p = util.HyperP(
        lr=0.5,
        batch_size=32,
        num_workers=1,
        epochs=300,
    )
    tb_trainer = ToyboxTrainer(
        tr=trs,
        nview=12,
        ratio=ratios,
        mode='sp',
        net=net,
        net_name=net_name,
        optimizer=optimizer,
        loss_func=loss_func,
        hyper_p=hyper_p
    )
    print(f'=== {tb_trainer.exp_name} ===')
    tb_trainer.train_test_save()


def main():
    ratios = [100]
    trs = ['rzplus', 'rzminus']
    exp_main(ratios, trs)


if __name__ == '__main__':
    main()
