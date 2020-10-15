import gc
import torch
import torch.nn as nn
import util

from nets.mvcnn_net import SVCNN, MVCNN
from exps.trainer import ToyboxTrainer

torch.backends.cudnn.benchmark = True


def exp_main(ratios, trs):

    # STAGE 1 Full View Representation Built
    # print('=== Stage 2 ===')
    # net_name = 'mvcnn_mix'
    # net = SVCNN(net_name, nclasses=12, pretraining=False, cnn_name='resnet18')
    # net.cuda()
    # net_2 = MVCNN(net_name, net, nclasses=12, cnn_name='resnet18', num_views=96)
    # net_2.cuda()
    # model_p = '/media/tengyu/DataU/Log/ToyboxExp/mvcnn_mix-rzplus_rzminus_rxplus_rxminus-12-100_075-mix/state/299-mvcnn_mix-rzplus_rzminus_rxplus_rxminus-12-100_075-mix-state.pkl'
    # net_2.load_state_dict(torch.load(model_p))
    #
    # del net
    # print(f'{sum(x.numel() for x in net_2.net_1.parameters()) + sum(x.numel() for x in net_2.net_2.parameters())} parameters in total')
    # print(f'{sum(x.numel() for x in net_2.net_2.parameters())} parameters in the last layer')

    # optimizer = torch.optim.Adam(net_2.parameters(), lr=5e-05, weight_decay=0.0, betas=(0.9, 0.999))
    # loss_func = nn.CrossEntropyLoss()
    hyper_p = util.HyperP(
        lr=5e-05,
        batch_size=1,
        num_workers=0,
        epochs=300,
    )
    # tb_trainer = ToyboxTrainer(
    #     tr=trs,
    #     nview=12,
    #     ratio=ratios,
    #     mode='mix',
    #     net=net_2,
    #     net_name=net_name,
    #     optimizer=optimizer,
    #     loss_func=loss_func,
    #     hyper_p=hyper_p
    # )
    # print(f'=== {tb_trainer.exp_name} ===')
    # tb_trainer.train_test_save()

    # STAGE 2
    net_name = 'mvcnn_mix'
    net = SVCNN(net_name, nclasses=12, pretraining=False, cnn_name='resnet18')
    net.cuda()
    net_2 = MVCNN(net_name, net, nclasses=12, cnn_name='resnet18', num_views=96)
    net_2.cuda()
    model_p = '/media/tengyu/DataU/Log/ToyboxExp/mvcnn_mix-rzplus_rzminus_rxplus_rxminus-12-100_075-mix/state/299-mvcnn_mix-rzplus_rzminus_rxplus_rxminus-12-100_075-mix-state.pkl'
    net_2.load_state_dict(torch.load(model_p))
    tb_trainer = ToyboxTrainer(
        tr=trs,
        nview=12,
        ratio=ratios,
        mode='mix',
        net=net_2,
        net_name=net_name,
        optimizer=None,
        loss_func=None,
        hyper_p=hyper_p,
        preload=False,
    )
    y_acti_hash = tb_trainer.hash_net_2_activation()

    del net

    print('=== Stage 1 ===')
    net_name = 'svcnn_mix'
    net = SVCNN(net_name, nclasses=12, pretraining=False, cnn_name='resnet18')
    net.cuda()
    print(f'{sum(x.numel() for x in net.net.parameters())} parameters in total')
    print(f'{sum(x.numel() for x in net.net.fc.parameters())} parameters in the last layer')

    optimizer = torch.optim.Adam(net.parameters(), lr=5e-05, weight_decay=0.0)
    loss_func = nn.CrossEntropyLoss()
    hyper_p = util.HyperP(
        lr=5e-05,
        batch_size=64,
        num_workers=0,
        epochs=300,
    )
    tb_trainer = ToyboxTrainer(
        tr=trs,
        nview=12,
        ratio=ratios,
        mode='sv',
        net=net,
        net_2=net_2,
        net_name=net_name,
        optimizer=optimizer,
        loss_func=loss_func,
        hyper_p=hyper_p,
        preload=False,
        y_acti_hash=y_acti_hash
    )
    print(f'=== {tb_trainer.exp_name} ===')
    tb_trainer.train_test_save()

    # del tb_trainer
    # gc.collect()  # clear unused memory



def main():
    ratios = [100, 75]
    trs = ['rzplus', 'rzminus', 'rxplus', 'rxminus']
    exp_main(ratios, trs)


if __name__ == '__main__':
    main()
