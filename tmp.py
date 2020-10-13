import conf
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from toybox_data import ToyboxData


root = conf.ToyboxDataDir
rot = ['rzplus', 'rzminus']
nview = 12
ratio = [100]
mode = 'sp'
dataset = 'train'
tb_data = ToyboxData(root, rot, nview, ratio, mode, dataset, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(
    tb_data,
    batch_size=32,
    shuffle=True,
    pin_memory=True,
    drop_last=True
)

for x, y_true, file in train_loader:
    plt.imshow(x[0].permute(1, 2, 0).cpu().numpy())
    plt.show()
    print(x.shape)

    print('test')
