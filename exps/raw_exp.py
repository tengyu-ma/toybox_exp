import conf

from data.raw_data import ToyboxRawData


def main():
    root = conf.ToyboxSquaredCropped
    ca = conf.ALL_CA
    no = conf.ALL_NO
    tr = conf.ALL_TR
    fr = conf.ALL_1FPS
    tb_data = ToyboxRawData(root, ca=ca, no=no, tr=tr, fr=fr)

    tb_data.squares_ready()


if __name__ == '__main__':
    main()
