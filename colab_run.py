import argparse

from exps import classic_exp, mvcnn_exp, sphere_exp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-ratios', type=str, help='The ratios for the Toybox data')
    parser.add_argument('-trs', type=str, help='The transformations for the Toybox data')
    parser.add_argument('-net_type', type=str, help='The net_type for training, classic, mvcnn or sphere')

    args = parser.parse_args()

    ratios = list(map(int, args.ratios.split(' ')))
    trs = args.trs.split(' ')
    net_type = args.net_type

    for r in ratios:
        if net_type == 'classic':
            classic_exp.exp_main([r], trs)
        elif net_type == 'mvcnn':
            mvcnn_exp.exp_main([r], trs)
        elif net_type == 'sphere':
            sphere_exp.exp_main([r], trs)
        else:
            raise ValueError(f'Invalid net type {net_type}')
