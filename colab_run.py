from exps import classic_exp, mvcnn_exp, sphere_exp


if __name__ == '__main__':
    ratio = 100
    net_type = 'mvcnn'

    if net_type == 'classic':
        classic_exp.exp_main(ratio)
    elif net_type == 'mvcnn':
        mvcnn_exp.exp_main(ratio)
    elif net_type == 'sphere':
        sphere_exp.exp_main(ratio)
    else:
        raise ValueError(f'Invalid net type {net_type}')
