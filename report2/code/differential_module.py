import argparse

import numpy as np
import yaml
from scipy.sparse import csr_matrix as sparse_matrix


def make_differential_ops(
    deriv:int, acc:int, nx:int=1000, dx:float=0.1, config:str="config/param.yaml"
    ):
    """make matrix of finite differential coefficient

    Args:
        nx (int): number of division
        dx (float): h
        deriv (int, optional): order of derivation
        acc (int, optional): accuracy
        config (str, optional): path of yaml file

    Returns:
        sparse_matrix: differential operator 
    """
    # get parameter of finite difference
    term, coeff = diff_param(deriv, acc, config)

    # define equation of differential operator
    eq_diff_matrix = f"{coeff['zer']} * f0"

    # operator (matrix) that shift vector components
    f0 = np.identity(nx, dtype=int)
    _range = term//2

    for i in range(1, _range+1):
        # define parameter
        ps_eq = f"ps_f{i} = np.roll(f0, i, axis=1)"
        ng_eq = f"ng_f{i} = ps_f{i}.transpose()"
        ps_coeff = coeff[f'ps{i}']
        ng_coeff = coeff[f'ng{i}']
        eq_diff_matrix += f"+ {ng_coeff} * ng_f{i} + {ps_coeff} * ps_f{i}"
        # execute the setting parameter equation
        exec(ps_eq)
        exec(ng_eq)
    eq_diff_matrix = "sparse_matrix(" + eq_diff_matrix + f") / dx ** {deriv}"
    return eval(eq_diff_matrix)


def diff_param(deriv:int=2, acc:int=4, config='config/param.yaml'):
    # get Finite difference coefficient
    arg = get_args(config)
    arg = arg[f'der{deriv}'][f"accuracy {acc}"]
    return arg.values()

def get_args(config_name):
    # yamlファイルからパラメータの取得
    parser = argparse.ArgumentParser(description='YAMLありの例')
    parser.add_argument('-c', '--config', type=str, help='設定ファイル(.yaml)')
    args = parser.parse_args([f'-c={config_name}'])

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    return config

