# !python 3.10.6
# REFERENCE: code of Prof. OTSUKI

import logging

import numpy as np

from .measurement import entropy, measure, specific_heat
from .supplement import mc_init, sweep

# ログの出力名を設定（1）
logger = logging.getLogger('LoggingTest')
 
# ログをコンソール出力するための設定（2）
sh = logging.StreamHandler()

def calc_mc(
    system: tuple, 
    J:float, 
    beta:float, 
    n_mc:float, 
    seed:int=1,
    h:float=0
    ):
    """MC計算を行うメインコード

    Args:
        system (tuple): siteの形状を指定
        J (float): 相互作用のエネルギー
        beta (float): 逆温度
        n_mc (float): (ビンの数, サンプル数, ウォームアップ)
        seed (int, optional): 乱数のseed. Defaults to 1.
    """
    n_bin, n_measure, n_warmup = n_mc

    logger.debug("Initializing MC...")
    mc_data = mc_init(system, J, beta, seed)

    logger.debug("Warming up...")
    quant = []  # 各ビンごとの結果を入れるリスト
    for _ in range(n_bin):
        quant_bin = []  # 測定データを入れるリスト
        for _ in range(n_measure):
            sweep(mc_data, h, beta)  # スイープ
            quant_bin.append(measure(mc_data))  # 測定
        quant.append(np.array(quant_bin).mean(axis=0))  # ビン内で平均
    quant = np.array(quant)
    logger.debug(f"  obtained data shape: {quant.shape}")
    print(".", end="")

    quant /= mc_data.state.size                             # サイトあたり
    quant_mean = quant.mean(axis=0)                         # ビンごとに平均化
    S_Nk = entropy(quant_mean)
    C_Nk = specific_heat(quant_mean, beta)
    quant_std = quant.std(axis=0)                           # 標準偏差
    
    return np.array([quant_mean, quant_std, S_Nk, C_Nk]).reshape(-1, 4)

