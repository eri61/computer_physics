# !python3.10.6
# Main code: ./monte_carlo.py

from collections import namedtuple

import numpy as np
from mesurement import prob_flip

MC = namedtuple('MC', ['state', 'indices_nn', 'rng', 'prob_up', 'prob_dn'])
"""
MC: 
'state: スピンの配置', 
'indices_nn: 最近接サイトへのインデックス(size, z)', 
'rng: 乱数', 
'prob_up: ', 'prob_dn'
"""

def mc_init(system, J, beta, seed):
    """初期化"""
    # サイトの総数(np.prod: 配列のサイズを計算)
    size = np.prod(system)

    # スピン配置(up: True, down: False)
    state = np.full(size, True, dtype=bool)

    # 乱数を生成
    rng = np.random.default_rng(seed)

    # 最近接サイトへのインデックス(size, z)
    indices_nn = gen_indices_nn(system)

    # 更新確率 e^{-2 K s_i sum_j s_j} を事前に計算
    # prob_up, prob_dn のインデックスは最近接サイトのアップスピンの数
    prob_up = []  # s_i = up の場合
    prob_dn = []  # s_j = down の場合
    K = J * beta  # 無次元化した相互作用
    z = 2 * len(system)  # 最近接格子点の数
    for num_up in range(z+1):
        sum_sigma = 2 * num_up - z  # sum_j s_j
        prob_up.append(np.exp(-2 * K * sum_sigma))
        prob_dn.append(np.exp(2 * K * sum_sigma))

    return MC(state, indices_nn, rng, prob_up, prob_dn)

def gen_indices_nn(system):
    ndim = len(system)
    size = np.prod(system)
    z = 2 * ndim

    # サイトの通し番号(0からサイズ-1)
    index = np.arange(size).reshape(system) # (nx, ny)

    # 最近接格子点の番号を取得
    indices_nn = []
    for axis in range(ndim):
        indices_nn.append(np.roll(index, 1, axis=axis))
        indices_nn.append(np.roll(index, -1, axis=axis))
    indices_nn = np.array(indices_nn)  # リストをNumPy配列に変換
    assert indices_nn.shape == (z,) + system  # (z, nx, ny)

    indices_nn = np.moveaxis(indices_nn, 0, -1)  # (z, nx, ny) → (nx, ny, z)
    assert indices_nn.shape == system + (z,)

    indices_nn = indices_nn.reshape(-1, z)  # (nx, ny, z) → (nx*ny, z)
    assert indices_nn.shape == (size, z)

    return indices_nn
    
# スウィープを行う関数
def sweep(_mc, h:float=0):
    state = _mc.state

    # 事前に必要な数だけ乱数を生成しておく
    randn = _mc.rng.random(state.size)  # 乱数 [0:1)
    sites = _mc.rng.permutation(state.size)  # 順列

    # 全てのスピンを1回づつランダムにスイープ
    for i, rand in zip(sites, randn):
        p = prob_flip(_mc, i, h)  # 更新確率
        if accept(p, rand):  # 更新をするかどうか
            flip(state, i)  # 更新
        # ここのデータ→変化

# サイトiのスピンを反転
def flip(_state, i):
    _state[i] ^= True

# メトロポリス法の判定式（更新を行う場合にはTrueを返す）
def accept(p, rand):
    return p > rand  # rand=[0:1) なので、p>=1 なら常にTrue
