import numpy as np


# 物理量を測定して、結果を1次元配列として返す関数
def measure(_mc):
    state = _mc.state
    m = 2 * np.count_nonzero(state) - state.size  # total spin
    return np.array([m,])  

# 確率 p = e^{-beta*E_1} / e^{-beta*E_0}
def prob_flip(_mc, i):
    state = _mc.state
    state_nn = state[_mc.indices_nn[i]]  # 最近接サイトのスピン配置
    num_up = np.count_nonzero(state_nn)  # 最近接サイトのアップスピンの数
    if state[i]:  # spin up
        return _mc.prob_up[num_up]
    else:  # spin down
        return _mc.prob_dn[num_up]