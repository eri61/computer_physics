import numpy as np


# 物理量を測定して、結果を1次元配列として返す関数
def measure(_mc):
    state = _mc.state
    
    # 測定データ
    m = 2 * np.count_nonzero(state) - state.size  # total spin
    return np.array([m,])  


def entropy(m:np.ndarray):
    # m==1の時計算出来ないため一次的に置き換え
    m = np.where(m==1, np.nan, m)

    m_plus = (1 + m) / 2
    m_minus = (1 - m) / 2
    S_Nk = -m_plus * np.log(m_plus) - m_minus * np.log(m_minus)

    # S==nanの時S=0
    return np.where(S_Nk==np.inf, 0, S_Nk)


def specific_heat(m, t):
    # m==1の時計算できないため一次的に置き換え
    m = np.where(m==1, np.nan, m)

    log = np.log((1 + m) / (1 - m))
    left_term  = (1 - m**2) / (1 - (1 - m**2) / t)
    C = left_term * log / 4

    # C==nanの時C=0
    return np.nan_to_num(C)


# 確率 p = e^{-beta*E_1} / e^{-beta*E_0}
def prob_flip(_mc, i, h):

    # probability at h=0
    state = _mc.state
    state_nn = state[_mc.indices_nn[i]]  # 最近接サイトのスピン配置
    num_up = np.count_nonzero(state_nn)  # 最近接サイトのアップスピンの数

    # probability at h
    prob_field = np.exp(h * measure(_mc))

    if state[i]:  # spin up
        return _mc.prob_up[num_up] * prob_field
    else:  # spin down
        return _mc.prob_dn[num_up] * prob_field