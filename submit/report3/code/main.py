# !python3.10.6
# mian function

import matplotlib.pyplot as plt
import numpy as np
from monte_carlo import calc_mc


def mc_plot_m(file_dat:str="./data/mc_t.dat", file_fig:str="./data/mc_t.pdf") -> None:
    data = np.loadtxt(file_dat)  # テキストファイルからデータを読み込み
    print(data.shape)

    # スライスを使って、2次元配列から1次元配列を抜き出す
    t = data[:, 0]  # 温度
    m_mean = np.absolute(data[:, 1])  # 磁化の平均値
    m_std = data[:, 2]  # 統計誤差

    # グラフ作成
    fig, ax = plt.subplots()  # オブジェクト指向インターフェース
    opt = dict(
        linestyle = 'solid',  # 線の種類
        linewidth = 1,  # 線の幅
        color = 'blue',  # 線の色
        ecolor = 'dimgray',  # 誤差棒の色
        elinewidth = 1,  # 誤差棒の線幅
        capsize = 2,  # 誤差棒の端の横棒の長さ
        marker = 'o',  # マーカーの種類
        markersize = 6,  # マーカーのサイズ
        markeredgewidth = 1,  # マーカーの淵の線幅
        markeredgecolor = 'darkblue',  # マーカーの淵の色
        markerfacecolor = 'lightblue',  # マーカーの内側の色
    )
    ax.errorbar(t, m_mean, m_std, **opt)  # 誤差棒付きグラフ
    ax.axhline(y=0, color='k')  # x軸
    # ax.set_xlim(left=0)
    ax.set_xlabel(r"$T$")  # xラベル
    ax.set_ylabel(r"$|m|$")  # yラベル
    fig.savefig(file_fig)  # グラフをファイルに保存


def main(
    filename:str = "./data/mc_t.dat",
    system = (4, 4),    # 系のサイズ
    tmin = 0.5,         
    tmax = 5.0,         
    nt = 10,            # 温度点の数
    n_measure = 1000,   # ビンあたりの測定回数
    n_bin = 10,         # ビンの数
    n_warmup = 100      # ウォームアップの回数
):
    ts = np.linspace(tmin, tmax, nt)  # 温度メッシュ
    qs = []  # 結果を保存するリスト
    for t in ts:
        print("\n================")
        print(f"T = {t}")
        n_mc = (n_bin, n_measure, n_warmup)
        q_mean, q_std = calc_mc(system, J=1, beta=1/t, n_mc=n_mc)  # MC計算
        print(f"mean: {q_mean}")
        print(f"std : {q_std}")
        qs.append([q_mean, q_std])  # 結果をリストqsに追加

    print("\n================")
    print("finish")

    # リストqsをNumPy配列に変換して整形
    qs = np.array(qs)  # shape=(nt, 2, nq); nqは物理量の個数
    qs = np.transpose(qs, [0, 2, 1])  # -> (nt, nq, 2)
    qs = qs.reshape((nt, -1))  # -> (t, 2*nq)
    print(qs.shape)

    # 結果をテキストファイルに保存
    #   format: T q1_mean q1_std q2_mean q2_std ...
    np.savetxt(filename, np.hstack([ts[:, None], qs]))  # *1参照
    print(f"Output results into '{filename}'")

if __name__ == '__main__':
    main()