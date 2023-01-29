import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_data(cond_files:str) -> tuple:
    files = glob.glob(cond_files)
    data = pd.DataFrame({"path":files}, columns=['path', 'J', 'T'])

    for i, f in enumerate(files):
        split = re.split('[/_]', f)
        t, j = split[-3:-1]
        data.loc[i, 'J'] = float(j[1:])
        data.loc[i, 'T'] = float(t[1:])

    J_data = data.sort_values(['J', 'T']).set_index(['J', 'T'])
    T_data = data.sort_values(['T', 'J']).set_index(['T', 'J'])
    return {"T_data": T_data, "J_data": J_data}