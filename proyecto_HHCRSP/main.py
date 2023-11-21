import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays as pyholidays
import math
import os
import sys

from app.libs.HHCRS_multiobjective_model import construct_model
from app.libs.constants import shift_type,HMM,DM

# demands dictionary has a first key representing the shift type, a second key representing the day of the week, and a third  separating
# holidays and not holidays using 1 to indicate holiday and 0 to indicate not holiday


if __name__=="__main__":

    
    """df=construct_model('instances/',sys.argv[1])
    if not os.path.exists('results/'):
        os.makedirs('results/')
    df.to_pickle(f"results/results_{sys.argv[1]}.pkl")"""

    working_dir='results/'
    dfs=[]
    for i in range(3,172):
        df=pd.read_pickle(f"{working_dir}results_{i}.pkl")
        df["instance"]=i
        dfs.append(df)
    df=pd.concat(dfs)
    df.to_csv(f"{working_dir}general_results.csv")
