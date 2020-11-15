import os
import sys
import itertools
from joblib import Parallel, delayed
sys.path.append(os.path.abspath(".."))

from tqdm import tqdm
import numpy as np
import pandas as pd
from app.desopo_and_lefkowitz import Model
import mlflow



# 確率を計算するチームを決定
team = "De"
season = 2018
flg_parallel = False

folder = "../data_csv/batting_number_data/"

# 打順別成績のロード
print("Start : {}_{}".format(team, season))
table = pd.read_csv(folder+team+"_"+str(season)+"_batting.csv", index_col="INDEX")
print("End : {}_{}".format(team, season))
# 走塁のデータ
runner_condition_table = pd.read_csv("../NPB_runner_condition_and_results_data_2018.csv")

# set a condition to Despo Lefkowitz model
model = Model(model_name=f"{team}_{season}", runner_condition_table=runner_condition_table)

def calc(batting_number):
    bn_string = "".join(map(str, [bn+1 for bn in batting_number]))
    with mlflow.start_run():
        mlflow.set_tag("Team", team)
        mlflow.set_tag("Season", season)
        mlflow.log_param(key='Batting_number', value=bn_string)
        ERBG, sigma, Run_prob = model.run(table=table.iloc[list(batting_number)])
        mlflow.log_metric(key='Avg', value=ERBG)
        mlflow.log_metric(key="Sigma", value=sigma)
        mlflow.log_metrics(dict(zip([f"Run_{r}" for r in range(model.R_max)], Run_prob)))

database = "sqlite:///Run_and_ERBG.db"
mlflow.set_tracking_uri(database)
batting_number_list = list(itertools.permutations(range(9)))

if flg_parallel:
    Parallel(n_jobs=-1)([delayed(calc)(batting_number) for batting_number in batting_number_list])
else:
    for batting_number in tqdm(batting_number_list):
        calc(batting_number)