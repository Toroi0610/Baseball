import pandas as pd
import matplotlib.pyplot as plt
from app.desopo_and_lefkowitz import Model

if __name__ == "__main__":
    # 確率を計算するチームを決定
    team_list = ["De"]
    season_list = [2018]

    folder = "./data_csv/batting_number_data/"

    # 打順別成績のロード
    tables = []
    data_index = []
    for team in team_list:
        for season in season_list:
            print("Start : {}_{}".format(team, season))
            table = pd.read_csv(folder+team+"_"+str(season)+"_batting.csv", index_col="INDEX")
            tables.append(table)
            data_index.append("{}_{}".format(team, season))
            print("End : {}_{}".format(team, season))

    # 走塁のデータ
    runner_condition_table = pd.read_csv("NPB_runner_condition_and_results_data_2018.csv")
    model = Model(model_name=data_index[0], runner_condition_table=runner_condition_table)

    ERBG, sigma, Run_prob = model.run(table=table)
    print("Ave   : ", ERBG)
    print("Sigma : ", sigma)
    print("Rp    : ", Run_prob)

