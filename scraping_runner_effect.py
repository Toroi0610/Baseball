# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 22:45:21 2019

@author: Yusuke Ogasawara
"""

import pandas as pd
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from time import sleep
import os
import numpy as np
from tqdm import tqdm
from pykakasi import kakasi

# 2018年度シーズンのすべての公式戦について求める．
# そのためにはurlが必要

def make_table_from_soup(search_url, year=2018):
    # pandasを使わずにhtml選手の番号を列に加える．
    resp = requests.get("http://npb.jp" + search_url[:-15] + "index.html")
    soup = BeautifulSoup(resp.content, 'html.parser')
    indexis = soup.find_all("table")

    # 引き分けの場合勝ち投手，負け投手，セーブ投手のtableがない場合
    if len(indexis) == 6:
        battery_info = indexis[2].find_all("tr")
    # 試合が行われなかった場合(中止，ノーゲーム)
    elif (len(indexis) == 2) or (len(indexis) >= 8):
        return None
    # 勝ち投手，負け投手，セーブ投手のtableがある場合
    else:
        battery_info = indexis[3].find_all("tr")

    team_a_pitcher_name = battery_info[0].find("a").text
    team_a_pitcher_id = int(battery_info[0].find("a")["href"].split("/")[3][:-5])
    team_b_pitcher_name = battery_info[1].find("a").text
    team_b_pitcher_id = int(battery_info[1].find("a")["href"].split("/")[3][:-5])
        

    nan_str = "\xa0"
    resp = requests.get("http://npb.jp" + search_url)
    soup = BeautifulSoup(resp.content, 'html.parser')
    tab = soup.find_all("table")

    # 回収するリストの列名
    columns = ["out", "first_base", "second_base", "third_base",
               "batter_name", "batter_id", "pitcher_name", "pitcher_id", 
               "ball", "strike", "result"]

    #　表裏の変更のタイミングを知るためのスイッチ
    pre_out, out = 0, 0
    table_list = []
    pitcher_name = np.array([team_a_pitcher_name, team_b_pitcher_name], dtype="<U8")
    pitcher_id = np.array([team_a_pitcher_id, team_b_pitcher_id], dtype="int")
    
    # 攻撃中のチームを決めるindex
    team_index = 1 # 1・・・先攻, 0・・・後攻
    
    for t in tab[3:]:
        row_list = []
        row = t.find_all("td")

        # 投手交代を含む行の場合
        if len(row) == 6:
            # 交代後のピッチャーの情報
            aft_pitcher = row[0].find_all("a")[-1]
            # その時のアウトカウント
            aft_out = int(row[1].text[0])
            # 攻撃の変わり目だった場合にはteam_indexを変更する．
            aft_team_index = team_index
            if (out == 2) & (aft_out == 0):
                aft_team_index = team_index + 1
                aft_team_index = aft_team_index % 2
            pitcher_name[aft_team_index] = aft_pitcher.text
            pitcher_id[aft_team_index] = int(aft_pitcher["href"].split("/")[3][:-5])
            
            # 投手交代の情報を削除
            del row[0]
            
        if len(row) == 5:
            # アウトカウント
            if row[0].text != nan_str:
                row_list.append(int(row[0].text[0]))
                pre_out = out
                out = int(row[0].text[0])
                if (pre_out == 2) & (out == 0):
                    team_index += 1
                    team_index = team_index % 2
            else:
                row_list.append(np.nan)
            
            # 塁の状況
            base_situation = [0, 0, 0]
            if row[1].text != nan_str:
                if "1" in row[1].text:
                    base_situation[0] = 1
                if "2" in row[1].text:
                    base_situation[1] = 1
                if "3" in row[1].text:
                    base_situation[2] = 1
                if "満" in row[1].text:
                    base_situation = [1, 1, 1]
                    
            row_list.extend(base_situation)
            
            # 打者の名前とID
            batter = row[2]
            if batter.text != nan_str:
                row_list.append(batter.text)
                row_list.append(int(batter.find("a")["href"].split("/")[-1][:-5]))
            else:
                row_list.extend([np.nan, np.nan])
                
            # 投手の名前とID
            if batter.text != nan_str:
                row_list.append(pitcher_name[team_index])
                row_list.append(pitcher_id[team_index])
            else:
                row_list.extend([np.nan, np.nan])
            

            # ボールとストライク
            if row[3].text != nan_str:
                row_list.append(int(row[3].text[0]))
                row_list.append(int(row[3].text[2]))
            else:
                row_list.extend([np.nan, np.nan])
            
            # 打撃結果
            if row[4].text == nan_str:
                row_list.append(np.nan)
            elif "盗塁" in row[4].text:
                runner = row[4]
                row_list.append(runner.text)
                row_list[4] = runner.text.split("）")[0].split("・")[-1]
                row_list[5] = int(runner.find("a")["href"].split("/")[-1][:-5])
                row_list[6] = pitcher_name[team_index]
                row_list[7] = pitcher_id[team_index]
            else:
                row_list.append(row[4].text)
                

        table_list.append(row_list)

    return pd.DataFrame(table_list, columns=columns)
    

def load_status_and_batting_results_data(year=2018):
    
    if os.path.isfile("NPB_status_and_batting_results_data_{}.csv".format(year)):
        table = pd.read_csv("NPB_status_and_batting_results_data_{}.csv".format(year))
        runner_condition_table = pd.read_csv("NPB_runner_condition_and_results_data_{}.csv".format(year))
    else:
        monthes = ["03", "04", "05", "06", "07", "08", "09", "10", "11"]
        search_urls = []
        for month in monthes:
            sleep(0.1)
            print("Start : " + month)
            url = "http://npb.jp/games/{}/schedule_{}_detail.html".format(year, month)
            resp = requests.get(url)
            soup = BeautifulSoup(resp.content, 'html.parser')
            td_code = soup.find_all("td")
            for td in td_code:
                a_list = td.find_all("a")
                if a_list != []:
                    search_urls.append(a_list[0].get_attribute_list("href")[0]+"playbyplay.html")
                    

        
        # 結果を保存するテーブルを作成
        table = pd.DataFrame([], columns=['out', 'first_base', 'second_base', 'third_base', 'batter_name',
                                          'batter_id', 'pitcher_name', 'pitcher_id', 'ball', 'strike', 'result'])
        
        for search_url in tqdm(search_urls):

            #print("Start : " + search_url)
            sleep(1)

#            # 試合URLから試合経過を取得
#            if year != 2018:
#                indexis = pd.read_html("http://npb.jp" + search_url[:-15] + "index.html")
#
#                # 引き分けの場合勝ち投手，負け投手，セーブ投手のtableがない場合
#                if len(indexis) == 6:
#                    index = indexis[2]
#                # 試合が行われなかった場合(中止，ノーゲーム)
#                elif (len(indexis) == 2) or (len(indexis) >= 8):
#                    continue
#                # 勝ち投手，負け投手，セーブ投手のtableがある場合
#                else:
#                    index = indexis[3]
#                team_a_starting_pitcher = index.iloc[:, 1][0].split("、")[0]
#                team_b_starting_pitcher = index.iloc[:, 1][1].split("、")[0]
#    
#                table_one_match = pd.read_html("http://npb.jp" + search_url)
#                
#                # playerの番号を取得する．
#                resp = requests.get(search_url)
#                soup = BeautifulSoup(resp.content, 'html.parser')
#                td_code = soup.find_all("td")
#                
#                
#    
#                # 先発投手の行を挿入
#                table_one_match.insert(3, pd.DataFrame([[team_b_starting_pitcher, np.nan, np.nan, np.nan, np.nan]],
#                                              columns=['アウト', '塁上', '打者', 'カウント', '結果']))
#    
#                # 1回の裏の開始位置を検索
#                outs = np.array([out.values[0][0] for out in table_one_match])
#                i = 0
#                for pre_out, out in zip(outs[:-1], outs[1:]):
#                    i += 1
#                    if (pre_out == "2アウト") & (out == "0アウト"):
#                        break
#            
#                table_one_match.insert(i, pd.DataFrame([[team_a_starting_pitcher, np.nan, np.nan, np.nan, np.nan]],
#                                              columns=['アウト', '塁上', '打者', 'カウント', '結果']))
#            
#            # 塁に選手がいる状態での打撃の結果を集計
#            table_one_match = pd.DataFrame(np.array([t.values[0] for t in table_one_match[3:]]),
#                                           columns=['アウト', '塁上', '打者', 'カウント', '結果'])
            
            table_one_match = make_table_from_soup(search_url, year)

            table = pd.concat([table, table_one_match])
                
        # 列名つける
        # table.columns = ["out", "base_situation", "batter_name", "counts", "result"]

        # index振りなおし
        table = table.reset_index(drop=True)
        
        # 盗塁時打者カウントffill埋め
        col = ["batter_name", "batter_id", "pitcher_name", "pitcher_id", "ball", "strike"]
        table[col] = table[col].fillna(method="bfill")
        
        # base_situationの無塁を加える．
        # table["base_situation"] = table["base_situation"].fillna("無塁")
        
        # ["out"]に投手の情報が入っているので，["count"]がnanの行の["out"]の投手名を取り出して["pitcher_name"]の列を作る．
        # extract_pitcher_name = lambda event : event.split(" ")[-1]
        # table["pitcher_name"] = table[table["result"].isnull()]["out"].map(extract_pitcher_name)

        # pitcher_nameのNaNを前の値で穴埋め
        # table["pitcher_name"] = table["pitcher_name"].fillna(method="ffill")
        
        # アウトカウント2→0の場所を攻撃の交代として，ピッチャーを入れ替える．ffillを用いるので変わり目に選手名を記述すればOK
        
#        k = (table["out"]=="0アウト") * np.roll(table["out"]!="0アウト", 1)
#        table["inning_change"] = k
#        table["pitcher_switch"] =  table["result"].isnull().values
#        pre_pitcher = np.array("あ")
#        pitcher = np.array("い")
#        s = np.array("う")
#        pitcher_name_array = table.pitcher_name.values
#        inds = table[table["inning_change"] + table["pitcher_switch"]].index.tolist()
#
#        for ind in tqdm(inds):
##            print("\nind : " + str(ind))
#            state = table.iloc[ind]
#            if (state[["inning_change", "pitcher_switch"]] == np.array([False, True])).all():
#                if (table.iloc[ind+1][["inning_change", "pitcher_switch"]] == np.array([True, False])).all():
#                    pre_pitcher = np.array(pitcher_name_array[ind])
#                else:
#                    pitcher = np.array(pitcher_name_array[ind])
##                print("pitcher_switch")
##                print("Pre_pitcher : " + pre_pitcher.tolist())
##                print("pitcher : " + pitcher.tolist())
#            elif (state[["inning_change", "pitcher_switch"]] == np.array([True, False])).all():
#                s = np.copy(pitcher)
#                pitcher = np.copy(pre_pitcher)
#                pre_pitcher = np.copy(s)
#                pitcher_name_array[ind] = pitcher.tolist()
##                print("inning_change")
##                print("Pre_pitcher : " + pre_pitcher.tolist())
##                print("pitcher : " + pitcher.tolist())
#
#        table["pitcher_name"] = pitcher_name_array
        
        # 投手交代の行を削除
#        table = table.dropna(subset=["result"])
#        table["pitcher_name"] = table["pitcher_name"].fillna(method="ffill")
        
        # strike, ballカウント
#        table["strike"] = table["counts"].dropna().map(lambda ball_counts : int(ball_counts[2]))
#        table["ball"] = table["counts"].dropna().map(lambda ball_counts : int(ball_counts[0]))
#        # outカウントをintに
#        table["out"] = table.dropna(subset=["result"])["out"].map(lambda count : int(count[0]))
        
        # first_base, second_base, third_base をTrue or False　で塁状況を表示
        """
        np.unique(table["base_situation"])
        array(['1・2塁', '1・3塁', '1塁', '2・3塁', '2塁', '3塁', '満塁', '無塁'], dtype=object)
        """
#        base_situation = table["base_situation"]
#        base_array = np.zeros([len(base_situation), 3], dtype=bool)
#        base_array[base_situation=='1・2塁'] = [1, 1, 0]
#        base_array[base_situation=='1・3塁'] = [1, 0, 1]
#        base_array[base_situation=='1塁'] = [1, 0, 0]
#        base_array[base_situation=='2・3塁'] = [0, 1, 1]
#        base_array[base_situation=='2塁'] = [0, 1, 0]
#        base_array[base_situation=='3塁'] = [0, 0, 1]
#        base_array[base_situation=='満塁'] = [1, 1, 1]
#        
#        table["first_base"], table["second_base"], table["third_base"] = base_array[:, 0], base_array[:, 1], base_array[:, 2]
        
        # 打球方向の決定
        # resultから打球を処理した人を取り出し，123456789のポジションと
        # 10・・・左中間，11・・・右中間でbatted_ball_directionを保存．
        direction_list = ["その他", "ピッチャー", "キャッチャー", "ファースト", "セカンド", "サード", "ショート",
                          "レフト", "センター", "ライト", "左中間", "右中間"]
        k = np.array([[direction in result for direction in direction_list] for result in table["result"].values])
        direction = np.argmax(k, axis=1)
        table["direction"] = direction
        
        # resultをsingle, double, triple, homerunなど
        result_eng_list = ["その他", "ヒット", "ツーベース", "スリーベース", "ホームラン", "フォアボール",
                           "デッドボール", "犠牲バント", "犠牲フライ", "三振", "振り逃げ", "併殺打", "ゴロ", "フライ", "ライナー",
                           "二塁盗塁成功", "二塁盗塁失敗", "三塁盗塁成功", "三塁盗塁失敗", "本塁盗塁成功", "本塁盗塁失敗"]
        
        result_name = ["other", "single", "double", "triple", "homerun", "walk",
                       "hit_by_pitch", "sacrifice_bant", "sacrifice_fly", "struck_out", "struck_out",
                       "double_play", "grounder", "fly", "liner", "stolen_base_to_second",
                       "caught_stealing_to_second", "stolen_base_to_third", "caught_stealing_to_third",
                       "stolen_base_to_home", "caught_stealing_to_home"]
        
        k = np.array([[result_eng in result for result_eng in result_eng_list] for result in table["result"].values])
        result_eng = [result_name[i] for i in np.argmax(k, axis=1)]
        table["result_name"] = result_eng
        
        # DEspo & Lefkowitzの25の状況を保存
        situation = np.zeros(len(table), dtype="int")
        situation_array = table[["out", "first_base", "second_base", "third_base"]].astype("int").values
        despo_lefkowitz = np.array([[0, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1],
                                    [0, 1, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 1],
                                    [0, 1, 1, 1],
                                    [1, 0, 0, 0],
                                    [1, 1, 0, 0],
                                    [1, 0, 1, 0],
                                    [1, 0, 0, 1],
                                    [1, 1, 1, 0],
                                    [1, 1, 0, 1],
                                    [1, 0, 1, 1],
                                    [1, 1, 1, 1],
                                    [2, 0, 0, 0],
                                    [2, 1, 0, 0],
                                    [2, 0, 1, 0],
                                    [2, 0, 0, 1],
                                    [2, 1, 1, 0],
                                    [2, 1, 0, 1],
                                    [2, 0, 1, 1],
                                    [2, 1, 1, 1],
                                    ], dtype="int")
        
        for i, sit in enumerate(despo_lefkowitz):
            situation[(situation_array == sit).all(axis=1)] = i
            
        # 現在のアウト塁の状況
        table["despo_lefkowitz"] = situation
        
        # 次のアウト塁の状況
        table["despo_lefkowitz_next"] = np.roll(table["despo_lefkowitz"].values, -1)
        
        pivot_table = table.pivot_table(values=["out"],
                                        index=["despo_lefkowitz", "result_name"], 
                                        columns=["despo_lefkowitz_next"],
                                        aggfunc="count", fill_value=0)
        
        # 2パターンの分岐を持つ条件
        runner_conditions_double_pattern = [[1, "single", 4, 5],
                                            [9, "single", 12, 13],
                                            [17, "single", 20, 21],
                                            [5, "single", 4, 5],
                                            [13, "single", 12, 13],
                                            [21, "single", 20, 21],
                                            [1, "double", 6, 2],
                                            [9, "double", 14, 10],
                                            [17, "double", 22, 18],
                                            [4, "double", 6, 2],
                                            [12, "double", 14, 10],
                                            [20, "double", 22, 18],
                                            [5, "double", 6, 2],
                                            [13, "double", 14, 10],
                                            [21, "double", 22, 18],
                                            [2, "single", 5, 1],
                                            [10, "single", 13, 9],
                                            [18, "single", 21, 17],
                                            [6, "single", 5, 1],
                                            [14, "single", 13, 9],
                                            [22, "single", 21, 17],
                                            ]
   
        # 2パターンの分岐を持つ条件の結果を保存
        results_runner_condition_double_pattern = []


        # 3パターンの分岐を持つ条件
        runner_conditions_triple_pattern = [[4, "single", 7, 4, 5],
                                            [12, "single", 15, 12, 13],
                                            [20, "single", 23, 20, 21],
                                            [7, "single", 7, 4, 5],
                                            [15, "single", 15, 12, 13],
                                            [23, "single", 23, 20, 21],
                                            ]
        
        # 3パターンの分岐を持つ条件の結果を保存
        results_runner_condition_triple_pattern = []
        
        # 2パターンの分岐を持つ条件の計算
        for runner_condition in runner_conditions_double_pattern:
            
            # 条件を満たす配列の抽出
            condition_array = pivot_table.loc[(runner_condition[0], runner_condition[1])][[runner_condition[2], runner_condition[3]]].values
            
            # 割合を保存
            results_runner_condition_double_pattern.append(condition_array)

        # 3パターンの分岐を持つ条件の計算
        for runner_condition in runner_conditions_triple_pattern:
            
            # 条件を満たす配列の抽出
            condition_array = pivot_table.loc[(runner_condition[0], runner_condition[1])][[runner_condition[2], runner_condition[3], runner_condition[4]]].values
            
            # 割合を保存
            results_runner_condition_triple_pattern.append(condition_array)
        
        
        results_runner_condition_double_pattern = np.array(results_runner_condition_double_pattern)

        results_runner_condition_triple_pattern = np.array(results_runner_condition_triple_pattern)

        percent_runner_condition_double_pattern = results_runner_condition_double_pattern / results_runner_condition_double_pattern.sum(axis=1)[:, None]
        percent_runner_condition_triple_pattern = results_runner_condition_triple_pattern / results_runner_condition_triple_pattern.sum(axis=1)[:, None]
        

        # データを成形        
        # 3パターンの分岐を持つ条件の結果データの成形
        runner_condition_table = pd.DataFrame(runner_conditions_triple_pattern,
                                               columns=["current_situation", 
                                                        "Reuslt_name",
                                                        "Next_situation_1",
                                                        "Next_situation_2",
                                                        "Next_situation_3",
                                                        ])
        
        runner_condition_table["situation_1"] = results_runner_condition_triple_pattern[:, 0]
        runner_condition_table["situation_2"] = results_runner_condition_triple_pattern[:, 1]
        runner_condition_table["situation_3"] = results_runner_condition_triple_pattern[:, 2]
        runner_condition_table["percent_1"] = percent_runner_condition_triple_pattern[:, 0]
        runner_condition_table["percent_2"] = percent_runner_condition_triple_pattern[:, 1]
        runner_condition_table["percent_3"] = percent_runner_condition_triple_pattern[:, 2]
        
        runner_condition_table_pre = pd.DataFrame(runner_conditions_double_pattern,
                                                   columns=["current_situation", 
                                                            "Reuslt_name",
                                                            "Next_situation_1",
                                                            "Next_situation_2",
                                                            ])

        runner_condition_table_pre["situation_1"] = results_runner_condition_double_pattern[:, 0]
        runner_condition_table_pre["situation_2"] = results_runner_condition_double_pattern[:, 1]
        runner_condition_table_pre["percent_1"] = percent_runner_condition_double_pattern[:, 0]
        runner_condition_table_pre["percent_2"] = percent_runner_condition_double_pattern[:, 1]
        
        
        
        runner_condition_table = runner_condition_table.append(runner_condition_table_pre)
        
        runner_condition_table = runner_condition_table.loc[:, ["current_situation", 
                                                                "Reuslt_name",
                                                                "Next_situation_1",
                                                                "Next_situation_2",
                                                                "Next_situation_3",
                                                                "situation_1",
                                                                "situation_2",
                                                                "situation_3",
                                                                "percent_1",
                                                                "percent_2",
                                                                "percent_3"
                                                                ]]

        # テーブルデータの保存
        table.to_csv("NPB_status_and_batting_results_data_{}.csv".format(year), index = False)

        # 条件分岐のテーブルデータの保存
        runner_condition_table.to_csv("NPB_runner_condition_and_results_data_{}.csv".format(year), index = False)
        
    return table, runner_condition_table


def process_table(table):

    # 代打を取り除く
    table["batter_name"] = [name.split("・")[-1] for name in table["batter_name"]]
    
    # 各選手で集計
    result_name = ["other", "single", "double", "triple", "homerun", "walk",
                   "hit_by_pitch", "sacrifice_bant", "sacrifice_fly", "struck_out",
                   "double_play", "grounder", "fly", "liner", "stolen_base_to_second",
                   "caught_stealing_to_second", "stolen_base_to_third", "caught_stealing_to_third",
                   "stolen_base_to_home", "caught_stealing_to_home"]

    batter_stats = pd.pivot_table(table, values=["out"], index=["batter_name", "batter_id"], aggfunc="count", columns=["result_name"], fill_value=0).astype("int")
    batter_stats.columns = batter_stats.columns.levels[1]
    batter_stats = batter_stats[result_name]
    
    pitcher_stats = pd.pivot_table(table, values=["out"], index=["pitcher_name", "pitcher_id"], aggfunc="count", columns=["result_name"], fill_value=0).astype("int")
    pitcher_stats.columns = pitcher_stats.columns.levels[1]
    pitcher_stats = pitcher_stats[result_name]
    
    return batter_stats, pitcher_stats

if __name__ == "__main__":
    for year in range(2017, 2019):
        table, runner_condition_table = load_status_and_batting_results_data(year)
    
    batter_stats, pitcher_stats = process_table(table)
    