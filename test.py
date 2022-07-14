import random 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math
import multiprocessing



def count(a,result_list):
    count = np.array([1,2,3,4.0])
    count[0] += a
    result_list.append(count)



if __name__ == "__main__":

    ### パラメータ
    trials = 100
    code_distance = 9
    p_div = np.arange(7,13,1)
    a = 1
    # プロセスを管理する人。デラックスな共有メモリ
    manager = multiprocessing.Manager()
    # マネージャーからリストオブジェクトを取得
    result_list = manager.list()
    # あとでまとめてjoin()するためのプロセスのリスト
    processes = []
    # プロセスを10個生成
    for _ in range(24):
        # マネージャーから取得したオブジェクトを引数に渡す
        process = multiprocessing.Process(target=count, args=(a,result_list))
        # プロセス開始
        process.start()
        # プロセスのリストに追加
        processes.append(process)

    # プロセスのリストでループ
    for p in processes:
        # プロセスの終了待ち
        p.join()

    print(result_list)


    for i in range(24):
        if i == 0:
            c = result_list[0]
        else:
            c += result_list[i]
    print(c)
    c /= 24
    print(c)
