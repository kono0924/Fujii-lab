import random 
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import math
import time
import os
import multiprocessing

###### ゲートやエラーの定義 #####
#エラーの定義
def x_error(qubits,i):
    qubits[0][i] ^= 1
def y_error(qubits,i):
    qubits[0][i] ^= 1    
    qubits[1][i] ^= 1
def z_error(qubits,i):
    qubits[1][i] ^=  1 

def single_biased(qubit,i,p,eta): # etaはバイアス
    p_x = p / (2*(1+eta))
    p_z = p * eta / (1 + eta) 
    prob = random.random() 
    if prob < p_z: #Z error
        z_error(qubit,i)
    elif prob < p_z+p_x: # X error
        x_error(qubit,i)
    elif prob < p_z+2*p_x: # Y error
        y_error(qubit,i)

def bitflip_error(qubit,i,p,eta): # etaはバイアス
    p_x = p / (2*(1+eta))
    p_z = p * eta / (1 + eta) 
    prob = random.random() 
    if prob < p_x+p_z: #Z error
        x_error(qubit,i)

# i番目にHadamard gateを作用させる
def H(qubits,i):
    qubits[0][i]^=qubits[1][i]
    qubits[1][i]^=qubits[0][i]
    qubits[0][i]^=qubits[1][i]

#i番目がcontrolビット j 番目がtargetビットのCNOTgate
def CNOT(qubits,c,t):
    qubits[0][t]^=qubits[0][c]
    qubits[1][c]^=qubits[1][t]

# qubitsからデータの部分のみ取得
def data_qubits(qubits, code_distance):
    a = [[],[]]
    for i in range(code_distance):
        a[0].append(qubits[0][2*i])
        a[1].append(qubits[1][2*i])
    return a

##### detection eventを作成する関数の定義 #####
def measurement(qubits,code_distance,p,eta):
    nqubits = 3*code_distance-1
    qubits = [[0 for _ in range(nqubits)],[0 for _ in range(nqubits)]]
    ancilla_1 = [0 for _ in range(code_distance-1)] #X測定の結果を格納
    ancilla_2 = [0 for _ in range(code_distance-1)] #X測定の結果を格納

    # T字の左
    for i in range(code_distance-1):
        CNOT(qubits,i,i+1)
        single_biased(qubits,i,p,eta)
        single_biased(qubits,i+1,p,eta)
        if i % 2 == 1:
            ancilla_1[int((i-1)/2)] = qubits[0][i]
            #qubits[0][i] = 0
    # T字の右
    for i in reversed(range(code_distance-1)):
        CNOT(qubits,code_distance+i,code_distance+i-1)
        single_biased(qubits,code_distance+i,p,eta)
        single_biased(qubits,code_distance+i-1,p,eta)
        if i % 2 == 0:
            ancilla_1[int((code_distance-1)/2)+int(i/2)] = qubits[0][code_distance+i]
            #qubits[0][code_distance+i] = 0
    # T字の下aass
    CNOT(qubits,code_distance-1,2*code_distance-1)
    single_biased(qubits,code_distance-1,p,eta)
    single_biased(qubits,2*code_distance-1,p,eta)
    for i in range(code_distance-1):
        CNOT(qubits,2*code_distance-1+i,2*code_distance+i)
        single_biased(qubits,2*code_distance-1+i,p,eta)
        single_biased(qubits,2*code_distance+i,p,eta)
        if i == 0:
            continue
        ancilla_2 = qubits[1][2*code_distance-1+i]
    result = qubits[0][3*code_distance-2]
    #print(qubits)

    #### ここで測定パート終わり
    D = [0]*(code_distance-1)
    for i in range(2*code_distance-1):
        if i%2 == 1:
            D[int((i-1)/2)] = qubits[1][int((i-1)/2)]
    #print("D0", D)

    #print(qubits)
    ### 元に戻すパート
    for i in reversed(range(code_distance-1)):
        CNOT(qubits,i,i+1)
        single_biased(qubits,i,p,eta)
        single_biased(qubits,i+1,p,eta)
    for i in range(code_distance-1):
        CNOT(qubits,code_distance+i,code_distance+i-1)
        single_biased(qubits,code_distance+i,p,eta)
        single_biased(qubits,code_distance+i-1,p,eta)

    # アンシラ測定前のアダマール
    for i in range(2*code_distance-1):
        if i%2 == 1:
            H(qubits,i) 
            #bitflip_error(qubits,i,p,eta)

    """
    ### アンしらによってデータ反転
    for i in reversed(range(code_distance-1)):
        if i %2 == 1:
            if qubits[0][i] == 1:
                qubits[1][i-1] ^= 1
    for i in range(code_distance-1):
        if i%2 == 0:
            if qubits[0][code_distance+i] == 1:
                qubits[1][code_distance+i+1] ^= 1
    """

    # アンしらの初期化
    for i in range(2*code_distance-1):
        if i %2 == 1:
            H(qubits,i) 
            qubits[1][i] = 0

    #print("②", data_qubits(qubits,code_distance))

    ###### 誤り訂正パート 一回目
    D = np.zeros((code_distance-1,3))
    
    for i in range(2*code_distance-2):
        if i % 2 == 0:
            CNOT(qubits,i+1,i)
            #single_biased(qubits,i,p,eta)
            #single_biased(qubits,i+1,p,eta)
        if i % 2 == 1:
            CNOT(qubits,i,i+1)
            #single_biased(qubits,i,p,eta)
            #single_biased(qubits,i+1,p,eta)
    for i in range(2*code_distance-1):
        if i%2 == 1:
            #bitflip_error(qubits,i,p,eta) #シンドローム測定前の反転
            D[int((i-1)/2)][1] = qubits[1][i]
    ### 誤り訂正2回目
    #シンドロームから
    for i in range(2*code_distance-1):
        if i %2 == 1:
            qubits[1][i] = 0
            qubits[0][i] = 0
    for i in range(2*code_distance-2):
        if i % 2 == 0:
            CNOT(qubits,i+1,i)
            #single_biased(qubits,i,p,eta)
            #single_biased(qubits,i+1,p,eta)
        if i % 2 == 1:
            CNOT(qubits,i,i+1)
            #single_biased(qubits,i,p,eta)
            #single_biased(qubits,i+1,p,eta)
    for i in range(2*code_distance-1):
        if i%2 == 1:
            #bitflip_error(qubits,i,p,eta) #シンドローム測定前の反転
            D[int((i-1)/2)][2] = qubits[1][i]
    
    """ #データビットから
    a = data_qubits(qubits,code_distance)[1]
    b = []
    for i in range(len(a)-1):
        c = (a[i]+a[i+1])%2
        b.append(c)
    for i in range(2*code_distance-1):
        if i%2 == 1:
            #bitflip_error(qubits,i,p,eta) #シンドローム測定前の反転
            D[int((i-1)/2)][2] = b[int((i-1)/2)]
    """

    # detection eventの行列
    E = np.zeros((code_distance-1,2))
    for i in range(code_distance-1):
        for j in range(2):
            E[i,j] = (D[i,j] + D[i,j+1]) % 2
    #print("D=", D)
    #print("E=", E)

    # detection eventの数が何個か数える
    edge_of_decoder_graph = []
    for i in range(code_distance-1):
        for j in range(2):
            if E[i,j] == 1:
                edge_of_decoder_graph.append((i,j))
    if len(edge_of_decoder_graph)%2==1:
            edge_of_decoder_graph.append('external')

    ### 最小距離のグラフの作成
    gp = nx.Graph()
    # 頂点の追加
    for i in range(code_distance-1):
        gp.add_node((i,0))
        gp.add_node((i,1))
    # 辺と重みの追加
    for i in range(code_distance-2):
        gp.add_edge((i,0),(i+1,0),weight=1)
        gp.add_edge((i,1),(i+1,1),weight=1)
    for i in range(code_distance-1):
        gp.add_edge((i,0),(i,1),weight=1)
    for i in range(code_distance-2):
        gp.add_edge((i,0),(i+1,1),weight=1)
    gp.add_node('external')
    gp.add_edge('external',(0,0),weight=1)
    gp.add_edge('external',(code_distance-2,0),weight=1)

    ### データqubitの訂正
    result_data = [0]*code_distance
    for i in range(code_distance):
        result_data[i] = qubits[1][2*i]
    aaa = result_data.copy()

    # イベント間の最小距離の計算
    mwpm_gp = nx.Graph() 
    for i in range(len(edge_of_decoder_graph)):
            mwpm_gp.add_node(i)
    for i in range(len(edge_of_decoder_graph)):
        for j in range(i):
            shortest_path_weight = nx.dijkstra_path_length(gp, edge_of_decoder_graph[i],edge_of_decoder_graph[j])
            mwpm_gp.add_edge(i,j,weight = shortest_path_weight)
    mwpm_res = nx.min_weight_matching(mwpm_gp)
    match_path = []
    for match_pair in mwpm_res:
        match_path.append(nx.dijkstra_path(gp,edge_of_decoder_graph[match_pair[0]],edge_of_decoder_graph[match_pair[1]]))
    for path in match_path:
        for i in range(len(path)): 
            if i !=0: #i=0は飛ばす
                if path[i-1] == 'external': # 左='external'
                    if path[i][0] == 0: #上側エラーなら
                        result_data[0] ^= 1 #上端を反転
                    else: #右端エラーなら
                        result_data[code_distance-1]^= 1 #右端を反転
                        
                elif path[i] == 'external': # 右='external'
                    if path[i-1][0] == 0:
                        result_data[0]^= 1 #上端を反転
                    else:
                        result_data[code_distance-1]^= 1 #右端を反転
                
                elif path[i-1][1] == path[i][1]: #端のエラーではなく、同じサイクルでのエラーなら
                    result_data[min(path[i-1][0],path[i][0])+1] ^= 1
                
                elif path[i-1][0] == path[i][0]:
                    continue

                else:
                    result_data[min(path[i-1][0],path[i][0])+1]^= 1
    for i in range(code_distance):
        qubits[1][2*i] = result_data[i]
    #print("③", data_qubits(qubits,code_distance))
    #print(qubits)

    ### 最後にLZがかかっているか
    result_LZ = 0
    for i in range(2*code_distance+1):
        if i%2 == 1:
            continue
        if qubits[0][i] == 1:
            result_LZ ^= 1
    ### 最後にLXがかかっているか
    result_LX = 0
    if result_data != [0]*code_distance:
        result_LX = 1
    #print("result_data_ver2=",result_data)
    #print("end")
    #print()

    return result_LX, result_LZ, result, aaa

def stastics(code_distance,p,eta):
    num_array = np.arange(3,code_distance+1,2)
    result = np.zeros((int((code_distance-1)/2),3))
    #print(result)
    for c_d in num_array:
        nqubits = 3*c_d-1
        qubits = [[0 for _ in range(nqubits)],[0 for _ in range(nqubits)]]
        a, b, c ,aaa = measurement(qubits,c_d,p,eta)
        result[int((c_d-3)/2),0] = a
        result[int((c_d-3)/2),1] = b
        result[int((c_d-3)/2),2] = c
    return result, aaa

def aaa(code_distance,p,eta,trials,result_list):
    num_array = np.arange(3,code_distance+1,div)
    result = np.zeros((3,int((code_distance-3)/div+1)))
    for c_d in num_array:
        for _ in range(trials):
            nqubits = 3*c_d-1
            qubits = [[0 for _ in range(nqubits)],[0 for _ in range(nqubits)]]
            a, b, c ,aaa = measurement(qubits,c_d,p,eta)
            result[0,int((c_d-3)/div)] += a
            result[1,int((c_d-3)/div)] += b
            result[2,int((c_d-3)/div)] += c
            #print(aaa, b)
    result /= trials
    result_list.append(result)

if __name__ == "__main__":

    ### パラメータ
    trials = 2000
    code_distance = 31
    p_ = 0.009
    eta = 1000
    div = 2
    pro = 500

    # プロセスを管理する人。デラックスな共有メモリ
    manager = multiprocessing.Manager()
    # マネージャーからリストオブジェクトを取得
    result_list = manager.list()
    # あとでまとめてjoin()するためのプロセスのリスト
    processes = []
    # プロセスを生成
    for _ in range(pro):
        # マネージャーから取得したオブジェクトを引数に渡す
        process = multiprocessing.Process(target=aaa, args=(code_distance,p_,eta,trials,result_list,))
        # プロセス開始
        process.start()
        # プロセスのリストに追加
        processes.append(process)

    # プロセスのリストでループ
    for p in processes:
        # プロセスの終了待ち
        p.join()

    for i in range(pro):
        if i == 0:
            c = result_list[0]
        else:
            c += result_list[i]
    c /= pro

    num_array = np.arange(3,code_distance+1,div)
    ind = ["LX", "LZ", "syn"]
    df = pd.DataFrame(data=c, columns=num_array,index=ind)
    df.to_csv('p='+str(p_*100)+'%,eta='+str(eta)+',d=('+str(3)+','+str(code_distance)+','+str(div)+'),trials='+str(trials*pro)+'.csv')
