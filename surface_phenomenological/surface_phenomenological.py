import random 
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import math
import multiprocessing

#エラーの定義 qubitは三次元配列で1つ目のインデックスでXかZか、2,3個目のインデックスで位置を指定
def x_error(qubit,i,j):
    qubit[0][i][j] = (qubit[0][i][j]+1)%2
def y_error(qubit,i,j):
    qubit[0][i][j] = (qubit[0][i][j]+1)%2    
    qubit[1][i][j] = (qubit[1][i][j]+1)%2
def z_error(qubit,i,j):
    qubit[1][i][j] =  (qubit[1][i][j]+1)%2 

def p_x_error(qubit,i,j,p):
    prob = random.random() 
    if prob < p:
        x_error(qubit,i,j)

def p_z_error(qubit,i,j,p):
    prob = random.random() 
    if prob < p:
        z_error(qubit,i,j)

# i番目にHadamard gateを作用させる
def H(qubit,i,j):
    qubit[0][i][j] = (qubit[0][i][j] + qubit[1][i][j]) % 2
    qubit[1][i][j] = (qubit[1][i][j] + qubit[0][i][j]) % 2
    qubit[0][i][j] = (qubit[0][i][j] + qubit[1][i][j]) % 2

#i番目がcontrolビット j 番目がtargetビットのCNOTgate
def CNOT(qubit_c,i,j,qubit_t,k,l):     #c, tには二次元[][]を代入する
    qubit_t[0][k][l] = (qubit_t[0][k][l] + qubit_c[0][i][j])%2 #コントロール側のXエラーはターゲットに
    qubit_c[1][i][j] = (qubit_c[1][i][j] + qubit_t[1][k][l])%2 #ターゲット側のZエラーはコントロールに

def rotated_surface_code(code_distance,p,p_m):

    qubits_d = np.zeros((2,code_distance,code_distance)) #データ量子ビットの格納
    qubits_d_Z = np.zeros((code_distance+1,code_distance,code_distance))
    qubits_m_in = np.zeros((2,code_distance-1,code_distance-1)) #測定量子ビット(中)の数
    qubits_m_out_X = np.zeros((2,2,int((code_distance-1)/2))) #測定量子ビット(外)の数

    syndrome_in = np.zeros((code_distance+2, code_distance-1, code_distance-1)) #シンドローム測定の回数+最初の状態のシンドローム+最後の測定から計算したシンドローム
    syndrome_out_X = np.zeros((code_distance+2,2,int((code_distance-1)/2)))

    #############  ループ部分  ##################

    for num in range(code_distance):

        ### 動的デカップリング (アイドリング中のエラー)
        for i in range(code_distance):
            for j in range(code_distance):
                p_z_error(qubits_d,i,j,p)
        
        ### シンドローム測定
        for i in range(code_distance-1):
            for j in range(code_distance-1):
                ### Xシンドローム
                # 内側
                if (i+j)%2 == 0: 
                    CNOT(qubits_m_in,i,j,qubits_d,i,j)
                    CNOT(qubits_m_in,i,j,qubits_d,i,j+1)
                    CNOT(qubits_m_in,i,j,qubits_d,i+1,j)
                    CNOT(qubits_m_in,i,j,qubits_d,i+1,j+1)
                # 外側
                if j == 0:
                    if i % 2 == 1:
                        CNOT(qubits_m_out_X,0,int((i-1)/2),qubits_d,i,j)
                        CNOT(qubits_m_out_X,0,int((i-1)/2),qubits_d,i+1,j)
                if j == code_distance-2:
                    if i % 2 == 0:
                        CNOT(qubits_m_out_X,1,int(i/2),qubits_d,i,code_distance-1)
                        CNOT(qubits_m_out_X,1,int(i/2),qubits_d,i+1,code_distance-1)
                """
                ### Zシンドローム 
                if (i+j) % 2 == 1: 
                    CNOT(qubits_d,i,j,qubits_m_in,i,j)
                    CNOT(qubits_d,i,j+1,qubits_m_in,i,j)
                    CNOT(qubits_d,i+1,j,qubits_m_in,i,j)
                    CNOT(qubits_d,i+1,j+1,qubits_m_in,i,j)
                """

        ### 測定結果の格納
        # 内側
        for i in range(code_distance-1):
            for j in range(code_distance-1):
                if (i+j)%2 == 0: ### Xシンドローム
                    p_z_error(qubits_m_in,i,j,p_m)
                    syndrome_in[num+1][i][j] =  qubits_m_in[1][i][j] # Zを格納
 
        # 外側
        for i in range(int((code_distance-1)/2)):
            p_z_error(qubits_m_out_X,0,i,p_m)
            syndrome_out_X[num+1][0][i] =  qubits_m_out_X[1][0][i] # 左
            p_z_error(qubits_m_out_X,1,i,p_m)
            syndrome_out_X[num+1][1][i] =  qubits_m_out_X[1][1][i] # 右

        #print("qubits_d= \n", qubits_d[1])
        #print("syndrome_in= \n", syndrome_in[num+1])
        #print("syndrome_out= \n", syndrome_out_X[num+1])

        #print("qubits_m_in= \n", qubits_m_in)
        #print("qubits_m_out= \n", qubits_m_out)

        ########################################
        for i in range(code_distance):
            for j in range(code_distance):
                qubits_d_Z[num+1][i][j] =  qubits_d[1][i][j]

        #############  初期化  ##################

        # 内側
        for i in range(code_distance-1):
            for j in range(code_distance-1):
                if (i+j)%2 == 0: ### Xシンドローム(Zエラー)
                    qubits_m_in[1][i][j] = 0
        # 外側
        for i in range(int((code_distance-1)/2)):
            ## Xのみやる
            qubits_m_out_X[1][0][i] = 0
            qubits_m_out_X[1][1][i] = 0

    ###################  ループ終了 #################

    #############  データビットの測定  ##################

    ### データ量子ビットをresult_dataに移す
    result_data_X = np.zeros((code_distance, code_distance))
    result_data_Z = np.zeros((code_distance, code_distance))
    for i in range(code_distance):
        for j in range(code_distance):
            result_data_Z[i][j] = qubits_d[1][i][j]
            result_data_X[i][j] = qubits_d[0][i][j]

    #print("result_data_X= \n", result_data_X)
    #print("result_data_Z= \n", result_data_Z)

    ### 測定結果からシンドロームを計算する
    # 内側
    for i in range(code_distance-1):
        for j in range(code_distance-1):
            if (i+j)%2 == 0: ### Xシンドローム
                syndrome_in[code_distance+1][i][j] =  (qubits_d[1][i][j]+qubits_d[1][i][j+1]+qubits_d[1][i+1][j]+qubits_d[1][i+1][j+1]) % 2
    # 外側
    for i in range(int((code_distance-1)/2)):
        syndrome_out_X[code_distance+1][0][i] = (qubits_d[1][2*i+1][0]+qubits_d[1][2*i+2][0]) % 2 # 左
        syndrome_out_X[code_distance+1][1][i] = (qubits_d[1][2*i][code_distance-1]+qubits_d[1][2*i+1][code_distance-1]) % 2 # 右

    #print("syndrome_in= \n", syndrome_in[code_distance+1])
    #print("syndrome_out= \n", syndrome_out_X[code_distance+1])

    #############  データビットの測定終了  ###############

    ############# detection eventの計算 ###############

    detection_event_in = np.zeros((code_distance+1, code_distance-1, code_distance-1))
    detection_event_out_X = np.zeros((code_distance+1, 2, int((code_distance-1)/2)))

    for num in range(code_distance+1):

        ### 内側
        for i in range(code_distance-1):
            for j in range(code_distance-1):
                detection_event_in[num,i,j] = (syndrome_in[num][i][j] + syndrome_in[num+1][i][j]) % 2
        ### 外側
        for i in range(int((code_distance-1)/2)):
            detection_event_out_X[num,0,i] = (syndrome_out_X[num,0,i] + syndrome_out_X[num+1,0,i]) % 2
            detection_event_out_X[num,1,i] = (syndrome_out_X[num,1,i] + syndrome_out_X[num+1,1,i]) % 2

    ############# detection eventの計算終了 ############
    dif_qubits_d_Z = np.zeros((code_distance,code_distance,code_distance))
    for num in range(code_distance):
        for i in range(code_distance):
            for j in range(code_distance):
                dif_qubits_d_Z[num][i][j] = (qubits_d_Z[num][i][j] + qubits_d_Z[num+1][i][j]) % 2

    #######################

    return detection_event_in, detection_event_out_X, result_data_Z, dif_qubits_d_Z
        
def sampling(code_distance,p,p_m):

    ############# 読み込み ################

    detection_event_in, detection_event_out, result_data, dif_qubits_d_Z = rotated_surface_code(code_distance,p,p_m)

    #print("input= \n", result_data)
    #print("detection_event_in= \n", detection_event_in)
    #print("detection_event_out= \n", detection_event_out)

    ############# detection_evemtを再構成 ################

    re_detection_event = np.zeros((code_distance+1,code_distance-1,code_distance+1))
    for num in range(code_distance+1):
        for i in range(code_distance-1):
            for j in range(code_distance+1):
                if j == 0: #左端
                    if i % 2 == 1:
                        if detection_event_out[num][0][int((i-1)/2)] == 1:
                            re_detection_event[num][i][j] = 1
                elif j == code_distance: #右端
                    if i % 2 == 0:
                        if detection_event_out[num][1][int((i)/2)] == 1:
                            re_detection_event[num][i][j] = 1
                else:
                    if detection_event_in[num][i][j-1] == 1:
                        re_detection_event[num][i][j] = 1

    #print("detection_event= \n",re_detection_event)

    ############# MWPM ################

    gp = nx.Graph()

    ############# 頂点の追加 ###############

    ### 内側
    for num in range(code_distance+1):
        for i in range(code_distance-1):
            for j in range(code_distance-1):
                if (i+j) % 2 == 0:
                    gp.add_node((num,i,j))

    ### 外側(Xシンドロームの追加)
    for num in range(code_distance+1):
        for i in range(code_distance-1):
            if i % 2 == 1:
                gp.add_node((num,i,-1))
            if i % 2 == 0:
                gp.add_node((num,i,code_distance-1))

    ### 外点
    gp.add_node('external')

    ############# 辺の追加 ###############

    ### 縦
    if p_m != 0:
        for num in range(code_distance):
            for i in range(code_distance-1):
                for j in range(-1,code_distance):
                    if (i+j) % 2 == 0:
                        gp.add_edge((num,i,j),(num+1,i,j),weight=-math.log(p_m))
    ### 横
    for num in range(code_distance+1):
        for i in range(code_distance-2):
            for j in range(-1,code_distance-1):
                if (i+j) % 2 == 0:
                    gp.add_edge((num,i,j),(num,i+1,j+1),weight=-math.log(p))
                if (i+j) % 2 == 1:
                    gp.add_edge((num,i+1,j),(num,i,j+1),weight=-math.log(p))

    ### 外点
    for num in range(code_distance+1):
        for j in range(-1,code_distance):
            if j == -1:
                gp.add_edge('external',(num,code_distance-2,j),weight=-math.log(p))
            elif j == code_distance-1:
                gp.add_edge('external',(num,0,j),weight=-math.log(p))
            elif j % 2 == 0:
                gp.add_edge('external',(num,0,j),weight=-math.log(p))
            elif j % 2 == 1:
                gp.add_edge('external',(num,code_distance-2,j),weight=-math.log(p))

    #nx.draw_networkx(gp)
    #plt.show()

    ########## シンドローム1の点の追加 ############

    edge_of_decoder_graph = []

    ### 内側
    for num in range(code_distance+1):
        for i in range(code_distance-1):
            for j in range(code_distance-1):
                if detection_event_in[num,i,j] == 1:
                    edge_of_decoder_graph.append((num,i,j))
    ### 外側
    for num in range(code_distance+1):
        for i in range(int((code_distance-1)/2)):
            if detection_event_out[num,1,i] == 1: #右
                edge_of_decoder_graph.append((num,2*i,code_distance-1))
            if detection_event_out[num,0,i] == 1: #左
                edge_of_decoder_graph.append((num,2*i+1,-1))
    ### 外点
    if len(edge_of_decoder_graph) % 2 == 1:
            edge_of_decoder_graph.append('external')

    ########## 最短距離の追加 ############

    mwpm_gp = nx.Graph() 

    ### 頂点の追加
    for v in range(len(edge_of_decoder_graph)):
        mwpm_gp.add_node(v)
    ### 辺の追加
    for i in range(len(edge_of_decoder_graph)):
        for j in range(i):
            shortest_path_weight = nx.dijkstra_path_length(gp, edge_of_decoder_graph[i],edge_of_decoder_graph[j])
            mwpm_gp.add_edge(i,j,weight = 10000000 - shortest_path_weight)

    ########## マッチング実行 ############
    mwpm_res = nx.max_weight_matching(mwpm_gp)
    match_path = []
    for match_pair in mwpm_res:
        match_path.append(nx.dijkstra_path(gp,edge_of_decoder_graph[match_pair[0]],edge_of_decoder_graph[match_pair[1]]))
    for path in match_path:
            #print(path)
            for i in range(len(path)): 
                if i !=0: #i=0は飛ばす
                    ### 外点がある場合
                    if path[i-1] == 'external': # pathの左='external'
                        if path[i][1] == 0: # 2番目の要素はy座標=0でここが外点とつながっているとき
                            result_data[0,path[i][2]] = (result_data[0,path[i][2]] + 1) % 2
                            #print("1(",0,path[i][2],")")
                        else: # 2番目の要素はy座標=code_distance-1でここが外点とつながっているとき
                            result_data[code_distance-1,path[i][2]+1] = (result_data[code_distance-1,path[i][2]+1] + 1) % 2
                            #print("2(",code_distance-1,path[i][2]+1,")")

                    elif path[i] == 'external': # pathの右='external'
                        if path[i-1][1] == 0: # 2番目の要素はy座標でここが外点とつながっているとき
                            result_data[0,path[i-1][2]] = (result_data[0,path[i-1][2]] + 1) % 2
                            #print("3(",0,path[i-1][2],")")
                        else: # 2番目の要素はy座標=code_distance-1でここが外点とつながっているとき
                            result_data[code_distance-1,path[i-1][2]+1] = (result_data[code_distance-1,path[i-1][2]+1] + 1) % 2
                            #print("4(",code_distance-1,path[i-1][2]+1,")")

                    ### numが同じ場合
                    elif path[i-1][0] == path[i][0]: 
                        result_data[min(path[i-1][1],path[i][1])+1,min(path[i-1][2],path[i][2])+1] = (result_data[min(path[i-1][1],path[i][1])+1,min(path[i-1][2],path[i][2])+1] + 1) % 2
                        #print("5(",min(path[i-1][1],path[i][1])+1,min(path[i-1][2],path[i][2])+1,")")

                    ### numが違う場合
                    #座標が同じ場合
                    elif path[i-1][1] == path[i][1] and path[i-1][2] == path[i][2]: 
                        continue

    ### Zシンドロームを繰り返すことによってエラーを左に集める
    Z_data = result_data.copy()
    for j in range(code_distance-1):
        if j % 2 == 0:
            for i in range(code_distance):
                if i == 0:
                    if Z_data[i,j] == 1:
                        Z_data[i,j] = (Z_data[i,j] + 1) %2
                        Z_data[i,j+1] = (Z_data[i,j+1] + 1) %2
                if i % 2 == 1:
                    if Z_data[i,j] == 1:
                        Z_data[i,j] = (Z_data[i,j] + 1) %2
                        Z_data[i,j+1] = (Z_data[i,j+1] + 1) %2
                        Z_data[i+1,j] = (Z_data[i+1,j] + 1) %2
                        Z_data[i+1,j+1] = (Z_data[i+1,j+1] + 1) %2
                if i % 2 == 0:
                    if Z_data[i,j] == 1:
                        Z_data[i,j] = (Z_data[i,j] + 1) %2
                        Z_data[i,j+1] = (Z_data[i,j+1] + 1) %2
                        Z_data[i-1,j] = (Z_data[i-1,j] + 1) %2
                        Z_data[i-1,j+1] = (Z_data[i-1,j+1] + 1) %2
        if j % 2 == 1:
            for i in range(code_distance):
                if i == code_distance-1:
                    if Z_data[i,j] == 1:
                        Z_data[i,j] = (Z_data[i,j] + 1) %2
                        Z_data[i,j+1] = (Z_data[i,j+1] + 1) %2
                if i % 2 == 0:
                    if Z_data[i,j] == 1:
                        Z_data[i,j] = (Z_data[i,j] + 1) %2
                        Z_data[i,j+1] = (Z_data[i,j+1] + 1) %2
                        Z_data[i+1,j] = (Z_data[i+1,j] + 1) %2
                        Z_data[i+1,j+1] = (Z_data[i+1,j+1] + 1) %2
                if i % 2 == 1:
                    if Z_data[i,j] == 1:
                        Z_data[i,j] = (Z_data[i,j] + 1) %2
                        Z_data[i,j+1] = (Z_data[i,j+1] + 1) %2
                        Z_data[i-1,j] = (Z_data[i-1,j] + 1) %2
                        Z_data[i-1,j+1] = (Z_data[i-1,j+1] + 1) %2

    ### 論理Zエラーがあるかの判定
    count = [0] * code_distance
    for i in range(code_distance): 
        for j in range(code_distance):
            if Z_data[i,j] == 1:
                count[i] = 1
                break
    judge = 0
    if count == [1] * code_distance:
        judge = 1

    """
    ### dtection eventの図示
    for num in range(code_distance):
        fig, ax = plt.subplots()
        ax.imshow(re_detection_event[num],cmap="copper")
        for i in range(code_distance):
            for j in range(code_distance):
                if (i+j)%2 == 0 and dif_qubits_d_Z[num][i][j] == 1:
                    X = []
                    Y = []
                    X.append(j+1)
                    X.append(j)
                    Y.append(i)
                    Y.append(i-1)
                    ax.plot(X,Y,color="b",lw=7)
                if (i+j)%2 == 1 and dif_qubits_d_Z[num][i][j] == 1:
                    X = []
                    Y = []
                    X.append(j+1)
                    X.append(j)
                    Y.append(i-1)
                    Y.append(i)
                    ax.plot(X,Y,color="b",lw=7)
        for path in match_path:
            for i in range(len(path)): 
                if i !=0: #i=0は飛ばす
                    ### 外点がある場合
                    if path[i-1] == 'external' and path[i][0] == num and path[i][1] == 0: # pathの左='external'
                        X = []
                        Y = []
                        X.append(path[i][2]+1)
                        Y.append(path[i][1])
                        X.append(path[i][2])
                        Y.append(path[i][1]-1)
                        ax.plot(X,Y,marker='o',color="r",lw=3,markersize=10)
                    elif path[i-1] == 'external' and path[i][0] == num and path[i][1] == code_distance-2: # pathの左='external'
                        X = []
                        Y = []
                        X.append(path[i][2]+1)
                        Y.append(path[i][1])
                        X.append(path[i][2]+2)
                        Y.append(path[i][1]+1)
                        ax.plot(X,Y,marker='o',color="r",lw=3,markersize=10)
                    elif path[i] == 'external' and path[i-1][0] == num and path[i-1][1] == 0: # pathの右='external'
                        X = []
                        Y = []
                        X.append(path[i-1][2]+1)
                        Y.append(path[i-1][1])
                        X.append(path[i-1][2])
                        Y.append(path[i-1][1]-1)
                        ax.plot(X,Y,marker='o',color="r",lw=3,markersize=10)
                    elif path[i] == 'external' and path[i-1][0] == num and path[i-1][1] == code_distance-2: # pathの右='external'
                        X = []
                        Y = []
                        X.append(path[i-1][2]+1)
                        Y.append(path[i-1][1])
                        X.append(path[i-1][2]+2)
                        Y.append(path[i-1][1]+1)
                        ax.plot(X,Y,marker='o',color="r",lw=3,markersize=10)
                    elif path[i-1][0] == path[i][0] and path[i][0] == num: 
                        X = []
                        Y = []
                        X.append(path[i-1][2]+1)
                        X.append(path[i][2]+1)
                        Y.append(path[i-1][1])
                        Y.append(path[i][1])
                        ax.plot(X,Y,marker='o',color="r",lw=3,markersize=10)
        ax.axis("off")
        plt.show()
        """

    return result_data, Z_data, judge

def count(trials,code_distance,p_div,pm,result_list):
    count = np.zeros((len(code_distance),len(p_div)))
    for _ in range(trials):
        num_d = 0
        for cd in code_distance:
            num_p =0
            for p in p_div:
                result_data, result, judge = sampling(cd,p/100,pm/100)
                #print("before= \n",result_data,"\nafter= \n",result, "\n", judge)
                if judge == 1:
                    count[num_d,num_p] += 1
                num_p += 1
            num_d += 1

    result_list.append(count/trials)

if __name__ == "__main__":

    ### パラメータ
    trials = 100
    p_s = 1
    p_e = 7
    p_d = 0.5
    d_s = 3
    d_e = 7
    d_d = 2
    pro = 1000
    code_distance = np.arange(d_s,d_e+1,d_d)
    p_div = np.arange(p_s,p_e+p_d,p_d)
    pm = 0

    # プロセスを管理する人。デラックスな共有メモリ
    manager = multiprocessing.Manager()
    # マネージャーからリストオブジェクトを取得
    result_list = manager.list()
    # あとでまとめてjoin()するためのプロセスのリスト
    processes = []
    # プロセスを生成
    for _ in range(pro):
        # マネージャーから取得したオブジェクトを引数に渡す
        process = multiprocessing.Process(target=count, args=(trials, code_distance, p_div,pm,result_list))
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

    df = pd.DataFrame(data=c, columns=p_div,index=code_distance)
    df.to_csv('pm='+str(pm)+',p=('+str(p_s)+','+str(p_e)+','+str(p_d)+'),d=('+str(d_s)+','+str(d_e)+','+str(d_d)+'),trials='+str(trials*pro)+'.csv')

    plt.rcParams["xtick.direction"] = "in"     
    plt.rcParams["ytick.direction"] = "in" 
    fig, ax = plt.subplots()
    num_d = 0
    for cd in code_distance:
        ax.plot(p_div,c[num_d]*100,marker='v',label="d ="+str(code_distance[num_d]))
        num_d += 1
    ax.set_xlabel("physical error rate (%)", fontsize=13)
    ax.set_ylabel("logical error rate (%)", fontsize=13)
    ax.set_ylim(0,)
    ax.set_xticks(p_div)
    ax.spines["top"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    ax.tick_params(direction="in", width=2, length=4, labelsize=12)
    ax.set_title("pm=" + str(pm) + "%, # of trials=" +str(trials*pro), fontsize=14)
    plt.legend()
    plt.savefig('pm='+str(pm)+',p=('+str(p_s)+','+str(p_e)+','+str(p_d)+'),d=('+str(d_s)+','+str(d_e)+','+str(d_d)+'),trials='+str(trials*pro)+ ".pdf")
