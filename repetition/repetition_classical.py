import random 
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import math
import time
import multiprocessing

###### ゲートやエラーの定義 #####
#エラーの定義
def x_error(qubit,i):
    qubit[0][i] ^= 1
def y_error(qubit,i):
    qubit[0][i] ^= 1    
    qubit[1][i] ^= 1
def z_error(qubit,i):
    qubit[1][i] ^=  1 

def single_biased(qubit,i,p,eta): # etaはバイアス
    p_x = p / (2*(1+eta))
    p_z = p * eta / (1 + eta) 
    prob = random.random() 
    if prob < p_z: #Z error
        z_error(qubit,i)
    elif prob < p_z+p_x: # X error
        x_error(qubit,i)
    elif prob < p+2*p_x: # Y error
        y_error(qubit,i)

def bitflip_error(qubit,i,p,eta): # etaはバイアス
    prob = random.random() 
    p_x = p / (2*(1+eta))
    if prob < 2*p_x: #Z error
        x_error(qubit,i)

def phaseflip_error(qubit,i,p,eta): # H後の測定でZエラー
    prob = random.random() 
    p_x = p / (2*(1+eta))
    if prob < 2*p_x: #Z error
        z_error(qubit,i)

# i番目にHadamard gateを作用させる
def H(qubit,i):
    qubit[0][i]^=qubit[1][i]
    qubit[1][i]^=qubit[0][i]
    qubit[0][i]^=qubit[1][i]

#i番目がcontrolビット j 番目がtargetビットのCNOTgate
def CNOT(qubit,c,t):
    qubit[0][t]^=qubit[0][c]
    qubit[1][c]^=qubit[1][t]

##### detection eventを作成する関数の定義 #####
def reptition(code_distance,rep,p,eta):
    nqubits = 2*code_distance-1
    qubit = [[0 for _ in range(nqubits)],[0 for _ in range(nqubits)]]

    D = np.zeros((code_distance-1,rep+2)) #シンドローム測定の回数+最初の状態のシンドローム+最後の測定から計算したシンドローム

    # 初期化のエラーなし
    ############  全てプラスに初期化  ##############
    # 測定を格納
    for i in range(code_distance-1):
        D[i][0] = qubit[1][2*i+1]   ### Zエラーがあるかは[1]
    for i in range(nqubits):
        H(qubit,i) 
        single_biased(qubit,i,p,eta) # Hゲート後のエラー
    #############################################
    #print(qubit)
    #############  ループ部分  ##################
    for i in range(rep):
        for j in range(nqubits-1):
            if j % 2 == 0:
                #single_biased(qubit,j,p,eta)
                CNOT(qubit,j+1,j)
                single_biased(qubit,j,p,eta)
                single_biased(qubit,j+1,p,eta)
            if j % 2 == 1:
                CNOT(qubit,j,j+1)
                single_biased(qubit,j,p,eta)
                single_biased(qubit,j+1,p,eta)
        for j in range(nqubits):
            if j % 2 == 0:
                single_biased(qubit,j,p,eta) #動的デカップリング部分

        # シンドローム測定
        #まず測定のbit反転
        for j in range(nqubits):
            if j % 2 == 1:
                #continue
                #H(qubit,i) 
                phaseflip_error(qubit,j,p,eta)
        #print(qubit)
        # 測定を格納
        for j in range(code_distance-1):
            D[j][i+1] = qubit[1][2*j+1]            #######要変更
        # 初期化&エラー
        for j in range(nqubits):
            if j % 2 == 1:
                #qubit[0][j] = 0   #######要変更
                qubit[1][j] = 0    ### X測定ならこっち
                single_biased(qubit,j,p,eta) # Hゲートを作用させたとして
                #phaseflip_error(qubit,j,p,eta)      ############   初期化失敗
    ############################################
    
    ##############  最後のデータビットを測定  ######
    result = [[],[]]
    for i in range(nqubits):
        if i % 2 == 0:
            phaseflip_error(qubit,i,p,eta) #測定前のHゲート
            result[0].append(qubit[0][i])
            result[1].append(qubit[1][i])
        if i % 2 == 1:
            result[0].append(qubit[0][i])
    #############################################
    #print(qubit)
    # データからシンドローム求める
    for i in range(code_distance-1):
        D[i][rep+1] = (result[1][i]+result[1][i+1])%2

    # detection eventの行列
    E = np.zeros((code_distance-1,rep+1))
    for i in range(code_distance-1):
        for j in range(rep+1):
            E[i,j] = (D[i,j] + D[i,j+1]) % 2

    #print("D= ", D.T)
    #print(result)
    #print("E= ", E.T)

    return E, result


##### 符号距離と、それ以下の距離での符号距離でのMWPM実行 #####
def sampling(E,result,code_distance,rep,p,eta):
    ### d_sのエラーはリスト中のd_s番目に保存
    count_z = [0]*(int((code_distance-1)/2))
    count_x = [0]*(int((code_distance-1)/2))
    for d_s in range(code_distance+1):
        if d_s == 1:
            continue
        if d_s % 2 == 0:
            continue

        #print("d_s=",d_s)

        E_re = E[0:d_s-1]
        result_re = [[],[]]
        result_re[0] = result[0][0:2*d_s-1]  ### 要変更
        result_re[1] = result[1][0:d_s]
        #print("E=",E_re.T)
        #print("result=",result_re)

        # 差分シンドロームが1のところは座標のデータを格納する
        edge_of_decoder_graph = []
        for i in range(d_s-1):
            for j in range(rep+1):
                if E_re[i,j] == 1:
                    edge_of_decoder_graph.append((i,j))
                    
        ### 最小距離のグラフの作成
        gp = nx.Graph()
        # 頂点の追加
        for i in range(d_s-1):
            for j in range(rep+1):
                gp.add_node((i,j))
        # 横辺の追加(反復方向)
        for i in range(d_s-1):
            for j in range(rep):
                gp.add_edge((i,j),(i,j+1),weight=-math.log(p*eta/(1+eta)))
        # 縦辺の追加(データ方向)
        for i in range(d_s-2):
            for j in range(rep+1):
                gp.add_edge((i,j),(i+1,j),weight=-math.log(p*eta/(1+eta)))
        # 斜め辺の追加(データ方向)
        for i in range(d_s-2):
            for j in range(rep):
                gp.add_edge((i,j),(i+1,j+1),weight=-math.log(p*eta/(1+eta)))
        #正方格子に外点を1つ加えておく（単点ではパリティを検出できないため、パリティoddになる頂点数が奇数になりうる）
        gp.add_node('external')
        for i in range(rep+1):
            gp.add_edge('external',(0,i),weight=-math.log(p*eta/(1+eta)))
            gp.add_edge('external',(d_s-2,i),weight=-math.log(p*eta/(1+eta)))

        #パリティoddの頂点数が奇数の場合は外点をdecoer graphに追加して頂点数を偶数に
        if len(edge_of_decoder_graph)%2==1:
            edge_of_decoder_graph.append('external')
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
        ##print(match_path)

        for path in match_path:
            for i in range(len(path)): 
                if i !=0: #i=0は飛ばす
                    if path[i-1] == 'external': # 左='external'
                        if path[i][0] == 0: #上側エラーなら
                            result_re[1][0] ^= 1 #上端を反転　　#### Zの方を訂正するなら一つ目の[]は[1]
                        else: #右端エラーなら
                            result_re[1][d_s-1]^= 1 #右端を反転
                            
                    elif path[i] == 'external': # 右='external'
                        if path[i-1][0] == 0:
                            result_re[1][0]^= 1 #上端を反転
                        else:
                            result_re[1][d_s-1]^= 1 #右端を反転
                    
                    elif path[i-1][1] == path[i][1]: #端のエラーではなく、同じサイクルでのエラーなら
                        result_re[1][min(path[i-1][0],path[i][0])+1] ^= 1
                    
                    elif path[i-1][0] == path[i][0]:
                        continue

                    else:
                        result_re[1][min(path[i-1][0],path[i][0])+1]^= 1

        ### 論理エラーのカウント
        # Zエラー
        if result_re[1] != [0]*d_s:
            count_z[int((d_s-3)/2)] +=1
        # Xエラー
        if sum(result_re[0])%2 == 1:
            count_x[int((d_s-3)/2)] += 1
        #print("result_re_X=",result_re[0],"result_re_Z=",result_re[1])
    count_z = np.array(count_z)
    count_x = np.array(count_x)
    return count_x, count_z


##### 実行するファイルの作成 #####
def repetiton_sampling(code_distance,rep,p,eta):

    ##### 距離を表すグラフの作成 #####
    #made_graph(code_distance,rep,p,eta)

    ##### 行列を導出 #####
    E, result = reptition(code_distance,rep,p,eta)
    return sampling(E,result,code_distance,rep,p,eta)



#### 実行条件 ####

###### 実行
def implement(code_distance,rep,p,eta,ex_num,result_list):
    count = np.zeros((2,int((code_distance-1)/2)))
    for _ in range(ex_num):
        a, b = repetiton_sampling(code_distance,rep,p,eta)
        count[0] += a
        count[1] += b
    count /= ex_num

    result_list.append(count)
    #return count, code_distance


if __name__ == "__main__":

    ### パラメータ
    code_distance=11
    rep=50
    p=0.005
    eta=1000
    trials=10
    pro = 100

    # プロセスを管理する人。デラックスな共有メモリ
    manager = multiprocessing.Manager()
    # マネージャーからリストオブジェクトを取得
    result_list = manager.list()
    # あとでまとめてjoin()するためのプロセスのリスト
    processes = []
    # プロセスを生成
    for _ in range(pro):
        # マネージャーから取得したオブジェクトを引数に渡す
        process = multiprocessing.Process(target=implement, args=(code_distance,rep,p,eta,trials,result_list))
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

    df = pd.DataFrame(data=c, columns=np.arange(3,code_distance+1,2),index=["LX","LZ"])
    df.to_csv('p='+str(p)+',eta='+str(eta)+',d='+str(code_distance)+',rep='+str(rep)+',trials='+str(trials*pro)+'.csv')

    plt.rcParams["xtick.direction"] = "in"     
    plt.rcParams["ytick.direction"] = "in" 
    fig, ax = plt.subplots()
    d = np.arange(3,code_distance+1,2)
    ax.plot(d,c[0]*100,marker='v',label="logical X error")
    ax.plot(d,c[1]*100,marker='v',label="logical Z error")
    ax.set_xlabel("physical error rate (%)", fontsize=13)
    ax.set_ylabel("logical error rate (%)", fontsize=13)
    ax.set_ylim(0,)
    ax.set_xticks(d)
    ax.spines["top"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    ax.tick_params(direction="in", width=2, length=4, labelsize=12)
    ax.set_title('p='+str(p)+',eta='+str(eta)+',rep='+str(rep)+',trials='+str(trials*pro), fontsize=14)
    plt.legend()
    plt.savefig('p='+str(p)+',eta='+str(eta)+',d='+str(code_distance)+',rep='+str(rep)+',trials='+str(trials*pro)+'.pdf')
