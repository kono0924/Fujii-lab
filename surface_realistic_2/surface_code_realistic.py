import random 
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import math
import os
import multiprocessing

from difflib import Match
import random
from re import A 
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import math

#エラーの定義 qubitは三次元配列で1つ目のインデックスでXかZか、2,3個目のインデックスで位置を指定
def x_error(qubit,i,j):
    qubit[0][i][j] = (qubit[0][i][j]+1)%2
def y_error(qubit,i,j):
    qubit[0][i][j] = (qubit[0][i][j]+1)%2    
    qubit[1][i][j] = (qubit[1][i][j]+1)%2
def z_error(qubit,i,j):
    qubit[1][i][j] =  (qubit[0][i][j]+1)%2 

def single_biased(qubit,i,j,p): # etaはバイアス
    p_x = p 
    p_z = p 
    prob = random.random() 
    if prob < p_z: #Z error
        z_error(qubit,i,j)
    elif prob < p_z+p_x: # X error
        x_error(qubit,i,j)
    elif prob < p_z+2*p_x: # Y error
        y_error(qubit,i,j)

def reversed_single_biased(qubit,i,j,p): # 表面符号と反復符号で逆になっているエラー
    p_x = p 
    p_z = p 
    prob = random.random() 
    if prob < p_z: #Z error
        x_error(qubit,i,j)
    elif prob < p_z+p_x: # X error
        z_error(qubit,i,j)
    elif prob < p_z+2*p_x: # Y error
        y_error(qubit,i,j)

def bitflip_error(qubit,i,j,p): # etaはバイアス
    p_x = p 
    p_z = p 
    prob = random.random() 
    if prob < 2*p_x: #Z error
        x_error(qubit,i,j)

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

def rotated_surface_code(code_distance,p_list):

    qubits_d = np.zeros((2,code_distance,code_distance)) #データ量子ビットの格納
    qubits_m_in = np.zeros((2,code_distance-1,code_distance-1)) #測定量子ビット(中)の数
    qubits_m_out = np.zeros((2,4,int((code_distance-1)/2))) #測定量子ビット(外)の数

    syndrome_in_X = np.zeros((code_distance+2, code_distance-1, code_distance-1)) #シンドローム測定の回数+最初の状態のシンドローム+最後の測定から計算したシンドローム
    syndrome_in_Z = np.zeros((code_distance+2, code_distance-1, code_distance-1))
    syndrome_out_X = np.zeros((code_distance+2,2,int((code_distance-1)/2)))
    syndrome_out_Z = np.zeros((code_distance+2,2,int((code_distance-1)/2)))

    #############  ループ部分  ##################

    for num in range(code_distance):

        ### 反復符号でのエラー
        for i in range(code_distance):
            for j in range(code_distance):
                #p_x_error(qubits_d,i,j,p_list[0])
                p_z_error(qubits_d,i,j,p_list[1])
        
        for i in range(code_distance-1):
            for j in range(code_distance-1):
                ### Xシンドローム
                # 内側
                if (i+j)%2 == 0: 
                    CNOT(qubits_m_in,i,j,qubits_d,i,j)
                    p_x_error(qubits_d,i,j,p_list[2])
                    p_z_error(qubits_d,i,j,p_list[3])
                    p_z_error(qubits_m_in,i,j,p_list[4])
                    CNOT(qubits_m_in,i,j,qubits_d,i,j+1)
                    p_x_error(qubits_d,i,j+1,p_list[2])
                    p_z_error(qubits_d,i,j+1,p_list[3])
                    p_z_error(qubits_m_in,i,j,p_list[4])
                    CNOT(qubits_m_in,i,j,qubits_d,i+1,j)
                    p_x_error(qubits_d,i+1,j,p_list[2])
                    p_z_error(qubits_d,i+1,j,p_list[3])
                    p_z_error(qubits_m_in,i,j,p_list[4])
                    CNOT(qubits_m_in,i,j,qubits_d,i+1,j+1)
                    p_x_error(qubits_d,i+1,j+1,p_list[2])
                    p_z_error(qubits_d,i+1,j+1,p_list[3])
                    p_z_error(qubits_m_in,i,j,p_list[4])
                # 外側
                if j == 0:
                    if i % 2 == 1:
                        CNOT(qubits_m_out,0,int((i-1)/2),qubits_d,i,j)
                        p_x_error(qubits_d,i,j,p_list[2])
                        p_z_error(qubits_d,i,j,p_list[3])
                        p_z_error(qubits_m_out,0,int((i-1)/2),p_list[4])
                        CNOT(qubits_m_out,0,int((i-1)/2),qubits_d,i+1,j)
                        p_x_error(qubits_d,i+1,j,p_list[2])
                        p_z_error(qubits_d,i+1,j,p_list[3])
                        p_z_error(qubits_m_out,0,int((i-1)/2),p_list[4])
                if j == code_distance-2:
                    if i % 2 == 0:
                        CNOT(qubits_m_out,1,int(i/2),qubits_d,i,code_distance-1)
                        p_x_error(qubits_d,i,code_distance-1,p_list[2])
                        p_z_error(qubits_d,i,code_distance-1,p_list[3])
                        p_z_error(qubits_m_out,1,int(i/2),p_list[4])
                        CNOT(qubits_m_out,1,int(i/2),qubits_d,i+1,code_distance-1)
                        p_x_error(qubits_d,i+1,code_distance-1,p_list[2])
                        p_z_error(qubits_d,i+1,code_distance-1,p_list[3])
                        p_z_error(qubits_m_out,1,int(i/2),p_list[4])        
                ### Zシンドローム
                # 内側
                if (i+j)%2 == 1: 
                    CNOT(qubits_d,i,j,qubits_m_in,i,j)
                    p_x_error(qubits_d,i,j,p_list[6])
                    p_z_error(qubits_d,i,j,p_list[5])
                    p_x_error(qubits_m_in,i,j,p_list[6])
                    p_z_error(qubits_m_in,i,j,p_list[5])
                    CNOT(qubits_d,i,j+1,qubits_m_in,i,j)
                    p_x_error(qubits_d,i,j+1,p_list[6])
                    p_z_error(qubits_d,i,j+1,p_list[5])
                    p_x_error(qubits_m_in,i,j,p_list[6])
                    p_z_error(qubits_m_in,i,j,p_list[5])
                    CNOT(qubits_d,i+1,j,qubits_m_in,i,j)
                    p_x_error(qubits_d,i+1,j,p_list[6])
                    p_z_error(qubits_d,i+1,j,p_list[5])
                    p_x_error(qubits_m_in,i,j,p_list[6])
                    p_z_error(qubits_m_in,i,j,p_list[5])
                    CNOT(qubits_d,i+1,j+1,qubits_m_in,i,j)
                    p_x_error(qubits_d,i+1,j+1,p_list[6])
                    p_z_error(qubits_d,i+1,j+1,p_list[5])
                    p_x_error(qubits_m_in,i,j,p_list[6])
                    p_z_error(qubits_m_in,i,j,p_list[5])
                # 外側
                if i == 0:
                    if j % 2 == 0:
                        CNOT(qubits_d,i,j,qubits_m_out,0,int(i/2))
                        p_x_error(qubits_d,i,j,p_list[6])
                        p_z_error(qubits_d,i,j,p_list[5])
                        p_x_error(qubits_m_out,2,int(i/2),p_list[6])
                        p_z_error(qubits_m_out,2,int(i/2),p_list[5])
                        CNOT(qubits_d,i,j+1,qubits_m_out,0,int(i/2))
                        p_x_error(qubits_d,i,j+1,p_list[6])
                        p_z_error(qubits_d,i,j+1,p_list[5])
                        p_x_error(qubits_m_out,2,int(i/2),p_list[6])
                        p_z_error(qubits_m_out,2,int(i/2),p_list[5])
                if i == code_distance-2:
                    if j % 2 == 1:
                        CNOT(qubits_d,i,code_distance-1,qubits_m_out,1,int((i-1)/2))
                        p_x_error(qubits_d,i,code_distance-1,p_list[6])
                        p_z_error(qubits_d,i,code_distance-1,p_list[5])
                        p_x_error(qubits_m_out,3,int((i-1)/2),p_list[6])
                        p_z_error(qubits_m_out,3,int((i-1)/2),p_list[5])
                        CNOT(qubits_d,i+1,code_distance-1,qubits_m_out,1,int((i-1)/2))
                        p_x_error(qubits_d,i+1,code_distance-1,p_list[6])
                        p_z_error(qubits_d,i+1,code_distance-1,p_list[5])
                        p_x_error(qubits_m_out,3,int((i-1)/2),p_list[6])
                        p_z_error(qubits_m_out,3,int((i-1)/2),p_list[5])

        ### 測定結果の格納 & 初期化
        ## Xシンドローム
        # 内側
        for i in range(code_distance-1):
            for j in range(code_distance-1):
                if (i+j)%2 == 0: ### Xシンドローム
                    syndrome_in_X[num+1][i][j] =  qubits_m_in[1][i][j] # Zを格納
                    qubits_m_in[1][i][j] = 0
        # 外側
        for i in range(int((code_distance-1)/2)):
            syndrome_out_X[num+1][0][i] =  qubits_m_out[1][0][i] # 左
            qubits_m_out[1][0][i] = 0
            syndrome_out_X[num+1][1][i] =  qubits_m_out[1][1][i] # 右
            qubits_m_out[1][1][i] = 0

        ## Zシンドローム
        # 内側
        for i in range(code_distance-1):
            for j in range(code_distance-1):
                if (i+j)%2 == 1: ### Xシンドローム
                    p_x_error(qubits_m_in,i,j,p_list[6])
                    syndrome_in_Z[num+1][i][j] =  qubits_m_in[0][i][j] # Xを格納
                    qubits_m_in[0][i][j] = 0
        # 外側
        for i in range(int((code_distance-1)/2)):
            p_x_error(qubits_m_out,2,i,p_list[6])
            syndrome_out_Z[num+1][0][i] =  qubits_m_out[0][2][i] # 上
            qubits_m_out[0][2][i] = 0
            p_x_error(qubits_m_out,3,i,p_list[6])
            syndrome_out_Z[num+1][1][i] =  qubits_m_out[0][3][i] # 下
            qubits_m_out[0][3][i] = 0

    ###################  ループ終了 #################

    #############  データビットの測定  ##################

    ### データ量子ビットをresult_dataに移す
    result_data_Z = np.zeros((code_distance, code_distance))
    for i in range(code_distance):
        for j in range(code_distance):
            result_data_Z[i][j] = qubits_d[1][i][j]
    result_data_X = np.zeros((code_distance, code_distance))
    for i in range(code_distance):
        for j in range(code_distance):
            result_data_X[i][j] = qubits_d[0][i][j]

    ### 測定結果からシンドロームを計算する
    ### Xシンドローム
    # 内側
    for i in range(code_distance-1):
        for j in range(code_distance-1):
            if (i+j)%2 == 0: 
                syndrome_in_X[code_distance+1][i][j] =  (qubits_d[1][i][j]+qubits_d[1][i][j+1]+qubits_d[1][i+1][j]+qubits_d[1][i+1][j+1]) % 2
    # 外側
    for i in range(int((code_distance-1)/2)):
        # 右
        syndrome_out_X[code_distance+1][1][i] = (qubits_d[1][2*i][code_distance-1]+qubits_d[1][2*i+1][code_distance-1]) % 2
        # 左
        syndrome_out_X[code_distance+1][0][i] = (qubits_d[1][2*i+1][0]+qubits_d[1][2*i+2][0]) % 2
    ### Zシンドローム
    # 内側
    for i in range(code_distance-1):
        for j in range(code_distance-1):
            if (i+j)%2 == 1: 
                syndrome_in_Z[code_distance+1][i][j] =  (qubits_d[0][i][j]+qubits_d[0][i][j+1]+qubits_d[0][i+1][j]+qubits_d[0][i+1][j+1]) % 2
    # 外側
    for i in range(int((code_distance-1)/2)):
        # 上
        syndrome_out_Z[code_distance+1][0][i] = (qubits_d[0][0][2*i]+qubits_d[0][0][2*i+1]) % 2
        # 下
        syndrome_out_Z[code_distance+1][1][i] = (qubits_d[0][code_distance-1][2*i+1]+qubits_d[0][code_distance-1][2*i+2]) % 2

    #############  データビットの測定終了  ###############

    ############# detection eventの計算 ###############

    detection_event_in = np.zeros((code_distance+1, code_distance-1, code_distance-1))
    detection_event_out_X = np.zeros((code_distance+1, 2, int((code_distance-1)/2)))
    detection_event_out_Z = np.zeros((code_distance+1, 2, int((code_distance-1)/2)))

    for num in range(code_distance+1):
        ### 内側
        for i in range(code_distance-1):
            for j in range(code_distance-1):
                if (i+j) % 2 == 0:
                    detection_event_in[num,i,j] = (syndrome_in_X[num][i][j] + syndrome_in_X[num+1][i][j]) % 2
                if (i+j) % 2 == 1:
                    detection_event_in[num,i,j] = (syndrome_in_Z[num][i][j] + syndrome_in_Z[num+1][i][j]) % 2
        ### 外側
        for i in range(int((code_distance-1)/2)):
            detection_event_out_X[num,0,i] = (syndrome_out_X[num,0,i] + syndrome_out_X[num+1,0,i]) % 2
            detection_event_out_X[num,1,i] = (syndrome_out_X[num,1,i] + syndrome_out_X[num+1,1,i]) % 2
            detection_event_out_Z[num,0,i] = (syndrome_out_Z[num,0,i] + syndrome_out_Z[num+1,0,i]) % 2
            detection_event_out_Z[num,1,i] = (syndrome_out_Z[num,1,i] + syndrome_out_Z[num+1,1,i]) % 2

    ############# detection eventの計算終了 ############
    #print(syndrome_in_Z,"\n",syndrome_out_Z,"\n",detection_event_out_Z)

    return detection_event_in, detection_event_out_X, result_data_Z, detection_event_out_Z, result_data_X
        
def sampling(code_distance,p_list):

    ############# 読み込み ################

    detection_event_in, detection_event_out_X, result_data_Z,  detection_event_out_Z, result_data_X = rotated_surface_code(code_distance,p_list)

    ############# detection_evemtを再構成 ################

    re_detection_event_X = np.zeros((code_distance+1,code_distance-1,code_distance+1))
    for num in range(code_distance+1):
        for i in range(code_distance-1):
            for j in range(code_distance+1):
                if j == 0: #左端
                    if i % 2 == 1:
                        if detection_event_out_X[num][0][int((i-1)/2)] == 1:
                            re_detection_event_X[num][i][j] = 1
                elif j == code_distance: #右端
                    if i % 2 == 0:
                        if detection_event_out_X[num][1][int((i)/2)] == 1:
                            re_detection_event_X[num][i][j] = 1
                else:
                    if detection_event_in[num][i][j-1] == 1 and (i+j)%2==1:
                        re_detection_event_X[num][i][j] = 1

    #print("detection_event= \n",re_detection_event_X)
    #print("result_data_before_match=\n",result_data_Z)

    re_detection_event_Z = np.zeros((code_distance+1,code_distance+1,code_distance-1))
    for num in range(code_distance+1):
        for i in range(code_distance+1):
            for j in range(code_distance-1):
                if i == 0: #上端
                    if j % 2 == 0:
                        if detection_event_out_Z[num][0][int((j)/2)] == 1:
                            re_detection_event_Z[num][i][j] = 1
                elif i == code_distance: #下端
                    if j % 2 == 1:
                        if detection_event_out_Z[num][1][int((j-1)/2)] == 1:
                            re_detection_event_Z[num][i][j] = 1
                else:
                    if detection_event_in[num][i-1][j] == 1 and (i+j)%2==0:
                        re_detection_event_Z[num][i][j] = 1

    #print("detection_event= \n",re_detection_event_Z)
    #print("result_data_before_match=\n",result_data_X)

    ############# MWPM ################

    gp_X = nx.Graph()
    gp_Z = nx.Graph()

    ############# 頂点の追加 ###############

    ### 内側
    for num in range(code_distance+1):
        for i in range(code_distance-1):
            for j in range(code_distance-1):
                if (i+j)%2 == 0:
                    gp_X.add_node((num,i,j))
                if (i+j)%2 == 1:
                    gp_Z.add_node((num,i,j))

    ### 外側(Xシンドロームの追加)
    for num in range(code_distance+1):
        for i in range(int(code_distance-1)):
            if i % 2 == 1:
                gp_X.add_node((num,i,-1))
                gp_Z.add_node((num,code_distance-1,i))
            if i % 2 == 0:
                gp_X.add_node((num,i,code_distance-1))
                gp_Z.add_node((num,-1,i))

    ### 外点
    gp_X.add_node('external_X')
    gp_Z.add_node('external_Z')

    ############# 辺の追加 ###############

    ### 縦
    for num in range(code_distance):
        ### 内側
        for i in range(code_distance-1):
            for j in range(-1,code_distance):
                if (i+j) % 2 == 0:
                    gp_X.add_edge((num,i,j),(num+1,i,j),weight=-math.log(4*p_list[4]))
                if (i+j) % 2 == 1:
                    gp_Z.add_edge((num,i,j),(num+1,i,j),weight=-math.log(5*p_list[6]))
        ### 外側
        for i in range(code_distance-1):
            if i % 2 == 0:
                gp_X.add_edge((num,i,code_distance-1),(num+1,i,code_distance-1),weight=-math.log(2*p_list[4]))
                gp_Z.add_edge((num,-1,i),(num+1,-1,i),weight=-math.log(3*p_list[6]))
            if i % 2 == 1:
                gp_X.add_edge((num,i,-1),(num+1,i,-1),weight=-math.log(2*p_list[4]))
                gp_Z.add_edge((num,code_distance-1,i),(num+1,code_distance-1,i),weight=-math.log(3*p_list[6]))
    ### 横
    for num in range(code_distance+1):
        ### Xシンドローム
        for i in range(code_distance-2):
            for j in range(-1,code_distance-1):
                if num == 0:
                    if (i+j) % 2 == 0:
                        gp_X.add_edge((num,i,j),(num,i+1,j+1),weight=-math.log(p_list[1]))
                    if (i+j) % 2 == 1:
                        gp_X.add_edge((num,i+1,j),(num,i,j+1),weight=-math.log(p_list[1]))
                elif num == code_distance:
                    if (i+j) % 2 == 0:
                        gp_X.add_edge((num,i,j),(num,i+1,j+1),weight=-math.log(2*p_list[5]+p_list[3]))
                    if (i+j) % 2 == 1:
                        gp_X.add_edge((num,i+1,j),(num,i,j+1),weight=-math.log(2*p_list[5]+p_list[3]))
                else:
                    if (i+j) % 2 == 0:
                        gp_X.add_edge((num,i,j),(num,i+1,j+1),weight=-math.log(2*p_list[5]+p_list[1]+p_list[3]))
                    if (i+j) % 2 == 1:
                        gp_X.add_edge((num,i+1,j),(num,i,j+1),weight=-math.log(2*p_list[5]+p_list[1]+p_list[3]))
        ### Zシンドローム
        for i in range(-1,code_distance-1):
            for j in range(code_distance-2):
                if num == 0:
                    if (i+j) % 2 == 1:
                        gp_Z.add_edge((num,i,j),(num,i+1,j+1),weight=-math.log(p_list[0]+2*p_list[2]))
                    if (i+j) % 2 == 0:
                        gp_Z.add_edge((num,i+1,j),(num,i,j+1),weight=-math.log(p_list[0]+2*p_list[2]))
                elif num == code_distance:
                    if (i+j) % 2 == 1:
                        gp_Z.add_edge((num,i,j),(num,i+1,j+1),weight=-math.log(p_list[6]))
                    if (i+j) % 2 == 0:
                        gp_Z.add_edge((num,i+1,j),(num,i,j+1),weight=-math.log(p_list[6]))
                else:
                    if (i+j) % 2 == 1:
                        gp_Z.add_edge((num,i,j),(num,i+1,j+1),weight=-math.log(p_list[6]+p_list[0]+2*p_list[2]))
                    if (i+j) % 2 == 0:
                        gp_Z.add_edge((num,i+1,j),(num,i,j+1),weight=-math.log(p_list[6]+p_list[0]+2*p_list[2]))

    ### 斜め
    for num in range(code_distance):
        ########## Xシンドローム
        ### 内側
        for i in range(code_distance-2):
            for j in range(code_distance-2):
                if num == 0:
                    continue
                else:
                    if (i+j) % 2 == 0:
                        gp_X.add_edge((num+1,i,j),(num,i+1,j+1),weight=-math.log(p_list[3]))
                    if (i+j) % 2 == 1:
                        gp_X.add_edge((num+1,i+1,j),(num,i,j+1),weight=-math.log(p_list[3]))
        ### 外側
        for i in range(code_distance-2):
            if i % 2 == 0:
                gp_X.add_edge((num,i,code_distance-1),(num+1,i+1,code_distance-2),weight=-math.log(p_list[3]))
                gp_X.add_edge((num,i+1,-1),(num+1,i,0),weight=-math.log(p_list[3]))
            if i % 2 == 1:
                gp_X.add_edge((num,i+1,code_distance-1),(num+1,i,code_distance-2),weight=-math.log(p_list[3]))
                gp_X.add_edge((num,i,-1),(num+1,i+1,0),weight=-math.log(p_list[3]))
        ########## Zシンドローム
        ### 内側
        for i in range(code_distance-2):
            for j in range(code_distance-2):
                if num == 0:
                    continue
                else:
                    if (i+j) % 2 == 1:
                        gp_Z.add_edge((num+1,i,j),(num,i+1,j+1),weight=-math.log(p_list[6]))
                    if (i+j) % 2 == 0:
                        gp_Z.add_edge((num+1,i+1,j),(num,i,j+1),weight=-math.log(p_list[6]))
        ### 外側
        for i in range(code_distance-2):
            if i % 2 == 0:
                gp_Z.add_edge((num,i,code_distance-1),(num+1,i+1,code_distance-2),weight=-math.log(p_list[6]))
                gp_Z.add_edge((num,i+1,-1),(num+1,i,0),weight=-math.log(2*p_list[6]))
            if i % 2 == 1:
                gp_Z.add_edge((num,i+1,code_distance-1),(num+1,i,code_distance-2),weight=-math.log(p_list[6]))
                gp_Z.add_edge((num,i,-1),(num+1,i+1,0),weight=-math.log(2*p_list[6]))

    ### 外点
    for num in range(code_distance+1):
        for j in range(-1,code_distance):
            if j % 2 == 0:
                gp_X.add_edge('external_X',(num,0,j),weight=-math.log(2*p_list[3]))
            if j % 2 == 1:
                gp_X.add_edge('external_X',(num,code_distance-2,j),weight=-math.log(2*p_list[3]))
        for i in range(-1,code_distance):
            if i % 2 == 0:
                gp_Z.add_edge('external_Z',(num,i,code_distance-2),weight=-math.log(2*p_list[6]))
            if i % 2 == 1:
                gp_Z.add_edge('external_Z',(num,i,0),weight=-math.log(2*p_list[6]))

    ########## シンドローム1の点の追加 ############

    edge_of_decoder_graph_X = []

    ### 内側
    for num in range(code_distance+1):
        for i in range(code_distance-1):
            for j in range(code_distance-1):
                if detection_event_in[num,i,j] == 1 and (i+j)%2==0 :
                    edge_of_decoder_graph_X.append((num,i,j)) 

    ### 外側
    for num in range(code_distance+1):
        for i in range(int((code_distance-1)/2)):
            if detection_event_out_X[num,1,i] == 1:
                edge_of_decoder_graph_X.append((num,2*i,code_distance-1))
            if detection_event_out_X[num,0,i] == 1:
                edge_of_decoder_graph_X.append((num,2*i+1,-1))

    ### 外点
    if len(edge_of_decoder_graph_X)%2==1:
            edge_of_decoder_graph_X.append('external_X')

    ########## 最短距離の追加 ############

    mwpm_gp = nx.Graph() 

    ### 頂点の追加
    for v in range(len(edge_of_decoder_graph_X)):
        mwpm_gp.add_node(v)
    ### 辺の追加
    for i in range(len(edge_of_decoder_graph_X)):
        for j in range(i):
            shortest_path_weight = nx.dijkstra_path_length(gp_X, edge_of_decoder_graph_X[i],edge_of_decoder_graph_X[j])
            mwpm_gp.add_edge(i,j,weight = 100000000 - shortest_path_weight)

    ########## マッチング実行 ############
    mwpm_res = nx.max_weight_matching(mwpm_gp)
    match_path = []
    for match_pair in mwpm_res:
        match_path.append(nx.dijkstra_path(gp_X,edge_of_decoder_graph_X[match_pair[0]],edge_of_decoder_graph_X[match_pair[1]]))
    for path in match_path:
        #print(path)
        for i in range(len(path)): 
            if i !=0: #i=0は飛ばす
                ### 外点がある場合
                if path[i-1] == 'external_X': # pathの左='external'
                    if path[i][1] == 0: # 2番目の要素はy座標=0でここが外点とつながっているとき
                        result_data_Z[0,path[i][2]] = (result_data_Z[0,path[i][2]] + 1) % 2
                    else: # 2番目の要素はy座標=code_distance-1でここが外点とつながっているとき
                        result_data_Z[code_distance-1,path[i][2]+1] = (result_data_Z[code_distance-1,path[i][2]+1] + 1) % 2
                elif path[i] == 'external_X': # pathの右='external'
                    if path[i-1][1] == 0: # 2番目の要素はy座標でここが外点とつながっているとき
                        result_data_Z[0,path[i-1][2]] = (result_data_Z[0,path[i-1][2]] + 1) % 2
                    else: # 2番目の要素はy座標=code_distance-1でここが外点とつながっているとき
                        result_data_Z[code_distance-1,path[i-1][2]+1] = (result_data_Z[code_distance-1,path[i-1][2]+1] + 1) % 2
                ### numが同じ場合
                elif path[i-1][0] == path[i][0]: 
                    result_data_Z[min(path[i-1][1],path[i][1])+1,min(path[i-1][2],path[i][2])+1] = (result_data_Z[min(path[i-1][1],path[i][1])+1,min(path[i-1][2],path[i][2])+1] + 1) % 2
                ### numが違う場合
                #座標が同じ場合
                elif path[i-1][1] == path[i][1] and path[i-1][2] == path[i][2]: 
                    continue
                #座標が違う場合
                else:
                    result_data_Z[min(path[i-1][1],path[i][1])+1,min(path[i-1][2],path[i][2])+1] = (result_data_Z[min(path[i-1][1],path[i][1])+1,min(path[i-1][2],path[i][2])+1] + 1) % 2


    ### Zシンドロームを繰り返すことによってエラーを左に集める
    Z_data = result_data_Z.copy()
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
    judge_X = 0
    if count == [1] * code_distance:
        judge_X = 1

    #################################################################################################################################################################################################################################
    # ここからはZシンドローム
    ########## シンドローム1の点の追加 ############

    edge_of_decoder_graph_Z = []

    ### 内側
    for num in range(code_distance+1):
        for i in range(code_distance-1):
            for j in range(code_distance-1):
                if detection_event_in[num,i,j] == 1 and (i+j)%2 ==1 :
                    edge_of_decoder_graph_Z.append((num,i,j))

    ### 外側
    for num in range(code_distance+1):
        for i in range(int((code_distance-1)/2)):
            if detection_event_out_Z[num,1,i] == 1:
                edge_of_decoder_graph_Z.append((num,code_distance-1,2*i+1))
            if detection_event_out_Z[num,0,i] == 1:
                edge_of_decoder_graph_Z.append((num,-1,2*i))

    ### 外点
    if len(edge_of_decoder_graph_Z)%2==1:
            edge_of_decoder_graph_Z.append('external_Z')
    ########## 最短距離の追加 ############

    mwpm_gp = nx.Graph() 

    ### 頂点の追加
    for v in range(len(edge_of_decoder_graph_Z)):
        mwpm_gp.add_node(v)
    ### 辺の追加
    for i in range(len(edge_of_decoder_graph_Z)):
        for j in range(i):
            shortest_path_weight = nx.dijkstra_path_length(gp_Z, edge_of_decoder_graph_Z[i],edge_of_decoder_graph_Z[j])
            mwpm_gp.add_edge(i,j,weight = 100000000 - shortest_path_weight)

    ########## マッチング実行 ############
    mwpm_res = nx.max_weight_matching(mwpm_gp)
    match_path = []
    for match_pair in mwpm_res:
        match_path.append(nx.dijkstra_path(gp_Z,edge_of_decoder_graph_Z[match_pair[0]],edge_of_decoder_graph_Z[match_pair[1]]))
    #print(match_path)
    for path in match_path:
            for i in range(len(path)): 
                if i !=0: #i=0は飛ばす
                    ### 外点がある場合
                    if path[i-1] == 'external_Z': # pathの左='external'
                        if path[i][2] == 0: # 2番目の要素はx座標=0でここが外点とつながっているとき
                            result_data_X[path[i][1]+1,0] = (result_data_X[path[i][1]+1,0] + 1) % 2
                        else: # 2番目の要素はy座標=code_distance-1でここが外点とつながっているとき
                            result_data_X[path[i][1],code_distance-1] = (result_data_X[path[i][1],code_distance-1] + 1) % 2
                    elif path[i] == 'external_Z': # pathの右='external'
                        if path[i-1][2] == 0: # 2番目の要素はx座標でここが外点とつながっているとき
                            result_data_X[path[i-1][1]+1,0] = (result_data_X[path[i-1][1]+1,0] + 1) % 2
                        else: # 2番目の要素はy座標=code_distance-1でここが外点とつながっているとき
                            result_data_X[path[i-1][1],code_distance-1] = (result_data_X[path[i-1][1],code_distance-1] + 1) % 2
                    ### numが同じ場合
                    elif path[i-1][0] == path[i][0]: 
                        result_data_X[min(path[i-1][1],path[i][1])+1,min(path[i-1][2],path[i][2])+1] = (result_data_X[min(path[i-1][1],path[i][1])+1,min(path[i-1][2],path[i][2])+1] + 1) % 2
                    ### numが違う場合
                    #座標が同じ場合
                    elif path[i-1][1] == path[i][1] and path[i-1][2] == path[i][2]: 
                        continue
                    else:
                        result_data_X[min(path[i-1][1],path[i][1])+1,min(path[i-1][2],path[i][2])+1] = (result_data_X[min(path[i-1][1],path[i][1])+1,min(path[i-1][2],path[i][2])+1] + 1) % 2

    ### Zシンドロームを繰り返すことによってエラーを左に集める
    X_data = result_data_X.copy()
    for i in range(code_distance-1):
        for j in range(code_distance):
            if i % 2 == 0:
                if j == code_distance-1:
                    if X_data[i,j] == 1:
                        X_data[i,j] = (X_data[i,j] + 1) %2
                        X_data[i+1,j] = (X_data[i+1,j] + 1) %2
                elif j % 2 == 0:
                    if X_data[i,j] == 1:
                        X_data[i,j] = (X_data[i,j] + 1) %2
                        X_data[i,j+1] = (X_data[i,j+1] + 1) %2
                        X_data[i+1,j] = (X_data[i+1,j] + 1) %2
                        X_data[i+1,j+1] = (X_data[i+1,j+1] + 1) %2
                elif j % 2 == 1:
                    if X_data[i,j] == 1:
                        X_data[i,j] = (X_data[i,j] + 1) %2
                        X_data[i,j-1] = (X_data[i,j-1] + 1) %2
                        X_data[i+1,j] = (X_data[i+1,j] + 1) %2
                        X_data[i+1,j-1] = (X_data[i+1,j-1] + 1) %2
            if i % 2 == 1:
                if j == 0:
                    if X_data[i,j] == 1:
                        X_data[i,j] = (X_data[i,j] + 1) %2
                        X_data[i+1,j] = (X_data[i+1,j] + 1) %2
                elif j % 2 == 1:
                    if X_data[i,j] == 1:
                        X_data[i,j] = (X_data[i,j] + 1) %2
                        X_data[i,j+1] = (X_data[i,j+1] + 1) %2
                        X_data[i+1,j] = (X_data[i+1,j] + 1) %2
                        X_data[i+1,j+1] = (X_data[i+1,j+1] + 1) %2
                elif j % 2 == 0:
                    if X_data[i,j] == 1:
                        X_data[i,j] = (X_data[i,j] + 1) %2
                        X_data[i,j-1] = (X_data[i,j-1] + 1) %2
                        X_data[i+1,j] = (X_data[i+1,j] + 1) %2
                        X_data[i+1,j-1] = (X_data[i+1,j-1] + 1) %2

    ### 論理Xエラーがあるかの判定
    count = [0] * code_distance
    for i in range(code_distance): 
        if X_data[code_distance-1,i] == 1:
            count[i] = 1
    judge_Z = 0
    if count == [1] * code_distance:
        judge_Z = 1
    #############################################

    return result_data_Z, Z_data, judge_X, result_data_X, X_data, judge_Z

def p_matrix(p,eta,round_rep,cd_rep):
    C = 0.02086
    p_th = 0.0146
    matrix = []
    matrix.append(1/2 * (1-(1-2*C*(p/p_th)**((cd_rep+1)/2))**round_rep)) #pL_x
    matrix.append(((4*cd_rep-3)*round_rep+cd_rep)*p/(eta+1)) #pL_z
    matrix.append(p*(4+10**(-5)*p*cd_rep**2)) # pg_x
    matrix.append((5*cd_rep-1)*p/(eta+1)) # pg_z
    matrix.append((3*cd_rep-2)**p/(eta+1)) # pg_syn
    matrix.append(1/(2*(eta+1))*p) #p_z
    matrix.append(eta/(eta+1)*p) #p_x
    return matrix
##################### ここから上をコピーする ######################

def count(trials,cd_sur_list,p_list,eta,cd_rep,result_list):
    count_X = np.zeros((len(cd_sur_list),len(p_list)))
    count_Z = np.zeros((len(cd_sur_list),len(p_list)))
    for _ in range(trials):
        for i in range(len(cd_sur_list)):
            for j in range(len(p_list)):
                result_data_Z, modefied_result_Z, judge_X, result_data_X, modefied_result_X, judge_Z  = sampling(cd_sur_list[i],p_matrix(p_list[j],eta,100//cd_sur_list[i],cd_rep))
                if judge_X == 1:
                    count_X[i,j] += 1
                if judge_Z == 1:
                    count_Z[i,j] += 1
                #print("result_Z\n", result_data_Z)
                #print("modefied_result_Z\n", modefied_result_Z)
                #print("result_X\n", result_data_X)
                #print("modefied_result_X\n", modefied_result_X)
    result_list.append(count_X/trials)
    result_list.append(count_Z/trials)

if __name__ == "__main__":

    ### パラメータ
    trials = 20
    d_s = 3
    d_e = 9
    d_d = 2
    p_s = 0.0001
    p_e = 0.0009
    p_d = 0.0001
    eta = 1000
    cd_rep = 5
    p_list = np.arange(p_s,p_e+p_d,p_d)
    cd_sur_list = np.arange(d_s,d_e+1,d_d)
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
        process = multiprocessing.Process(target=count, args=(trials,cd_sur_list,p_list,eta,cd_rep,result_list,))
        # プロセス開始
        process.start()
        # プロセスのリストに追加
        processes.append(process)

    # プロセスのリストでループ
    for process in processes:
        # プロセスの終了待ち
        process.join()

    for i in range(2*pro):
        if i == 0:
            c_X = result_list[0]
        elif i == 1:
            c_Z = result_list[1]
        elif i%2 == 0:
            c_X += result_list[i]
        elif i%2 == 1:
            c_Z += result_list[i]
    c_X /= pro
    c_Z /= pro

    df_X = pd.DataFrame(data=c_X, columns=p_list,index=cd_sur_list)
    df_Z = pd.DataFrame(data=c_Z, columns=p_list,index=cd_sur_list)
    df_X.to_csv('X,p=('+str(p_s)+','+str(p_e)+','+str(p_d)+'),d=('+str(d_s)+','+str(d_e)+','+str(d_d)+'),d(rep)='+str(cd_rep)+',eta='+str(eta)+',trials='+str(trials*pro)+'.csv')
    df_Z.to_csv('Z,p=('+str(p_s)+','+str(p_e)+','+str(p_d)+'),d=('+str(d_s)+','+str(d_e)+','+str(d_d)+'),d(rep)='+str(cd_rep)+',eta='+str(eta)+',trials='+str(trials*pro)+'.csv')