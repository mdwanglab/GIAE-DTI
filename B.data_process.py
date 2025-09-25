import argparse
import random
import copy
import numpy as np
import math

from maskgae.utils import set_seed, tab_printer


# Singular value decomposition
def svd_dimension_reduction(matrix, dim):
    U, s, V = np.linalg.svd(matrix)
    U_reduced = U[:, :dim]
    return U_reduced


# X......
def prepare_X(SD, ST, dim):
    feature_matrix1 = svd_dimension_reduction(SD, dim)
    feature_matrix2 = svd_dimension_reduction(ST, dim)
    X = np.vstack((feature_matrix1, feature_matrix2))
    print("X.shape:", X.shape)
    np.savetxt("../whole_data/divide_result/X" + ".txt", X)
    print("节点特征 X 准备完成！")


# WKNKN
def WKNKN(DTI, drugSimilarity, proteinSimilarity, K, r):
    drugCount = DTI.shape[0]
    proteinCount = DTI.shape[1]
    flagDrug = np.zeros([drugCount])
    flagProtein = np.zeros([proteinCount])
    for i in range(drugCount):
        for j in range(proteinCount):
            if (DTI[i][j] == 1):
                flagDrug[i] = 1
                flagProtein[j] = 1
    Yd = np.zeros([drugCount, proteinCount])
    Yt = np.zeros([drugCount, proteinCount])
    for d in range(drugCount):
        dnn = KNearestKnownNeighbors(d, drugSimilarity, K, flagDrug)
        w = np.zeros([K])
        Zd = 0
        for i in range(K):
            w[i] = math.pow(r, i) * drugSimilarity[d, dnn[i]]
            Zd += drugSimilarity[d, dnn[i]]
        for i in range(K):
            Yd[d] = Yd[d] + w[i] * DTI[dnn[i]]
        Yd[d] = Yd[d] / Zd

    for t in range(proteinCount):
        tnn = KNearestKnownNeighbors(t, proteinSimilarity, K, flagProtein)
        w = np.zeros([K])
        Zt = 0
        for j in range(K):
            w[j] = math.pow(r, j) * proteinSimilarity[t, tnn[j]]
            Zt += proteinSimilarity[t, tnn[j]]
        for j in range(K):
            Yt[:, t] = Yt[:, t] + w[j] * DTI[:, tnn[j]]
        Yt[:, t] = Yt[:, t] / Zt

    Ydt = Yd + Yt
    Ydt = Ydt / 2

    ans = np.maximum(DTI, Ydt)
    return ans


def KNearestKnownNeighbors(node, matrix, K, flagNodeArray):
    KknownNeighbors = np.array([])
    featureSimilarity = matrix[node].copy()
    featureSimilarity[node] = -100
    featureSimilarity[flagNodeArray == 0] = -100
    KknownNeighbors = featureSimilarity.argsort()[::-1]
    KknownNeighbors = KknownNeighbors[:K]
    return KknownNeighbors


# def divide_A():
#     print("药物数量：", m, "|| 靶标数量：", n)
#     print('negative: positive = ', times)
#     labels = A.flatten()
#     # Positive sample
#     i = 0
#     list_1 = []
#     while i < len(labels):
#         if labels[i] == 1:
#             list_1.append(i)
#         i = i + 1
#     num1 = len(list_1)
#     group_size1 = int(num1 / fold)
#     random.shuffle(list_1)
#
#     array_1 = np.array(list_1)[0:fold * group_size1]
#     grouped_data1 = np.reshape(array_1, (fold, group_size1))
#     np.savetxt("../whole_data/divide_result/index_1.txt", grouped_data1)
#
#     # Negative sample
#     i = 0
#     list_0 = []
#     while i < len(labels):
#         if labels[i] == 0:
#             list_0.append(i)
#         i = i + 1
#     list_0 = random.sample(list_0, times * len(array_1))  # Random sampling of negative samples
#     num0 = len(list_0)
#     group_size0 = int(num0 / fold)
#     random.shuffle(list_0)
#
#     array_0 = np.array(list_0)[0:fold * group_size0]
#     grouped_data0 = np.reshape(array_0, (fold, group_size0))
#     np.savetxt("../whole_data/divide_result/index_0.txt", grouped_data0)
#
#     print('Number of positive samples：', len(array_1))
#     print('Number of negative samples：', len(array_0))
#
#     # A......
#     f = 0
#     while f < fold:
#         print('prepare A ......', f, '-fold ')
#         i = 0
#         DTI = copy.deepcopy(A_WKNKN)
#         while i < group_size1:
#             r = int(grouped_data1[f, i] / n)
#             c = int(grouped_data1[f, i] % n)
#             DTI[r, c] = 0
#             i += 1
#
#         feature_matrix3 = np.hstack((DDI, DTI))
#         feature_matrix4 = np.hstack((DTI.T, TTI))
#         adj = np.vstack((feature_matrix3, feature_matrix4))
#         np.savetxt("../whole_data/divide_result/A" + str(f) + ".txt", adj)
#         f += 1
#     print('end.........')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=512,
                        help="SVD dimension reduction")
    parser.add_argument('--seed', type=int, default=2022304254,
                        help='Random seed for model and dataset. (default: 2022304254)')
    parser.add_argument('--f', dest="fold", type=int, default=5,
                        help="divided data into ? fold")
    parser.add_argument('--times', type=int, default=1,
                        help="negative: positive")
    parser.add_argument('--K', type=int, default=7,
                        help="divided data into ? fold")
    parser.add_argument('--p', type=float, default=0.9)

    try:
        args = parser.parse_args()
        print(tab_printer(args))
    except:
        parser.print_help()
        exit(0)

    print("-" * 100)
    print("1.准备DTI网络的特征矩阵Xdt......")
    print("将SD和ST进行奇异值分解降到同一维度.......：", args.dim)
    SD = np.loadtxt('../whole_data/multi_similarity/drug_fusion_similarity_708_708.txt')
    ST = np.loadtxt('../whole_data/multi_similarity/target_fusion_similarity_1512_1512.txt')
    # X
    prepare_X(SD, ST, args.dim)
    set_seed(args.seed)
    fold = args.fold
    times = args.times
    A = np.loadtxt("../whole_data/Luo data/mat_drug_protein.txt")
    # A_WKNKN = np.loadtxt('../whole_data/multi_similarity/DTI_708_1512_WKNKN_MAX_DISCRETIZE.txt')  # WKNKN补全矩阵
    m = A.shape[0]
    n = A.shape[1]
    DDI = np.loadtxt("../whole_data/Luo data/mat_drug_drug.txt")
    TTI = np.loadtxt("../whole_data/Luo data/mat_protein_protein.txt")

    # 对每一折A进行处理并划分数据。
    # divide_A()
    # print("药物数量：", m, "|| 靶标数量：", n)
    print('negative: positive = ', times)
    labels = A.flatten()
    # Positive sample
    i = 0
    list_1 = []
    while i < len(labels):
        if labels[i] == 1:
            list_1.append(i)
        i = i + 1
    num1 = len(list_1)
    group_size1 = int(num1 / fold)
    random.shuffle(list_1)

    array_1 = np.array(list_1)[0:fold * group_size1]
    grouped_data1 = np.reshape(array_1, (fold, group_size1))
    np.savetxt("../whole_data/divide_result/index_1.txt", grouped_data1)

    # Negative sample
    i = 0
    list_0 = []
    while i < len(labels):
        if labels[i] == 0:
            list_0.append(i)
        i = i + 1
    list_0 = random.sample(list_0, times * len(array_1))  # Random sampling of negative samples
    num0 = len(list_0)
    group_size0 = int(num0 / fold)
    random.shuffle(list_0)

    array_0 = np.array(list_0)[0:fold * group_size0]
    grouped_data0 = np.reshape(array_0, (fold, group_size0))
    np.savetxt("../whole_data/divide_result/index_0.txt", grouped_data0)

    print('Number of positive samples：', len(array_1))
    print('Number of negative samples：', len(array_0))

    # A......
    f = 0
    while f < fold:
        print("2.WKNKN对矩阵A进行丰富........")
        DTI = np.loadtxt('../whole_data/Luo data/mat_drug_protein.txt')
        A = copy.deepcopy(DTI)
        i = 0
        while i < group_size1:
            r = int(grouped_data1[f, i] / n)
            c = int(grouped_data1[f, i] % n)
            A[r, c] = 0
            i += 1
        print("A中非0元素个数", np.count_nonzero(A == 1))
        predict_Y = WKNKN(DTI=A, drugSimilarity=SD, proteinSimilarity=ST, K=args.K, r=args.p)

        float_array = copy.deepcopy(predict_Y[(predict_Y > 0) & (predict_Y < 1)])
        # float_median = np.median(float_array)
        # ***********************************************************************
        sorted_array = np.sort(float_array)
        length = len(sorted_array)
        percentile = 1 - 25/100
        index = int(length * percentile)
        float_median = sorted_array[index]
        # ******************************
        print("预测结果中将>=阈值设为1，<阈值设为0")
        predict_Y[predict_Y > float_median] = 1
        predict_Y[predict_Y <= float_median] = 0
        A_WKNKN = predict_Y
        feature_matrix3 = np.hstack((DDI, A_WKNKN))
        feature_matrix4 = np.hstack((A_WKNKN.T, TTI))
        adj = np.vstack((feature_matrix3, feature_matrix4))
        np.savetxt("../whole_data/divide_result/A" + str(f) + ".txt", adj)
        f += 1
    print('end.........')