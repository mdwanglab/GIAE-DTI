import argparse
import random
import copy
import numpy as np
import math
from utils.utils import set_seed, tab_printer


# Singular value decomposition
def svd_dimension_reduction(matrix, dim):
    U, s, V = np.linalg.svd(matrix)
    U_reduced = U[:, :dim]
    return U_reduced


# SD: Drug similarity matrix. ST: Target similarity matrix.
def prepare_X(SD, ST, dim):
    feature_matrix1 = svd_dimension_reduction(SD, dim)
    feature_matrix2 = svd_dimension_reduction(ST, dim)
    X = np.vstack((feature_matrix1, feature_matrix2))
    np.savetxt("../network data/X" + ".txt", X)


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
    parser.add_argument('--K', type=int, default=4,
                        help="divided data into ? fold")
    parser.add_argument('--p', type=float, default=0.9)

    try:
        args = parser.parse_args()
        print(tab_printer(args))
    except:
        parser.print_help()
        exit(0)

    print("-" * 100)
    # Loading similarity matrix
    SD = np.loadtxt('../data_pro/drug_fusion_similarity.txt')
    ST = np.loadtxt('../data_pro/target_fusion_similarity.txt')

    # Prepare the node feature matrix X
    prepare_X(SD, ST, args.dim)
    set_seed(args.seed)  # Guaranteed reproducibility
    fold = args.fold
    times = args.times
    A = np.loadtxt("../data/mat_drug_protein.txt")
    m = A.shape[0]
    n = A.shape[1]
    DDI = np.loadtxt("../data/mat_drug_drug.txt")
    TTI = np.loadtxt("../data/mat_protein_protein.txt")

    # Data partitioning
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
    index_pos = np.reshape(array_1, (fold, group_size1))
    np.savetxt("../divide_result/index_pos.txt", index_pos)

    # Negative sample
    i = 0
    list_0 = []
    while i < len(labels):
        if labels[i] == 0:
            list_0.append(i)
        i = i + 1
    # Random sampling of negative samples
    list_0 = random.sample(list_0, times * len(array_1))
    num0 = len(list_0)
    group_size0 = int(num0 / fold)
    random.shuffle(list_0)

    array_0 = np.array(list_0)[0:fold * group_size0]
    index_neg = np.reshape(array_0, (fold, group_size0))
    np.savetxt("../divide_result/index_neg.txt", index_neg)

    print('Number of positive samples：', len(array_1))
    print('Number of negative samples：', len(array_0))

    # A: Adjacency matrix of heterogeneous network
    f = 0
    while f < fold:
        DTI = np.loadtxt('../data/mat_drug_protein.txt')
        A = copy.deepcopy(DTI)
        i = 0
        # Hiding the test set label for each fold
        while i < group_size1:
            r = int(index_pos[f, i] / n)
            c = int(index_pos[f, i] % n)
            A[r, c] = 0
            i += 1

        # WKNKN Preprocessing
        predict_Y = WKNKN(DTI=A, drugSimilarity=SD, proteinSimilarity=ST, K=args.K, r=args.p)
        # Set threshold
        float_array = copy.deepcopy(predict_Y[(predict_Y > 0) & (predict_Y < 1)])
        sorted_array = np.sort(float_array)
        length = len(sorted_array)
        percentile = 1 - 25 / 100
        index = int(length * percentile)
        threshold = sorted_array[index]
        # Set the threshold to 1 if it is greater than the threshold and 0 if it is less than the threshold
        predict_Y[predict_Y > threshold] = 1
        predict_Y[predict_Y <= threshold] = 0
        A_WKNKN = predict_Y
        edge1 = np.hstack((DDI, A_WKNKN))
        edge2 = np.hstack((A_WKNKN.T, TTI))
        Edge = np.vstack((edge1, edge2))
        np.savetxt("../network data/A" + str(f) + ".txt", Edge)
        f += 1
    print('end.........')