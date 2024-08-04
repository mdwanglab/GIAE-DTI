import numpy as np
from tqdm import tqdm


# Jaccard similarity matrix calculation
def Jaccard_similarity(A):
    B = np.zeros((A.shape[0], A.shape[0]))
    for i in tqdm(range(B.shape[0])):
        for j in range(i + 1, B.shape[1]):
            if np.sum(A[i]) == 0 and np.sum(A[j]) == 0:
                B[i][j] = 0
            else:
                jiaoji = 0
                bingji = 0
                for k in range(A.shape[1]):
                    if A[i][k] == 1 and A[j][k] == 1:
                        jiaoji += 1
                        bingji += 1
                    elif A[i][k] == 1 or A[j][k] == 1:
                        bingji += 1
                B[i][j] = jiaoji / bingji
    row, col = np.diag_indices_from(B)
    B[row, col] = 1
    B += B.T - np.diag(B.diagonal())
    return B


if __name__ == "__main__":
    # drug similarity matrix
    drug_drug_interaction = np.loadtxt('../data/mat_drug_drug.txt')
    drug_disease_association = np.loadtxt('../data/mat_drug_disease.txt')
    drug_sideeffect_association = np.loadtxt('../data/mat_drug_se.txt')
    drug_drug_interaction_similarity = Jaccard_similarity(drug_drug_interaction)
    drug_disease_association_similarity = Jaccard_similarity(drug_disease_association)
    drug_sideeffect_association_similarity = Jaccard_similarity(drug_sideeffect_association)

    # target similarity matrix
    target_disease_association = np.loadtxt('../data/mat_protein_disease.txt')
    target_target_interaction = np.loadtxt('../data/mat_protein_protein.txt')
    target_disease_association_similarity = Jaccard_similarity(target_disease_association)
    target_target_interaction_similarity = Jaccard_similarity(target_target_interaction)