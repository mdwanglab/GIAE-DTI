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
    drug_drug_interaction = np.loadtxt('../whole_data/Luo data/mat_drug_drug.txt')
    drug_disease_association = np.loadtxt('../whole_data/Luo data/mat_drug_disease.txt')
    drug_sideeffect_association = np.loadtxt('../whole_data/Luo data/mat_drug_se.txt')
    drug_drug_chemistry_similarity = np.loadtxt('../whole_data/Luo data/Similarity_Matrix_Drugs.txt')
    # Jaccard similarity
    drug_drug_interaction_similarity = Jaccard_similarity(drug_drug_interaction)
    drug_disease_association_similarity = Jaccard_similarity(drug_disease_association)
    drug_sideeffect_association_similarity = Jaccard_similarity(drug_sideeffect_association)
    np.save('../whole_data/multi_similarity/drug_drug_interaction_similarity.npy', drug_drug_interaction_similarity)
    np.save('../whole_data/multi_similarity/drug_disease_association_similarity.npy', drug_disease_association_similarity)
    np.save('../whole_data/multi_similarity/drug_sideeffect_association_similarity.npy',
            drug_sideeffect_association_similarity)
    np.save('../whole_data/multi_similarity/drug_drug_chemistry_similarity.npy', drug_drug_chemistry_similarity)
    # Fusion similarity
    x1 = np.maximum(drug_drug_chemistry_similarity, drug_drug_interaction_similarity)
    x2 = np.maximum(x1, drug_disease_association_similarity)
    drug_fusion_similarity = np.maximum(x2, drug_sideeffect_association_similarity)
    np.savetxt('../whole_data/multi_similarity/drug_fusion_similarity_708_708.txt', drug_fusion_similarity)

    # target similarity matrix
    target_disease_association = np.loadtxt('../whole_data/Luo data/mat_protein_disease.txt')
    target_target_interaction = np.loadtxt('../whole_data/Luo data/mat_protein_protein.txt')
    target_target_sequence_similarity = np.loadtxt('../whole_data/Luo data/Similarity_Matrix_Proteins.txt')
    # Jaccard similarity
    target_disease_association_similarity = Jaccard_similarity(target_disease_association)
    target_target_interaction_similarity = Jaccard_similarity(target_target_interaction)
    np.save('../whole_data/multi_similarity/target_disease_association_similarity.npy',
            target_disease_association_similarity)
    np.save('../whole_data/multi_similarity/target_target_interaction_similarity.npy',
            target_target_interaction_similarity)
    np.save('../whole_data/multi_similarity/target_target_sequence_similarity.npy', target_target_sequence_similarity)
    # Fusion similarity
    y1 = np.maximum(target_target_sequence_similarity, target_disease_association_similarity)
    target_fusion_similarity = np.maximum(y1, target_target_interaction_similarity)
    np.savetxt('../whole_data/multi_similarity/target_fusion_similarity_1512_1512.txt', target_fusion_similarity)
    print('end......')
    
    