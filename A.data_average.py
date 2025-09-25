import numpy as np

if __name__ == "__main__":
    # drug similarity matrix
    drug_drug_interaction_similarity = np.load('../whole_data/multi_similarity/drug_drug_interaction_similarity.npy')
    drug_disease_association_similarity = np.load(
        '../whole_data/multi_similarity/drug_disease_association_similarity.npy')
    drug_sideeffect_association_similarity = np.load(
        '../whole_data/multi_similarity/drug_sideeffect_association_similarity.npy')
    drug_drug_chemistry_similarity = np.load('../whole_data/multi_similarity/drug_drug_chemistry_similarity.npy')

    drug_fusion_similarity = (drug_drug_interaction_similarity + drug_disease_association_similarity +
                              drug_sideeffect_association_similarity + drug_drug_chemistry_similarity) / 4

    # target similarity matrix
    target_disease_association_similarity = np.load(
        '../whole_data/multi_similarity/target_disease_association_similarity.npy')
    target_target_interaction_similarity = np.load(
        '../whole_data/multi_similarity/target_target_interaction_similarity.npy')
    target_target_sequence_similarity = np.load('../whole_data/multi_similarity/target_target_sequence_similarity.npy')

    target_fusion_similarity = (target_disease_association_similarity + target_target_interaction_similarity +
                                target_target_sequence_similarity) / 3

    np.savetxt('../whole_data/multi_similarity/drug_fusion_similarity_708_708.txt', drug_fusion_similarity)
    np.savetxt('../whole_data/multi_similarity/target_fusion_similarity_1512_1512.txt', target_fusion_similarity)

    print('end')
