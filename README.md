# GIAE-DTI: Predicting Drug-Target Interactions Based on Heterogeneous Network and GIN-based Graph Autoencoder

## Abstract

Accurate prediction of drug-target interactions (DTIs) is essential for advancing drug discovery and repurposing. However, the sparsity of DTI data limits the effectiveness of existing computational methods, which primarily focus on sparse DTI networks and have poor performance in aggregating information from neighboring nodes and representing isolated nodes within the network. In this study, we propose a novel deep learning framework, named GIAE-DTI, which considers cross-modal similarity of drugs and targets and constructs a heterogeneous network for DTI prediction. Firstly, the model calculates the cross-modal similarity of drugs and proteins from the relationships among drugs, proteins, diseases, and side effects, and performs similarity integration by taking the average. Then, a drug-target heterogeneous network is constructed, including drug-drug interactions, protein-protein interactions, and drug-target interactions processed by weighted K nearest known neighbors. In the heterogeneous network, a graph autoencoder based on a graph isomorphism network is employed for feature extraction, while a dual decoder is utilized to achieve better self-supervised learning, resulting in latent feature representations for drugs and targets. Finally, a deep neural network is employed to predict DTIs. The experimental results indicate that on the benchmark dataset, GIAE-DTI achieves AUC and AUPR scores of 0.9533 and 0.9619, respectively, in DTI prediction, outperforming the current state-of-the-art methods. Additionally, case studies on four 5-hydroxytryptamine receptor-related targets and five drugs related to mental diseases show the great potential of the proposed method in practical applications.

## Dependency
In order to run this code, you need to install following dependencies:
* python: 3.7.0
* torch: 1.10.1
* torch-cluster: 1.5.9
* torch-scatter: 2.0.9
* torch-sparse: 0.6.12
* torch-spline-conv: 1.2.1

## Datasets
#  Data description
`data` Folder. This folder contains the DTINet dataset.
-   drug.txt: list of drug names.
-   protein.txt: list of protein names.
-   disease.txt: list of disease names.
-   se.txt: list of side effect names.
-   drug_dict_map: a complete ID mapping between drug names and DrugBank ID.
-   protein_dict_map: a complete ID mapping between protein names and UniProt ID.
-   mat_drug_drug.txt: Drug-Drug interaction matrix.
-   mat_drug_disease.txt: Drug-Disease association matrix.
-   mat_drug_protein.txt: Drug-Protein interaction matrix.
-   mat_drug_se.txt: Drug-SideEffect association matrix.
-   mat_protein_disease.txt: Protein-Disease association matrix.
-   mat_protein_drug.txt: Protein-Drug interaction matrix.
-   mat_protein_protein.txt: Protein-Protein interaction matrix.
-   Similarity_Matrix_Drugs.txt: Drug comprehensive similarity matrix
-   Similarity_Matrix_Proteins.txt: Protein comprehensive similarity matrix
