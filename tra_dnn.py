import torch
from utils.utils import set_seed
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, recall_score, precision_score, \
    accuracy_score, confusion_matrix
import numpy as np
import copy
import argparse
import torch.nn.functional as F

device = torch.device("cuda:0")
DTI = np.loadtxt('../data/mat_drug_protein.txt')
n = DTI.shape[0]
m = DTI.shape[1]
print("drug number：", n, "|| target number：", m)


class predict(torch.nn.Module):
    def __init__(self, input_dim):
        super(predict, self).__init__()
        self.fullynet = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(64, 2),
            torch.nn.Sigmoid(),
        )

    def forward(self, feature):
        outputs = self.fullynet(feature)
        return outputs


def test(model, feature, label, device):
    model.eval()

    out = model(feature)
    pred = out.cpu().detach().numpy()
    test_label = torch.as_tensor(label).float()

    loss = F.binary_cross_entropy(out[:, 1], test_label.to(device))

    fpr, tpr, auc_thresholds = roc_curve(test_label, pred[:, 1])
    AUC = auc(fpr, tpr)  # auc
    precision, recall, pr_threshods = precision_recall_curve(test_label, pred[:, 1])
    AUPR = auc(recall, precision)  # aupr

    # The threshold is determined by F1-score.
    all_F_measure = np.zeros(len(pr_threshods))
    pred_cls = np.zeros(len(test_label))

    for k in range(0, len(pr_threshods)):
        if (precision[k] + recall[k]) > 0:
            all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
        else:
            all_F_measure[k] = 0

    max_index = all_F_measure.argmax()
    threshold = pr_threshods[max_index]
    p = pred[:, 1]
    pred_cls[p > threshold] = 1

    ACC = accuracy_score(test_label, pred_cls)  # acc
    Recall = recall_score(test_label, pred_cls)  # recall
    Precision = precision_score(test_label, pred_cls)  # precision
    f1 = f1_score(test_label, pred_cls)  # f1
    tn, fp, fn, tp = confusion_matrix(test_label, pred_cls).ravel()
    specificity = tn / (tn + fp)  # specificity

    return loss.item(), AUC, AUPR, ACC, Recall, Precision, f1, specificity, pred[:, 1], pred_cls,


def get_feature(drug_feature, target_feature, label, index, drug_num, target_num):
    dti_feature = []  # save feature
    output = []  # save label
    for i in range(index.shape[0]):
        drug = int(index[i] / target_num)  # drug index
        target = int(index[i] % target_num)  # target index
        feature = np.hstack((drug_feature[drug], target_feature[target]))
        dti_feature.append(feature.tolist())
        output.append(label[drug, target])
    return np.array(dti_feature), np.array(output)


def preparedata(fold, args):
    index_1 = np.loadtxt('../divide_result/index_pos.txt')
    index_0 = np.loadtxt('../divide_result/index_neg.txt')
    index = np.hstack((index_1, index_0))
    drug_feature = np.loadtxt('../Result/embedding' + str(fold) + '_' + str(args.layer) + '.txt')[0:n, :]
    target_feature = np.loadtxt('../Result/embedding' + str(fold) + '_' + str(args.layer) + '.txt')[n:, :]

    # prepare data......
    idx = copy.deepcopy(index)
    test_index = copy.deepcopy(idx[fold])
    idx = np.delete(idx, fold, axis=0)
    train_index = idx.flatten()
    insersection = np.intersect1d(test_index, train_index)
    if insersection.size > 0:
        raise ValueError("There is an intersection between the test set and the training set")

    np.random.shuffle(train_index)

    train_feature, train_label = get_feature(drug_feature, target_feature, DTI, train_index, n, m)
    train_feature = torch.from_numpy(train_feature).float().to(device)
    test_feature, test_label = get_feature(drug_feature, target_feature, DTI, test_index, n, m)
    test_feature = torch.from_numpy(test_feature).float().to(device)

    return train_feature, train_label, test_feature, test_label, test_index


def preparetrain(train_feature, train_label, test_feature, test_label, test_index, fold, args):
    model = predict(args.hid)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight)
    # train
    for epoch in tqdm(range(args.epoch)):
        model.train()
        out = model(train_feature)
        train_label = torch.as_tensor(train_label).float()
        loss = F.binary_cross_entropy(out[:, 1], train_label.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # test
    test_loss, test_auc, test_aupr, test_acc, test_recall, test_precision, test_f, specificity, pred, pred_cls = test(
        model, test_feature, test_label, device)

    return test_auc, test_aupr, test_acc, test_recall, test_precision, test_f, specificity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hid', type=int, default=128, help="model input dimension")
    parser.add_argument('--epoch', type=int, default=3000)
    parser.add_argument('--weight', type=float, default=1e-5, help="weight_decay")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--seed', type=int, default=2023, help='Random seed for model and dataset. (default: 2023)')
    parser.add_argument('--f', dest="folds", type=int, default=5, help="Cross validation folds")
    parser.add_argument("--mask", nargs="?", default="None")
    parser.add_argument('--p', type=float, default=0, help='Mask ratio or sample ratio for MaskEdge/MaskPath')
    parser.add_argument("--layer", nargs="?", default="gin", help="GNN layer")

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(0)

    folds = args.folds
    all_performance = []
    for fold in range(folds):
        set_seed(args.seed)
        print("prepare: ", fold, "-fold data!!!")
        train_feature, train_label, test_feature, test_label, test_index = preparedata(fold, args)
        print('start training............')
        test_auc, test_aupr, test_acc, test_recall, test_precision, test_f, specificity = preparetrain(
            train_feature, train_label, test_feature, test_label, test_index, fold, args)
        all_performance.append(
            [test_auc, test_aupr, test_acc, test_recall, test_precision, test_f, specificity])
    print(all_performance)
