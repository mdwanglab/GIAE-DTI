import numpy as np
import csv
import math

import matplotlib.pyplot as plt
from numpy import interp

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

fold = 5


def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName, encoding='utf-8'))
    next(csv_reader)
    for row in csv_reader:
        try:
            for i in range(len(row)):
                if row[i].strip():
                    row[i] = float(row[i])
                else:
                    row[i] = 0.0
        except ValueError as e:
            print(f"Error converting string to float: {e}")
            print(f"Error occurred in row {row}, at column {i}")
        SaveList.append(row)
    return


def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return


def MyConfusionMatrix(y_real, y_predict):
    from sklearn.metrics import confusion_matrix
    CM = confusion_matrix(y_real, y_predict)
    CM = CM.tolist()
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]
    print('TN:%d, FP:%d, FN:%d, TP:%d' % (TN, FP, FN, TP))
    Acc = (TN + TP) / (TN + TP + FN + FP)
    Sen = TP / (TP + FN)
    Spec = TN / (TN + FP)
    Prec = TP / (TP + FP)
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    print('Acc:', round(Acc, 4))
    print('Sen:', round(Sen, 4))
    print('Spec:', round(Spec, 4))
    print('Prec:', round(Prec, 4))
    print('MCC:', round(MCC, 4))
    Result = []
    Result.append(round(Acc, 4))
    Result.append(round(Sen, 4))
    Result.append(round(Spec, 4))
    Result.append(round(Prec, 4))
    Result.append(round(MCC, 4))
    return Result


def MyAverage(matrix):
    SumAcc = 0
    SumSen = 0
    SumSpec = 0
    SumPrec = 0
    SumMcc = 0
    counter = 0
    while counter < len(matrix):
        SumAcc = SumAcc + matrix[counter][0]
        SumSen = SumSen + matrix[counter][1]
        SumSpec = SumSpec + matrix[counter][2]
        SumPrec = SumPrec + matrix[counter][3]
        SumMcc = SumMcc + matrix[counter][4]
        counter = counter + 1
    print('AverageAcc:', SumAcc / len(matrix))
    print('AverageSen:', SumSen / len(matrix))
    print('AverageSpec:', SumSpec / len(matrix))
    print('AveragePrec:', SumPrec / len(matrix))
    print('AverageMcc:', SumMcc / len(matrix))
    return


def MyStd(result):
    import numpy as np
    NewMatrix = []
    counter = 0
    while counter < len(result[0]):
        row = []
        NewMatrix.append(row)
        counter = counter + 1
    counter = 0
    while counter < len(result):
        counter1 = 0
        while counter1 < len(result[counter]):
            NewMatrix[counter1].append(result[counter][counter1])
            counter1 = counter1 + 1
        counter = counter + 1
    StdList = []
    MeanList = []
    counter = 0
    while counter < len(NewMatrix):
        # std
        arr_std = np.std(NewMatrix[counter], ddof=1)
        StdList.append(arr_std)
        # mean
        arr_mean = np.mean(NewMatrix[counter])
        MeanList.append(arr_mean)
        counter = counter + 1
    result.append(MeanList)
    result.append(StdList)
    # 换算成百分比制
    counter = 0
    while counter < len(result):
        counter1 = 0
        while counter1 < len(result[counter]):
            result[counter][counter1] = round(result[counter][counter1] * 100, 2)
            counter1 = counter1 + 1
        counter = counter + 1
    return result


precisions = []
average_precisions = []
mean_fpr = np.linspace(0, 1, 1000)
colorlist = ['red', 'gold', 'purple', 'green', 'blue', 'black']

AllResult = []
Ps = []
Rs = []
RPs = []
mean_R = np.linspace(0, 1, 1000)

for f in range(fold):
    RealAndPrediction = []
    Name = 'test_result_' + str(f) + '.csv'
    ReadMyCsv(RealAndPrediction, Name)
    Real = []
    Prediction = []
    PredictionProb = []
    counter = 0
    while counter < len(RealAndPrediction):
        PredictionProb.append(RealAndPrediction[counter][1])
        Real.append(int(RealAndPrediction[counter][2]))
        Prediction.append(RealAndPrediction[counter][3])
        counter = counter + 1

    average_precision = average_precision_score(Real, PredictionProb)
    precision, recall, _ = precision_recall_curve(Real, PredictionProb)

    Ps.append(interp(mean_R, precision, recall))
    RPs.append(average_precision)
    plt.plot(recall, precision, lw=1.5, alpha=0.8, color=colorlist[f],
             label='fold %d (AUPR = %0.4f)' % (f + 1, average_precision))

    print('average_precision', average_precision)

mean_P = np.mean(Ps, axis=0)
mean_RPs = np.mean(RPs, axis=0)
std_RPs = np.std(RPs)
plt.plot(mean_P, mean_R, color='black',
         label=r'Mean (AUPR = %0.4f)' % (mean_RPs),
         lw=1.25, alpha=1)

PAndR = []
counter = 0
while counter < len(mean_P):
    pair = []
    pair.append(mean_P[counter])
    pair.append(mean_R[counter])
    PAndR.append(pair)
    counter = counter + 1
storFile(PAndR, 'PAndRA+B.csv')

plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.plot([1, 0], [0, 1], color='navy', lw=2, linestyle='--')
plt.legend(bbox_to_anchor=(0.5, 0.4))

plt.savefig('PR-curve.svg')
plt.savefig('PR-curve.tif')
plt.show()
