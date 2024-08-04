import numpy as np
import csv
import math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

fold = 5


def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName, encoding='utf-8'))
    next(csv_reader)
    for row_idx, row in enumerate(csv_reader):
        try:
            for col_idx in range(len(row)):
                if row[col_idx].strip():
                    row[col_idx] = float(row[col_idx])
                else:
                    row[col_idx] = 0.0
        except ValueError as e:
            print(f"Error converting string to float: {e}")
            print(f"Error occurred in row {row_idx}, at column {col_idx}")
        SaveList.append(row)
    return


def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return


def MyEnlarge(x0, y0, width, height, x1, y1, times, mean_fpr, mean_tpr, thickness=1, color='blue'):
    def MyFrame(x0, y0, width, height):
        import matplotlib.pyplot as plt
        import numpy as np

        x1 = np.linspace(x0, x0, num=20)
        y1 = np.linspace(y0, y0, num=20)
        xk = np.linspace(x0, x0 + width, num=20)
        yk = np.linspace(y0, y0 + height, num=20)

        xkn = []
        ykn = []
        counter = 0
        while counter < 20:
            xkn.append(x1[counter] + width)
            ykn.append(y1[counter] + height)
            counter = counter + 1

        plt.plot(x1, yk, color='k', linestyle=':', lw=1, alpha=1)  # 左
        plt.plot(xk, y1, color='k', linestyle=':', lw=1, alpha=1)  # 下
        plt.plot(xkn, yk, color='k', linestyle=':', lw=1, alpha=1)  # 右
        plt.plot(xk, ykn, color='k', linestyle=':', lw=1, alpha=1)  # 上

        return

    width2 = times * width
    height2 = times * height
    MyFrame(x0, y0, width, height)
    MyFrame(x1, y1, width2, height2)

    xp = np.linspace(x0 + width, x1, num=20)
    yp = np.linspace(y0, y1 + height2, num=20)
    plt.plot(xp, yp, color='k', linestyle=':', lw=1, alpha=1)

    XDottedLine = []
    YDottedLine = []
    counter = 0
    while counter < len(mean_fpr):
        if mean_fpr[counter] > x0 and mean_fpr[counter] < (x0 + width) and mean_tpr[counter] > y0 and mean_tpr[
            counter] < (y0 + height):
            XDottedLine.append(mean_fpr[counter])
            YDottedLine.append(mean_tpr[counter])
        counter = counter + 1

    counter = 0
    while counter < len(XDottedLine):
        XDottedLine[counter] = (XDottedLine[counter] - x0) * times + x1
        YDottedLine[counter] = (YDottedLine[counter] - y0) * times + y1
        counter = counter + 1

    plt.plot(XDottedLine, YDottedLine, color=color, lw=thickness, alpha=1)
    return


def MyEnlarge2(x0, y0, width, height, x1, y1, times, mean_fpr, mean_tpr, thickness=1, color='blue'):
    def MyFrame(x0, y0, width, height):
        import matplotlib.pyplot as plt
        import numpy as np

        x1 = np.linspace(x0, x0, num=20)
        y1 = np.linspace(y0, y0, num=20)
        xk = np.linspace(x0, x0 + width, num=20)
        yk = np.linspace(y0, y0 + height, num=20)

        xkn = []
        ykn = []
        counter = 0
        while counter < 20:
            xkn.append(x1[counter] + width)
            ykn.append(y1[counter] + height)
            counter = counter + 1

        plt.plot(x1, yk, color='k', linestyle=':', lw=1, alpha=1)  # 左
        plt.plot(xk, y1, color='k', linestyle=':', lw=1, alpha=1)  # 下
        plt.plot(xkn, yk, color='k', linestyle=':', lw=1, alpha=1)  # 右
        plt.plot(xk, ykn, color='k', linestyle=':', lw=1, alpha=1)  # 上

        return

    width2 = times * width
    height2 = times * height
    MyFrame(x0, y0, width, height)
    MyFrame(x1, y1, width2, height2)

    xp = np.linspace(x0, x1 + 2 * width, num=20)
    yp = np.linspace(y0, y1 + height2, num=20)
    plt.plot(xp, yp, color='k', linestyle=':', lw=1, alpha=1)

    XDottedLine = []
    YDottedLine = []
    counter = 0
    while counter < len(mean_fpr):
        if mean_fpr[counter] > x0 and mean_fpr[counter] < (x0 + width) and mean_tpr[counter] > y0 and mean_tpr[
            counter] < (y0 + height):
            XDottedLine.append(mean_fpr[counter])
            YDottedLine.append(mean_tpr[counter])
        counter = counter + 1

    counter = 0
    while counter < len(XDottedLine):
        XDottedLine[counter] = (XDottedLine[counter] - x0) * times + x1
        YDottedLine[counter] = (YDottedLine[counter] - y0) * times + y1
        counter = counter + 1

    plt.plot(XDottedLine, YDottedLine, color=color, lw=thickness, alpha=1)
    return


def MyConfusionMatrix(y_real, y_predict):
    from sklearn.metrics import confusion_matrix
    CM = confusion_matrix(y_real, y_predict)
    print(CM)
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
    counter = 0
    while counter < len(result):
        counter1 = 0
        while counter1 < len(result[counter]):
            result[counter][counter1] = round(result[counter][counter1] * 100, 2)
            counter1 = counter1 + 1
        counter = counter + 1

    counter = 0
    while counter < len(StdList):
        result[5][counter] = str(result[5][counter]) + str('+') + str(result[6][counter])
        counter = counter + 1

    return result


tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 1000)
colorlist = ['red', 'gold', 'purple', 'green', 'blue']

AllResult = []
dataset_list = ['DTINet']

for dataset in dataset_list:
    tprs_dataset = []
    aucs_dataset = []
    for f in range(fold):
        RealAndPrediction = []
        Name = 'test_result' + str(f) + '.csv'
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

        fpr, tpr, thresholds = roc_curve(Real, PredictionProb)
        fpr = fpr.tolist()
        tpr = tpr.tolist()
        fpr.insert(0, 0)
        tpr.insert(0, 0)

        roc_auc = auc(fpr, tpr)
        # lw = 0.95
        plt.plot(fpr, tpr, color=colorlist[f], lw=0.95,
                 label='Fold %d (AUC = %0.4f)' % (f + 1, roc_auc))

        tprs_dataset.append(np.interp(mean_fpr, fpr, tpr))
        tprs_dataset[-1][0] = 0.0
        aucs_dataset.append(roc_auc)

    mean_tpr = np.mean(tprs_dataset, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs_dataset)
    print(mean_auc)
    plt.plot(mean_fpr, mean_tpr, color='black', label=r'Mean  (AUC = %0.4f)' % mean_auc, lw=1.25, alpha=1)
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('ROC-curve.svg')
    plt.savefig('ROC-curve.tif')
    plt.close()

    AllResult.append([mean_auc] + aucs_dataset)

print('All datasets ROC curves have been plotted.')
