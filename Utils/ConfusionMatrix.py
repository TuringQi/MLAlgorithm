# coding:utf-8
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # 如果没有传入title参数，则根据以下逻辑生成title
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # 通过klearn.metrics包中的函数计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    '''
    unique_labels(列表1，列表2...) 将列表们的值去交集、去重，按最后结果按从左到右的出现顺序排列
    unique_labels([1, 2, 10], [5, 11])
    array([ 1,  2,  5, 10, 11])
    '''

    # 更新列表，仅使用数据中显示的标签
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        pass
        print('Confusion matrix, without normalization')

    print('confusion_matrix:\n')
    print(cm)
    '''
    Axes.imshow(self, X, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None, origin=None, extent=None, shape=, filternorm=1, filterrad=4.0, imlim=, resample=None, url=None, *, data=None, **kwargs)
    参数X表示图像的数据
    渐变色 cmap 取值参照https://matplotlib.org/stable/tutorials/colors/colormaps.html
    透明度 alpha 0-1
    aspect用于指定热图的单元格的大小
    interpolation 控制热图的颜色显示形式，是否平滑 常用nearest/lanczos
    '''
    fig, ax = plt.subplots()

    im = ax.imshow(X=cm, interpolation='lanczos', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes)-0.5, -0.5)

    # 旋转刻度标签并设置其对齐方式
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 循环数据标注并创建文本注释
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    return ax

# y_test为真实label，y_pred为预测label，classes为类别名称，是个ndarray数组，内容为string类型的标签
y_test = [1, 0, 2, 1, 0, 1, 1]
y_pred = [0, 0, 2, 1, 0, 1, 2]
class_names = np.array(["cat", "dog", "pig"]) #按实际需要修改名称
plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=False)

