import matplotlib.pyplot as plt
import csv
import numpy as np

'''读取csv文件, 画图前需要删除csv中的第一行'''

def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for row in plots:
        y.append(float(row[2]))
        x.append(float(row[1]))
    return x, y

# accuracy
def acc_plt():

    # 设置X轴标签，以及X轴标签对应的y值
    x = ['Influence\nMap', 'Object Bounding\nBox', 'Object\nTrajectory', 'Optical\nFlow', 'Attention\nMap']
    train_acc = [97.3, 96.4, 93.8, 67.0, 86.6]
    val_acc = [95.9, 91.7, 93.0, 65.9, 84.3]

    # 定位好柱子的开始位置
    x_len = np.arange(len(x))
    total_width, n = 1.0, 5
    width = total_width / n # 0.2

    # 定义图片大小，为图片画上网格
    xticks = x_len + 0.25 - (total_width - width) / 2 # [-0.4  0.6  1.6  2.6  3.6]
    # xticks = [-0.8, 0.8, 1.8, 2.8, 3.8]

    plt.figure(figsize = (30, 20), dpi=200)
    ax = plt.axes()
    plt.grid(axis="y", c='#d2c9eb', linestyle = '--',zorder=0)

    # 设置柱子所需要的属性
    plt.bar(xticks, train_acc, width=1.5 * width, label="Training Accuracy", color="#056eee", edgecolor='black',
            linewidth=2, zorder=10)
    plt.bar(xticks + 0.3, val_acc, width=1.5 * width, label="Validation Accuracy", color="orange", edgecolor='black',
            linewidth=2, zorder=10)

    # 为柱子上方添加数值
    plt.text(xticks[0], train_acc[0] - 3, train_acc[0], ha='center', weight='bold', fontsize=35, zorder=10)
    plt.text(xticks[1], train_acc[1] - 3, train_acc[1], ha='center', weight='bold', fontsize=35, zorder=10)
    plt.text(xticks[2], train_acc[2] - 3, train_acc[2], ha='center', weight='bold', fontsize=35, zorder=10)
    plt.text(xticks[3], train_acc[3] - 3, train_acc[3], ha='center', weight='bold', fontsize=35, zorder=10)
    plt.text(xticks[4], train_acc[4] - 3, train_acc[4], ha='center', weight='bold', fontsize=35, zorder=10)

    plt.text(xticks[0] + 0.3, val_acc[0] - 3, val_acc[0], ha='center', weight='bold', fontsize=35, zorder=10)
    plt.text(xticks[1] + 0.3, val_acc[1] - 3, val_acc[1], ha='center', weight='bold', fontsize=35, zorder=10)
    plt.text(xticks[2] + 0.3, val_acc[2] - 3, val_acc[2], ha='center', weight='bold', fontsize=35, zorder=10)
    plt.text(xticks[3] + 0.3, val_acc[3] - 3, val_acc[3], ha='center', weight='bold', fontsize=35, zorder=10)
    plt.text(xticks[4] + 0.3, val_acc[4] - 3, val_acc[4], ha='center', weight='bold', fontsize=35, zorder=10)

    plt.legend(prop={'size': 35}, ncol=3)

    # 设置X轴，Y轴标签的字体及大小
    plt.xticks(x_len, x, fontproperties='Cambria Math', fontsize=40)
    plt.yticks(fontproperties='Cambria Math', fontsize=40)

    # 控制Y轴的值
    plt.ylim(0, 100)

    # 设置对柱状图X轴的说明和Y轴的说明
    plt.xlabel("Data Types", fontproperties='Cambria Math', fontsize=50)
    plt.ylabel("Accuracy (%)", fontproperties='Cambria Math', fontsize=50)

    # 为柱状图画框
    ax.spines['bottom'].set_linewidth('2.0')
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_linewidth('2.0')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_linewidth('2.0')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_linewidth('2.0')
    ax.spines['left'].set_color('black')
    # plt.savefig('./graph/contrast experiment accuracy.png')
    plt.show()

if __name__ == '__main__':

    acc_plt()
