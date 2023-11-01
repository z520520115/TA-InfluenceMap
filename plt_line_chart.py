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

# loss
def train_loss_plt():
    plt.figure(1)
    x1, y1 = readcsv('./result/influence_map/train_loss_50.csv')
    plt.plot(x1, y1, color='dodgerblue', label='Influence Map')
    x2, y2 = readcsv('./result/object_bounding_box/train_loss_50.csv')
    plt.plot(x2, y2, color='orange', label='Object Bounding Box')
    x3, y3 = readcsv('./result/object_trajectory/train_loss_50.csv')
    plt.plot(x3, y3, color='indigo', label='Object Trajectory')
    x4, y4 = readcsv('./result/optical_flow/train_loss_50.csv')
    plt.plot(x4, y4, color='teal', label='Optical Flow')
    x5, y5 = readcsv('./result/attention_map/train_loss_50.csv')
    plt.plot(x5, y5, color='violet', label='Attention Map')

    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    # plt.savefig('./graph/contrast experiment train loss 50.png')
    plt.show()

def val_loss_plt():
    plt.figure(2)
    x1, y1 = readcsv('./result/influence_map/val_loss_50.csv')
    plt.plot(x1, y1, color='dodgerblue', label='Influence Map')
    x2, y2 = readcsv('./result/object_bounding_box/val_loss_50.csv')
    plt.plot(x2, y2, color='orange', label='Object Bounding Box')
    x3, y3 = readcsv('./result/object_trajectory/val_loss_50.csv')
    plt.plot(x3, y3, color='indigo', label='Object Trajectory')
    x4, y4 = readcsv('./result/optical_flow/val_loss_50.csv')
    plt.plot(x4, y4, color='teal', label='Optical Flow')
    x5, y5 = readcsv('./result/attention_map/val_loss_50.csv')
    plt.plot(x5, y5, color='violet', label='Attention Map')

    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    # plt.savefig('./graph/contrast experiment val loss 50.png')
    plt.show()

# acc
def train_acc_plt():
    plt.figure(3)
    x1, y1 = readcsv('./result/influence_map/train_acc_50.csv')
    plt.plot(x1, y1, color='dodgerblue', label='Influence Map')
    x2, y2 = readcsv('./result/object_bounding_box/train_acc_50.csv')
    plt.plot(x2, y2, color='orange', label='Object Bounding Box')
    x3, y3 = readcsv('./result/object_trajectory/train_acc_50.csv')
    plt.plot(x3, y3, color='indigo', label='Object Trajectory')
    x4, y4 = readcsv('./result/optical_flow/train_acc_50.csv')
    plt.plot(x4, y4, color='teal', label='Optical Flow')
    x5, y5 = readcsv('./result/attention_map/train_acc_50.csv')
    plt.plot(x5, y5, color='violet', label='Attention Map')

    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    # plt.savefig('./graph/contrast experiment train acc 50.png')
    plt.show()

def val_acc_plt():
    plt.figure(4)
    x1, y1 = readcsv('./result/influence_map/val_acc_50.csv')
    plt.plot(x1, y1, color='dodgerblue', label='Influence Map')
    x2, y2 = readcsv('./result/object_bounding_box/val_acc_50.csv')
    plt.plot(x2, y2, color='orange', label='Object Bounding Box')
    x3, y3 = readcsv('./result/object_trajectory/val_acc_50.csv')
    plt.plot(x3, y3, color='indigo', label='Object Trajectory')
    x4, y4 = readcsv('./result/optical_flow/val_acc_50.csv')
    plt.plot(x4, y4, color='teal', label='Optical Flow')
    x5, y5 = readcsv('./result/attention_map/val_acc_50.csv')
    plt.plot(x5, y5, color='violet', label='Attention Map')

    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    # plt.savefig('./graph/contrast experiment val acc 50.png')
    plt.show()

if __name__ == '__main__':
    train_loss_plt()
    train_acc_plt()
    val_loss_plt()
    val_acc_plt()