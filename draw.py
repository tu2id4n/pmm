import csv
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def smooth_and_plot(csv_path, weight=0.99):
    """
    数据平滑，tensorboard
    @param csv_path: tensorboard导出的csv数据
    @param weight:
    @return:
    """
    data = pd.read_csv(filepath_or_buffer=csv_path, header=0, names=['Step', 'Value'],
                       dtype={'Step': np.int, 'Value': np.float})

    steps = data['Step'].values
    stepspan = []

    for step in steps:
        stepspan.append(step / 1000)

    scalar = data['Value'].values
    last = scalar[0]
    smoothed = []

    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    # stepspan = steps.tolist()

    return stepspan, smoothed


def plot_with_label(csvs, labels, save_path, smooth_rate=0.99, xlabel='frame', ylabel='score',
                    y_zero=False, yLocal=None):
    assert len(csvs) == len(labels)
    assert type(csvs) == list
    assert type(labels) == list
    assert type(save_path) == str

    for i in range(len(csvs)):
        step, smooth = smooth_and_plot(csvs[i], smooth_rate)
        plt.plot(step, smooth, label=labels[i])

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if y_zero:
        plt.ylim(bottom=0)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    tag = "win_rate"
    csvs = [
        "csv/DFP_13_" + tag + ".csv",
        "csv/DFP_11_" + tag + ".csv",
    ]  # csvs: 数据来源
    labels = [
        "DFP1",
        "DFP2",
    ]  # 图例
    save_path = "fig/final_" + tag + ".png"  # save_path: 图片保存位置
    smooth_rate = 0.99  # 数据平滑值，和tensorboard一样
    xlabel = 'timespan'  # x轴
    ylabel = tag  # y轴
    y_zero = False  # y轴从0开始
    yLocal = None  # y轴的倍数
    plot_with_label(csvs, labels, save_path, smooth_rate, xlabel, ylabel,
                    y_zero, yLocal)
