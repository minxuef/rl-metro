
import pylab
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def getTrateForce(veo):
    # UNTITLED3 根据牵引曲线计算牵引力大小
    #  传入速度单位为m/s
    f = 0
    u = veo
      # 单位换算
    if u <= 42:
        f = 558
    else:
        f = -0.0004976 * u * u * u + 0.1774 * u * u - 22.67 * u + 1223
    return f

def getBrakeForce(veo):
    # UNTITLED3 根据制动曲线计算阻力大小
    #  传入速度单位为m/s
    #  阻力单位为KN
    f = 0
    u = veo
      # 单位换算
    if u <= 70:
        f = 563.1
    else:
        f = 0.03773 * u * u - 11.77 * u +1194
    return f

def plotnihe():
    # 绘制曲线
    # 绘制牵引曲线
    x = np.arange(0, 140, 0.1)
    y = np.vectorize(getTrateForce)(x)
    plt.rcParams["font.sans-serif"] = ['simhei']  # 设置字体
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams['font.size'] = 14

  #绘制实际数据，观察拟合效果
    df = pd.read_excel('./raw_data/列车牵引力制动力特性.xlsx', sheet_name=None)
    v = df['Sheet1']['速度(km/h)'].values
    f = df['Sheet1']['牵引力(kN)'].values

    plot1 = pylab.plot(v, f, '*', label='原始数据')
    plot2 = pylab.plot(x, y, 'r', label='拟合数据')
    pylab.title('最大牵引力')
    pylab.xlabel('速度(km/h)')
    pylab.ylabel('力(kN)')
    pylab.legend(loc=3, borderaxespad=0., bbox_to_anchor=(0, 0))
    pylab.grid()
    pylab.show(format='svg')

def nihe():
    df = pd.read_excel('D:/桌面/study/code-study/rl4metro-main1/曲线拟合.xlsx', sheet_name='Sheet1')
    x = df['x1'].values
    y = df['y1'].values

    z1 = np.polyfit(x, y, 2)  # 曲线拟合，返回值为多项式的各项系数
    p1 = np.poly1d(z1)  # 返回值为多项式的表达式，也就是函数式子

    print(p1)
    y_pred = p1(x)  # 根据函数的多项式表达式，求解 y
    # print(np.polyval(p1, 29))             根据多项式求解特定 x 对应的 y 值
    # print(np.polyval(z1, 29))             根据多项式求解特定 x 对应的 y 值

    plot1 = pylab.plot(x, y, '*', label='真实数据')
    plot2 = pylab.plot(x, y_pred, 'r', label='拟合数据')
    pylab.title('')
    pylab.xlabel('')
    pylab.ylabel('')
    pylab.legend(loc=3, borderaxespad=0., bbox_to_anchor=(0, 0))
    pylab.grid()
    pylab.show()





if __name__ == "__main__":
    plotnihe()


