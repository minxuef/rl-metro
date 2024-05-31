import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def smooth(data, weight=0.9):
    '''用于平滑曲线，类似于Tensorboard中的smooth

    Args:
        data (List):输入数据
        weight (Float): 平滑权重，处于0-1之间，数值越高说明越平滑，一般取0.9

    Returns:
        smoothed (List): 平滑后的数据
    '''
    last = data[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_rewards(rewards, path=None, tag='train'):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(
        f"{tag}ing curve ")
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(smooth(rewards), label='smoothed')
    plt.legend()


def plot_energy(cfg):
    seed = cfg['seed']
    np.random.seed(seed)
    plt.figure()
    plt.title('Energy Accumulation Testing Result')
    plt.xlabel('time[s]')
    plt.ylabel('energy[kwh]')
    y = np.random.randn()

def plot_3d(env):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for m in env.metros:
        y = m.get_speed_info()[0]
        x = np.linspace(0, 8, len(y))
        z = m.get_speed_info()[1]
        ax.plot(x, y, z, label=f'{m.name} Speed')

    ax.set_xlabel('车辆', fontsize=14)
    ax.set_ylabel('时间(s)', fontsize=14)
    ax.set_zlabel('速度 (m/s)', fontsize=14)
    ax.legend('best')
    plt.savefig('./results/速度曲线_3D.png')
    plt.show()

if __name__ == '__main__':
    plot_rewards(path = './logs/2024-0319-16-49-08-PPO-8-24/monitor.csv')