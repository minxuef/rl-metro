import simpy
import gym
import math
import numpy as np
import matplotlib.pyplot as plt
from environment.metro import Metro
from environment.metro_station import MetroStation
from plot import plot_3d
from collections import defaultdict


# 导入必要的库
# simpy 是一个事件驱动的模拟库，用于模拟系统中的事件和进程。
# gym 是一个用于创建和使用强化学习环境的库。
# math 和 numpy 提供数学和数值计算功能。
# matplotlib.pyplot 是一个绘图库，用于创建图表和可视化数据。

class MetroEnvEventbased(gym.Env):
    # 继承自gym.Env
    def __init__(self, parameters) -> None:
        super().__init__()  # 调用父类（gym.Env）的初始化方法。

        self.num_envs = 1

        self.metro_station_intervals = None
        self.metro_station_dwell_time = None
        self.parameters = parameters  # 保存传入的环境参数
        # self.seed(seed=self.parameters["seed"])
        self.first_metro_time = parameters["first_metro_time"]
        self.last_metro_time = parameters["last_metro_time"]
        self.global_clock = 0
        # 设置地铁开始和结束的时间，初始化全局时钟

        self.overlap_time = []  # 每步的重叠时间
        self.overlap_time_episodes = []  # 每个episode的重叠时间

        self.tr_energy_episodes = []  # 用于保存每个episode的能耗
        self.re_energy_episodes = []
        self.energy_episodes = []  # 用于保存每个episode的能耗

        self.episode_rewards = []  # 用于保存每个episode的奖励
        self.step_rewards = []  # 用于保存每个step的奖励,并计算回合奖励
        self.step_rewards_list = []  # 用于保存训练过程中每个step的奖励，方便画图

        self.total_deviation_time = []  # 用于保存每一步的总时间偏差
        self.total_deviation_time_episodes = []  # 用于保存每个episode的总时间偏差
        self.total_traction_time = []  # 用于保存每一步的总牵引时间
        self.total_traction_time_episodes = []  # 用于保存每个episode的总牵引时间
        # 定义动作空间
        # 停站时间 and 巡航时间
        # self.action_space = gym.spaces.Box(low=np.array([self.parameters['stop_time_low'], self.parameters['cruise_speed_low']]), 
        #                                    high=np.array([self.parameters['stop_time_upper'], self.parameters['cruise_speed_upper']]), shape=(2,), dtype=np.float64)
        self.action_space = gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), shape=(2,), dtype=np.float64)
        self.observation_space = gym.spaces.Box(-1, np.inf, shape=(
            1, (self.parameters['num_metros'] - 1) * (self.parameters["num_metro_stations"] * 4 + 4)), dtype=np.float64)
        # 状态空间特征包括历史加速时间点、历史恒速时间点、历史减速时间点、历史停留时间点；当前位置、到下一站的距离、列车方向和列车是否到达终点站。

    def step(self, action):
        # action: 停车等待时间 and 巡航速度

        # print(action)

        # 使用了线性插值，将动作值映射到预定义的停车时间和巡航速度范围。
        stop_time = ((self.parameters['stop_time_upper'] - self.parameters['stop_time_low']) * (action[0] + 1)) / 2 + \
                    self.parameters['stop_time_low']
        cruise_speed = ((self.parameters['cruise_speed_upper'] - self.parameters['cruise_speed_low']) * (
                action[1] + 1)) / 2 + self.parameters['cruise_speed_low']
        # print(stop_time, cruise_speed)

        self.steps += 1  # 记录仿真的步数

        # 遍历所有地铁列车，找到正在等待动作的列车（wait_action 为 True）。
        # 为这辆列车设置停车时间和巡航速度，并更新 continue_criteria 事件，以便列车可以继续运行。
        for m in self.metros:
            if m.wait_action == True:
                decision_metro = m  # 找到等待动作的列车
                m.stop_time = stop_time  # 设置停车时间
                m.cruise_speed = cruise_speed  # 设置巡航速度
                print("metro", m.name, "stop_time:", m.stop_time, "cruise_speed:", m.cruise_speed)
                self.parameters['continue_criteria'].succeed()  # 更新继续条件事件，通知列车继续运行
                self.parameters['continue_criteria'] = self.env.event()  # 创建继续条件事件
                break  # 找到一辆车后退出循环

        # 运行仿真环境，直到达到预定义的步骤结束条件。
        self.env.run(until=self.parameters['step_criteria'])

        # 获取下一个观察值和奖励。这些值将用于智能体的学习过程。
        next_obs = self.get_observation(decision_metro)
        reward = self.get_reward(decision_metro)
        self.step_rewards.append(reward)
        self.step_rewards_list.append(reward)

        # 所有列车都已经到达终点，设置 done 为 True，表示一个回合结束
        done = True if all([m.finished for m in self.metros]) else False
        if done:
            self.episode_rewards.append(sum(self.step_rewards))
            self.overlap_time_episodes.append(self.overlap_time[-1])
            self.total_deviation_time_episodes.append(self.total_deviation_time[-1])
            self.total_traction_time_episodes.append(self.total_traction_time[-1])
            tra, re = self.caculate_energy_total()
            self.tr_energy_episodes.append(tra)
            self.re_energy_episodes.append(re)
            self.energy_episodes.append(tra - re)
            print(
                f"回合结束episode_reward ={sum(self.step_rewards)} | 重叠时间 = {self.overlap_time[-1]} | 总时间偏差 = {self.total_deviation_time[-1]}")

        return next_obs, reward, done, {'steps': self.steps}
        # 返回下一个观察值、奖励、是否结束的布尔值以及一个包含步数的字典

    def reset(self, seed=2023):

        # 初始化环境，包括设置随机种子和重置环境状态。
        self.env = simpy.Environment()
        self.cur_time = 0
        self.steps = 0
        self.old_reward_matrix = np.zeros((2, 1))
        # 更新环境参数，包括步骤结束条件和继续条件。
        self.parameters.update({'step_criteria': self.env.event()})
        self.parameters.update({'continue_criteria': self.env.event()})
        self.config_environment()
        self.env.run(until=self.parameters['step_criteria'])

        self.step_rewards = []

        # 获取初始观察值并返回。
        obs = self.get_observation()
        return obs

    def get_observation(self, decision_metro=None):
        # 初始化观察矩阵，观察其他车的状态。加速时间点、减速时间点、恒速时间点、停留时间点；当前位置、距离，列车方向、是否到达终点
        obs = np.zeros((self.num_metros - 1, self.num_metro_stations * 4 + 4))
        # 如果没有指定决策矩阵，直接返回全零矩阵
        if decision_metro == None:
            return obs.reshape(1, -1)
        # 如果指定了决策地铁，遍历其他地铁列车，构建观察值
        for i, m in enumerate([m for m in self.metros if m is not decision_metro]):
            if m.time_info == []:  # 如果列车没有时间信息，观察值为零
                observation = []
            else:
                # 获取时间信息并构建观察值
                observation = [list(x.values())[-1][-1] for x in m.time_info]
            observation.extend(0 for _ in range(self.num_metro_stations * 4 - len(observation)))
            observation.extend([m.cur_metro_station.id, m.direction, m.cur_metro_station.dis2nextstation, m.finished])
            obs[i] = observation
        return obs.reshape(1, -1)

    def get_reward(self, decision_metro=None):
        # 初始化奖励矩阵，两行，列数为当前仿真时间向上取整，确保即使当前时间不是整数，列数也能正确计算。
        # 在某一时间段内牵引和制动的列车的数量

        rewards_matrix = np.zeros((2, math.ceil(self.env.now)))  # 第二个维度的大小是当前时间的大小
        # 遍历所有地铁列车，更新奖励矩阵
        for i, m in enumerate(self.metros):
            # 先读取time—info
            if m.time_info != []:
                for info in m.time_info:
                    # 获取时间信息的类型
                    # if (m.direction==0 and list(info.keys())[0] == 'traction_time') or (m.direction==1 and list(info.keys())[0] == 'brake_time'):
                    # 如果时间信息类型是牵引时间（'traction_time'）， 在牵引时间的区间内，奖励矩阵的第一行对应列增加1。
                    # `math.ceil` 确保时间区间的起始和结束位置都是整数。
                    if list(info.keys())[0] == 'traction_time' or list(info.keys())[0] == 'cruise_time':
                        rewards_matrix[0, math.ceil(list(info.values())[0][0]):math.ceil(
                            list(info.values())[0][1])] += 1  # 牵引时间段内的列增加1
                    # elif (m.direction==0 and list(info.keys())[0] == 'brake_time') or (m.direction==1 and list(info.keys())[0] == 'traction_time'):
                    elif list(info.keys())[0] == 'brake_time':
                        rewards_matrix[1, math.ceil(list(info.values())[0][0]):math.ceil(
                            list(info.values())[0][1])] += 1
                    # 如果是停留时间和巡航时间就跳过
                    elif list(info.keys())[0] == 'stop_time':
                        continue
                    else:
                        # 如果不是上述的任何一种就抛出异常
                        raise AssertionError

        total_run_time = sum([metro.get_total_time() for metro in self.metros])
        total_deviation_time = sum([m.get_deviation_time() for m in self.metros])
        total_traction_time = sum([m.get_traction_time() for m in self.metros])

        deviation_time = decision_metro.get_deviation_time()
        # 计算奖励值。这里使用 `np.clip` 确保奖励值在0到正无穷之间。
        # 奖励值是当前奖励矩阵的最小值之和减去旧奖励矩阵的最小值之和。
        # 计算奖励矩阵在每一列上的最小值，得到一个包含所有时间步最小奖励的奖励矩阵
        self.overlap_time.append(np.clip(rewards_matrix.min(axis=0), 0, np.inf).sum())
        self.total_deviation_time.append(total_deviation_time)
        self.total_traction_time.append(total_traction_time)
        # r1 = np.clip(rewards_matrix.min(axis=0), 0, np.inf).sum() - np.clip(self.old_reward_matrix.min(axis=0), 0,
        #                                                                   np.inf).sum()
        r1 = (np.clip(rewards_matrix.min(axis=0), 0, np.inf).sum() / self.steps)** 2
        r2 = -(total_traction_time / self.steps)** 2
        # r3 = -total_deviation_time / self.steps * 0.5
        # r1 = np.clip(rewards_matrix.min(axis=0), 0, np.inf).sum()
        r3 = -( total_deviation_time** 2/ self.steps)

        '''
        if r1 > 5:
            r1 = 10
        elif 0 < r1 <= 5:
            r1 = 5
        else:
            r1 = -2

        if 0 < deviation_time <= 30:
            r2 = 5
        elif 30 < deviation_time <= 60:
            r2 = 1  # 偏差时间较短
        elif 60 < deviation_time <= 120:
            r2 = -2  # 偏差时间适中
        else:
            r2 = -5  # 偏差时间较长
        '''
        r = 0.6*(0.8*r1+0.2*r2)+0.8*r3
        #r = 0.8*r3+0.8*r1

        print(f'奖励值：{r}，r1：{r1}，r2：{r2}, r3: {r3}')

        self.old_reward_matrix = rewards_matrix

        return r

    def config_environment(self):
        # # 真实数据
        self.metro_stations_name_list = self.parameters["metro_stations_name_list"]
        # ~m
        metro_station_distances = self.parameters["metro_station_distances"]
        self.metro_station_intervals = self.parameters["metro_station_intervals"]  # 获得了正向的站间运行时间间隔
        self.metro_station_dwell_time = self.parameters["metro_station_dwell_time"]  # 获得正向每个站的停留时间(从第二个站开始)

        # 初始化车站数据：正向行驶+反向行驶
        self.num_metro_stations = len(self.metro_stations_name_list)  # 计算地铁站的数量，名称列表的长度
        self.metro_stations_forward = [MetroStation(self.parameters, self.env, i, self.metro_stations_name_list[i],
                                                    metro_station_distances[
                                                        i] if i < self.num_metro_stations - 1 else 0) for i in
                                       range(self.num_metro_stations)]
        self.metro_stations_backward = [
            MetroStation(self.parameters, self.env, self.num_metro_stations - i - 1, self.metro_stations_name_list[::-1]
            [i], metro_station_distances[::-1][i] if i < self.num_metro_stations - 1 else 0) for i in
            range(self.num_metro_stations)]

        # 初始化列车数据   列车数量一分为二，分别代表正向和反向行驶的列车，每个列车都有唯一的id、方向（0表示正向，1表示反向）
        self.num_metros = self.parameters["num_metros"]  # 获取地铁列车的数量
        self.metros = [Metro(parameters=self.parameters, env=self.env, id=i, direction=0, init_time=0, mass=40,
                             stations=self.metro_stations_forward) for i in range(
            self.num_metros // 2)] + [
                          Metro(parameters=self.parameters, env=self.env, id=i, direction=1, init_time=0, mass=40,
                                stations=self.metro_stations_backward) for i in range(self.num_metros // 2)]

        # 将初始化的列车和车站列表添加到参数字典中，便于调用
        self.parameters['metros'] = self.metros
        self.parameters['metro_stations_forward'] = self.metro_stations_forward
        self.parameters['metro_station_backward'] = self.metro_stations_backward

    # 计算牵引能耗和再生能
    def caculate_energy_total(self):
        re_energy_total = 0  # 再生能
        positive_energy = 0
        tra_energy_total = 0  # 牵引能耗
        re = 0
        regenerate_energy = defaultdict(lambda: defaultdict(float))
        for m in self.metros:
            for time, power in m.cur_power_info:
                if power >= 0:
                    if time in regenerate_energy:
                        regenerate_energy[time][0] += power
                    else:
                        regenerate_energy[time][0] = power
                elif power < 0:
                    if time in regenerate_energy:
                        regenerate_energy[time][1] += -power * self.parameters['fbeta3']
                    else:
                        regenerate_energy[time][1] = -power * self.parameters['fbeta3']

        for value in regenerate_energy.values():
            tra_energy_total += value[0]
            re_energy_total += min(value[0], value[1])
        return tra_energy_total / 3600000, re_energy_total / 3600000

    def render(self, mode="human"):
        color_li = self.parameters['colors']
        plt.rcParams["font.sans-serif"] = ['simhei']  # 设置字体
        plt.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(figsize=(15, 10))  # 设置图表大小
        cum_dis = np.cumsum([m.dis2nextstation for m in self.metro_stations_forward])  # 计算地铁站点之间的累积距离，这将用于在图表中表示站点位置。
        # plt.yticks(list(range(0, len(self.metro_stations))),
        #           [str(m.name) for m in self.metro_stations])
        ax.set_yticks(list(np.insert(cum_dis, 0, 0)[:-1]))  # 设置刻度
        ax.set_yticklabels([str(m.name) for m in self.metro_stations_forward],
                           rotation=30, fontsize=28)  # 设置刻度标签

        # 运行时图，遍历所有地铁列车，绘制运行图，每个列车的轨迹用不同颜色表示
        for metro in self.metros:
            if metro.direction == 0:
                dot_list_y = []
                dot_list_x = []
                # t_tmp = metro.departure_time
                n = 0
                for k, v in metro.info.items():
                    if k == 'departure':
                        dot_list_x.append(v)
                        dot_list_y.append(
                            list(np.insert(cum_dis, 0, 0)[:-1])[n])
                    else:
                        dot_list_y.append(
                            list(np.insert(cum_dis, 0, 0)[:-1])[n + 1])
                        dot_list_x.append(np.cumsum(list(v.values()))[3])

                        dot_list_y.append(
                            list(np.insert(cum_dis, 0, 0)[:-1])[n + 1])
                        dot_list_x.append(np.cumsum(list(v.values()))[-1])
                        n += 1
                        if n == self.parameters['num_metro_stations'] - 1:
                            break
                plt.plot(dot_list_x, dot_list_y, color=np.random.choice(
                    color_li), label=f'Metro-Train-Station')
            else:
                dot_list_y = []
                dot_list_x = []
                # t_tmp = metro.departure_time
                n = self.parameters['num_metro_stations'] - 1
                for k, v in metro.info.items():
                    if k == 'departure':
                        dot_list_x.append(v)
                        dot_list_y.append(
                            list(np.insert(cum_dis, 0, 0)[:-1])[n])
                    else:
                        dot_list_y.append(
                            list(np.insert(cum_dis, 0, 0)[:-1])[n - 1])
                        dot_list_x.append(np.cumsum(list(v.values()))[3])

                        dot_list_y.append(
                            list(np.insert(cum_dis, 0, 0)[:-1])[n - 1])
                        dot_list_x.append(np.cumsum(list(v.values()))[-1])
                        n -= 1
                        if n == 0:
                            break
                plt.plot(dot_list_x, dot_list_y, label=f'Metro-Train-Station')

        # 为图表添加网格线
        plt.grid(axis='y', linestyle='-.',
                 linewidth=1, color='black', alpha=0.5)
        plt.xticks(fontsize=28)
        plt.xlabel('时间 (s)', fontsize=28)
        plt.ylabel('地铁站', fontsize=28)
        plt.savefig('./results/运行图.png')
        plt.show()

        # 每个车的速度曲线
        # plot_3d(self)

        # render每个车的速度加速度以及功率信息
        for m in self.metros:
            m.render()
            m.export_info_to_excel()

        # 将每列车的速度曲线画到同一张图上
        x = []
        y = []
        for m in self.metros:
            i = 0
            x.append(m.get_speed_info()[0])
            y.append(m.get_speed_info()[1])
            i += 1

        for i in range(len(x)):
            plt.plot(x[i], y[i], label=f'Metro-Train-{i}')
        plt.xlabel('时间 (s)', fontsize=14)
        plt.ylabel('速度 (m/s)', fontsize=14)
        plt.legend()
        plt.savefig('./results/合并速度曲线.png')
        plt.show()
