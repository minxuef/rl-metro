import os
from collections import defaultdict
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


class Metro:
    def __init__(self, parameters, env, id, direction, init_time, mass, stations):
        # 初始化函数，用于创建 Metro 类的实例并设置初始属性值
        # 参数：parameters，包含系统参数的字典
        # 参数：env，仿真环境
        # 参数：id，列车的唯一标识符
        # 参数：init_time，列车的初始时间
        # 参数：direction，列车的行驶方向 (0 或 1)
        # 参数：mass，列车的质量
        # 参数：stations，列车经过的站点列表

        self.parameters = parameters
        self.env = env
        self.id = id
        self.direction = direction
        self.name = str(direction) + '-' + str(id)
        self.init_time = init_time
        self.current_speed = 0  # 列车当前时刻的速度
        # Variables
        self.stop_time = 0  # 列车停留时间
        self.cruise_speed = 0  # 列车滑行时间

        self.deviation_time_info = []
        self.deviation_time = 0  # 记录列车在下一站的到达偏差时间
        # 根据列车的方向，设置列车的出发时间
        # 如果列车方向为0，设置列车的出发时间为初始时间加上列车编号乘以系统参数中的时间间隔
        # 如果列车方向为1，设置列车的出发时间为初始时间加上1e-9再加上列车编号乘以系统参数中的时间间隔
        # 1e-9用于避免浮点数精度问题导致出发时间重叠
        if direction == 0:
            self.departure_time = self.init_time + (self.id) * self.parameters["intervals"]
            # self.stations = stations
        elif direction == 1:
            self.departure_time = self.init_time + 1e-9 + (self.id) * self.parameters["intervals"]
            # self.stations = stations[::-1]
        self.stations = stations

        # 初始化当前站点上一站点和下一站点
        self.last_metro_station = None
        self.cur_metro_station = self.stations[0]
        self.next_metro_station = self.stations[1]

        # 初始化其他变量，如信息字典、距离、速度、力学和时间信息列表。设置当前交通状态为“停止”，并启动列车运行过程。
        self.wait_action = False
        self.finished = False

        self.info = defaultdict(dict)
        self.distance_info = []
        self.speed_info = []
        self.mechanics_info = []

        self.time_info = []

        self.cur_speed_info = []  # 每个时刻对应的速度信息
        self.cur_power_info = []  # 每个时刻对应的功率信息

        self.cur_traffic_state = "stop"  # 列车运行工况
        self.process = self.env.process(self.running())
        # self.env.process()将一个方法（或函数）转换为 SimPy 进程将在仿真环境中作为一个独立的进程运行，以模拟列车的行为。
        self.total_time = []  # 用来记录运行时间
        self.traction_time = []  # 用来记录牵引时间


        self.deviation_time_info = {}  # 记录该车在每个站点的偏差时间
        self.scheduled_info = []  # 记录该车在每个站点的计划到达时间
        self.metro_station_intervals = self.parameters["metro_station_intervals"]  # 获得了正向的站间运行时间间隔
        self.metro_station_dwell_time = self.parameters["metro_station_dwell_time"]  # 获得正向每个站的停留时间(从第二个站开始)

    # 获取该车的初始时刻表

    def get_cur_state(self):  # 获取当前信息，如果当前状态不是下列四种状态之一返回异常，防止代码错误

        if self.cur_traffic_state == "stop":
            label = 's'
        elif self.cur_traffic_state == "traction":
            label = 't'
        elif self.cur_traffic_state == "cruise":
            label = 'c'
        elif self.cur_traffic_state == "brake":
            label = 'b'
        else:
            raise ValueError
        return label

    # 用于等待指令。它首先标记当前步骤成功，然后设置一个新的事件。在等待过程中，它将等待 continue_criteria 事件
    def wait_instruction(self):
        self.parameters['step_criteria'].succeed()
        self.parameters['step_criteria'] = self.env.event()

        self.wait_action = True
        yield self.parameters['continue_criteria']  # 等待执行continue
        self.wait_action = False

    def running(self):
        n = 0  # 当前站的执行过程
        self.deviation_time_info = []  # 初始化
        while True:
            # 初始站点和出发时间
            if n == 0:
                assert self.departure_time >= 0  # 确保出发时间是正的
                yield self.env.timeout(self.departure_time)  # 暂停直到到达出发时间
                self.info['departure'] = self.env.now  # 记录出发时间
                if self.direction == 0:
                    self.scheduled_time = self.departure_time + self.metro_station_intervals[0]
                    self.scheduled_info.append(self.scheduled_time)

                elif self.direction == 1:
                    self.scheduled_time = self.departure_time + self.metro_station_intervals[-1]

            if n != 0:
                if self.direction == 0:
                    self.scheduled_time += self.metro_station_intervals[n] + self.metro_station_dwell_time[n - 1]
                elif self.direction == 1:
                    self.scheduled_time += self.metro_station_intervals[n] + self.metro_station_dwell_time[-n - 1]

            # 决策过程
            yield self.env.process(self.wait_instruction())  # 等待指令
            total_time, traction_time, cruise_time, brake_time = self.time_model(self.stop_time, self.cruise_speed,
                                                                                 self.cur_metro_station)
            self.total_time.append(total_time)
            self.traction_time.append(traction_time)

            self.actual_time = total_time - self.stop_time + self.env.now

            deviation_time = self.actual_time - self.scheduled_time

            self.deviation_time = abs(deviation_time)

            self.deviation_time_info.append(abs(deviation_time))

            # cur_time是在每个站的出发时间，
            self.info[self.cur_metro_station.name] = {'cur_time': self.env.now, 'traction_time': traction_time,
                                                      'cruise_time': cruise_time, 'brake_time': brake_time,
                                                      'stop_time': self.stop_time
                                                      }  # 更新站点信息

            # --------下面为执行过程--------------------------------------------------------------------------------

            self.cur_speed_info.append([math.ceil(self.env.now), self.current_speed])

            # 牵引阶段
            self.cur_traffic_state = "traction"  # 设置当前状态为牵引
            for t in np.arange(0, traction_time, self.parameters['time_step']):
                _, _, speed_list, power_list, force_list = self.mechanics_model(self.get_cur_state(),
                                                                                self.cruise_speed)
                if t < len(speed_list):
                    self.cur_speed_info.append([math.ceil(self.env.now + t), speed_list[t]])
                if t < len(power_list):
                    self.cur_power_info.append([math.ceil(self.env.now + t), power_list[t]])
                if t < len(force_list):
                    self.mechanics_info.append([math.ceil(self.env.now + t), force_list[t]])

            yield self.env.timeout(traction_time)

            self.time_info.append(
                dict(traction_time=[self.env.now - traction_time, self.env.now]))

            # 巡航阶段
            self.cur_traffic_state = "cruise"
            force = self.get_force("c", self.cruise_speed)
            for t in np.arange(0, cruise_time, self.parameters['time_step']):
                self.cur_speed_info.append([math.ceil(self.env.now + t), self.cruise_speed])
                self.cur_power_info.append([math.ceil(self.env.now + t), force*self.cruise_speed])
                self.mechanics_info.append([math.ceil(self.env.now + t),force])
            yield self.env.timeout(cruise_time)

            self.time_info.append(
                dict(cruise_time=[self.env.now - cruise_time, self.env.now]))

            # 处理制动阶段
            self.cur_traffic_state = "brake"
            for t in np.arange(0, brake_time, self.parameters['time_step']):
                _, _, speed_list, power_list, force_list = self.mechanics_model(self.get_cur_state(),
                                                                                self.cruise_speed)
                if t < len(speed_list):
                    self.cur_speed_info.append([math.ceil(self.env.now + t), speed_list[t]])
                if t < len(power_list):
                    self.cur_power_info.append([math.ceil(self.env.now + t), power_list[t]])
                if t < len(force_list):
                    self.mechanics_info.append([math.ceil(self.env.now + t), force_list[t]])

            yield self.env.timeout(brake_time)

            self.time_info.append(dict(brake_time=[self.env.now - brake_time, self.env.now]))

            # 处理停站阶段
            self.cur_traffic_state = "stop"
                        # train时为了加快训练速度，不留存详细信息
                        # test时解除注释
            for t in np.arange(0, self.stop_time, self.parameters['time_step']):
                self.current_speed = 0
                self.cur_speed_info.append([math.ceil(self.env.now) + t, self.current_speed])
                self.cur_power_info.append([math.ceil(self.env.now + t), 0])
                self.mechanics_info.append([math.ceil(self.env.now + t), 0])


            yield self.env.timeout(self.stop_time)

            # print(f"station: {self.cur_metro_station} stop_time: {self.stop_time}")

            self.time_info.append(dict(stop_time=[self.env.now - self.stop_time, self.env.now]))

            # 确定前后站，下一站的执行过程
            n += 1
            self.last_metro_station = self.stations[n - 1]  # 上一站
            self.cur_metro_station = self.stations[n]  # 下一站
            if n >= len(self.stations) - 1:  # 如果到达了倒数第二个站，接近终点站
                self.next_metro_station = None
                self.finished = True
                if all([m.finished for m in self.parameters['metros']]) == True:  # 如果所有列车都完成
                    self.parameters['step_criteria'].succeed()  # 标记步骤成功
                # self.parameters['global_info'][self.id][-1] = 1
                break  # 终止循环
            else:
                self.next_metro_station = self.stations[n + 1]

            # 扰动
            if not self.finished:
                # disturbance = random.choice(range(0, 14))  # 加入随机扰动
                disturbance = 10
                yield self.env.timeout(disturbance)  # 等待扰动的时间
                self.info[self.cur_metro_station.name]['disturbance'] = disturbance
                # print(f"metro {self.id} at station {self.cur_metro_station.name} disturbance: {self.info[self.cur_metro_station.name]['disturbance']}")

    def get_force(self, label, cur_speed):
        # 计算该速度下的受力
        # 传入速度m/s
        # 返回力kN
        u = cur_speed * 3.6
        a = 2.031
        b = 0.062
        c = 0.019
        # 基本阻力
        fdavis = a + b * u + c * u * u

        if label == "t":  # 牵引
            if u <= 42:
                f = 558
            else:
                f = -0.0005 * u * u * u + 0.18 * u * u - 22.67 * u + 1223
            #print(f"牵引fdavis: {fdavis} f: {f}")
            return f * self.parameters['fbeta1'] - fdavis

        elif label == "b":  # 制动
            if u <= 70:
                f = 563.1
            else:
                f = 0.038 * u * u - 11.77 * u + 1194
            #print(f"制动fdavis: {fdavis} f: {f}")
            return -f * self.parameters['fbeta2'] - fdavis

        elif label == "c":  # 巡航
            force = fdavis / self.parameters['fbeta1']
            #print(f"巡航fdavis: {fdavis} force: {force}")
            return force

        elif label == "s":  # 停车
            force = 0
            return force

    def mechanics_model(self, label, vc):
        # 动力学过程计算牵引制动过程的运动信息
        # vc 为巡航速度m/s
        # 返回距离，时间，以及信息列表
        t = 0
        a = 0
        s = 0
        cur_speed = []
        cur_power = []
        v = 0
        cur_force = []
        if label == 't':
            cur_speed.append(v)
            cur_power.append(0)

            while True:
                s_change = v * self.parameters['time_step']
                s = s + s_change

                force = self.get_force(label, v)
                cur_force.append(force)
                a = force*1000 / self.parameters['metro_mass']
                cur_power.append(force * v)
                v += a * self.parameters['time_step']
                #print(f"label:{label}force: {force}N  a: {a} v: {v} vc: {vc}")
                if v >= vc:
                    v = vc
                    cur_speed.append(v)
                    #print('break')
                    break
                # cur_speed.append(v)
                t += self.parameters['time_step']
            return t, s, cur_speed, cur_power, cur_force

        elif label == 'b':
            v = vc
            cur_speed.append(v)
            while True:

                s_change = v * self.parameters['time_step']
                s = s + s_change
                force = self.get_force(label, v)
                cur_force.append(force)
                a = force*1000 / self.parameters['metro_mass']

                cur_power.append(force * v)
                v += a * self.parameters['time_step']

                if v<=0:
                    v=0
                    cur_speed.append(v)
                    #print('break')
                    break
                cur_speed.append(v)
                t += self.parameters['time_step']

            return t, s, cur_speed, cur_power, cur_force

    def time_model(self, stop_time, cruise_speed, metro_station):
        # 返回总时间，牵引时间，巡航时间，制动时间，总时间= 牵引时间 + 制动时间 + 巡航时间 +停站时间
        assert cruise_speed is not None
        traction_time, traction_distance, _, _, _ = self.mechanics_model('t', cruise_speed)
        brake_time, brake_distance, _, _, _ = self.mechanics_model('b', cruise_speed)

        cruise_distance = metro_station.dis2nextstation - traction_distance - brake_distance
        self.distance_info.append([traction_distance, cruise_distance, brake_distance])

        if cruise_distance < 0:
            raise ValueError(
                f"metro_name: {self.name} | metro_station: {metro_station.name} | distance:{metro_station.dis2nextstation} | t_distance: {traction_distance} | b_distance: {brake_distance}")
        cruise_time = cruise_distance / cruise_speed
        total_time = stop_time + traction_time + brake_time+ cruise_time
        print( f"total_time: {total_time} | traction_time: {traction_time} | cruise_time: {cruise_time} | brake_time: {brake_time}")
        return total_time, traction_time, cruise_time, brake_time



    def get_tra_energy(self):
        # 计算牵引能耗，没用到
        tra_energy = 0
        for entry in self.cur_power_info:
            power = entry[1]
            if power > 0:
                tra_energy += power
        return tra_energy



    def get_speed_info(self):
        t_v = list(list(zip(*self.cur_speed_info))[0])
        v = list(list(zip(*self.cur_speed_info))[1])
        return t_v, v

    def export_info_to_excel(self):

        '''
        data = pd.DataFrame(self.cur_power_info, columns=['Time', 'Power'])
        excel_file_path = f'D:/桌面/study/code-study/rl4metro-main1/rl4metro-main 3.5/results/metro {self.name}.xlsx'
        data.to_excel(excel_file_path, index=False)

        speed = pd.DataFrame(self.cur_speed_info, columns=['Time', 'Speed'])
        excel_file_path1 = f'D:/桌面/study/code-study/rl4metro-main1/rl4metro-main 3.5/results/ {self.name}.xlsx'
        speed.to_excel(excel_file_path1, index=False)
    '''

        df = pd.DataFrame()
        # 遍历 self.info 字典，将每个站点的信息添加到 DataFrame 中
        for station, info in self.info.items():
            # 将字典转换为 DataFrame 行
            row = pd.DataFrame([info], index=[station])
            # 将行添加到 DataFrame 中
            df = pd.concat([df, row], ignore_index=True)
            excel_file_path = f'./results/time_info{self.name}.xlsx'
            # 保存 DataFrame 到 Excel 文件，index=False 表示不保存行索引
            df.to_excel(excel_file_path)

    def get_total_time(self):
        if self.total_time == []:
            total_time = 0
        else:
            total_time = sum(self.total_time)
        return total_time

    def get_deviation_time(self):
        # total_deviation = sum(self.deviation_time_info)
        # return total_deviation
        return self.deviation_time

    def get_traction_time(self):
        if self.traction_time == []:
            return 0
        return sum(self.traction_time)

    def render(self):
        t_v = list(list(zip(*self.cur_speed_info))[0])
        v = list(list(zip(*self.cur_speed_info))[1])

        t_f = list(list(zip(*self.mechanics_info))[0])
        f = list(list(zip(*self.mechanics_info))[1])

        t_p = list(list(zip(*self.cur_power_info))[0])
        p = list(list(zip(*self.cur_power_info))[1])

        fig, ax = plt.subplots(3, 1, figsize=(25, 10))

        # 速度图
        ax1 = ax[0]
        ax1.set_title(f"metro{self.name} speed graph")
        # ax1.set_ylim(bottom = 0)
        ax1.set_ylabel('speed (m/s)')
        ax1.set_xlabel('time (s)')
        ax1.plot(t_v, v)
        # 力图
        ax2 = ax[1]
        ax2.set_title(f"metro{self.name} mechanics graph")
        ax2.set_ylabel('force(N)')
        ax2.set_xlabel('time (s)')
        ax2.plot(t_f, f)

        # 功率图
        ax3 = ax[2]
        ax3.set_title(f"metro{self.name} power graph")
        ax3.set_ylabel('power (W)')
        ax3.set_xlabel('time (s)')
        ax3.plot(t_p, p)
        # 能耗图
        # ax3 = ax[2]

        file_name = f'metro {self.name}.svg'
        file_path = os.path.join('./results', file_name)
        plt.savefig(file_path)
        plt.show()

