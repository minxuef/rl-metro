import argparse
import datetime as dt
import pandas as pd
import numpy as np


def get_args():
    """ 超参数
    """
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--algo_name', default='PPO',
                        type=str, help="name of algorithm")
    parser.add_argument('--env_name', default='Metro-Train-Env',
                        type=str, help="name of environment")


    parser.add_argument('--train_eps', default=2000,
                        type=int, help="episodes of training")



    parser.add_argument('--test_eps', default=1500, type=int,
                        help="episodes of testing")
    parser.add_argument('--max_steps', default=1500, type=int,
                        help="steps per episode, much larger value can simulate infinite steps")
    parser.add_argument('--gamma', default=0.99,
                        type=float, help="discounted factor")
    parser.add_argument('--critic_lr', default=1e-3,
                        type=float, help="learning rate of critic")
    parser.add_argument('--actor_lr', default=1e-4,
                        type=float, help="learning rate of actor")
    parser.add_argument('--memory_capacity', default=8000,
                        type=int, help="memory capacity")
    parser.add_argument('--batch_size', default=3000, type=int)
    parser.add_argument('--target_update', default=2, type=int)
    parser.add_argument('--tau', default=1e-2, type=float)
    parser.add_argument('--critic_hidden_dim', default=256, type=int)
    parser.add_argument('--actor_hidden_dim', default=256, type=int)
    parser.add_argument('--device', default='cuda',
                        type=str, help="cpu or cuda")
    parser.add_argument('--seed', default=23, type=int, help="random seed")
    parser.add_argument('--test_mode', default=False, type=int, help="for test")
    parser.add_argument('--log_path', default='./logs')
    args = parser.parse_args([])
    args = {**vars(args)}  # 将args转换为字典
    return args


def get_env_args(args):
    args.update({'time_step': 1})  #步长1s
    args.update({'num_metros': 4})
    args.update({"raw_data": pd.read_excel('./raw_data/成都17号线数据.xlsx', sheet_name=None)})
    # 站点数据
    args.update({"metro_stations_name_list": args["raw_data"]["Sheet1"]['站点'].tolist()})
    args.update({"num_metro_stations": len(args["metro_stations_name_list"])})
    # 站间距离 m
    args.update(
        {"metro_station_distances": (args["raw_data"]["Sheet2"]["间隔距离"] * 1000 + np.random.random() * 1000).tolist()})
    # 站间时间
   #args.update(
       # {"metro_station_intervals": (args["raw_data"]["Sheet2"]["间隔时间"].apply(lambda x: x.minute) * 60).tolist()})
    args.update(
        {"metro_station_intervals": (args["raw_data"]["Sheet2"]["区间运行时间s"]).tolist()})
    # 停站时间
    args.update({"metro_station_dwell_time": args["raw_data"]["Sheet2"]["停站时间"].tolist()})
    args.update({"station_id": [1, 2, 3]})
    args.update({"low_num_stations": 10})
    args.update({"upper_num_stations": 30})

    # 发车间隔300s
    args.update({'intervals': 300})

    # 定义调整空间
    # 停站时间平均30s
    args.update({'stop_time_low': 20.0})
    args.update({'stop_time_upper': 40.0})
    # 巡航速度m/s^2
    args.update({'cruise_speed_low': 11.0})
    args.update({'cruise_speed_upper': 23.0})

    args.update({'num_actions': 2})  # 动作空间的维度是2

    # 状态空间的维度
    # 列车数量减一是因为不包含当前列车的状态，而是关注其他列车的状态
    # +4
    args.update({'num_observations': (args['num_metros'] - 1) * (args['num_metro_stations'] * 4 + 4)})
    args.update({'first_metro_time': dt.datetime.strptime(
        '2023-02-08 07:00:00', '%Y-%m-%d %H:%M:%S')})
    args.update({'last_metro_time': dt.datetime.strptime(
        '2023-02-08 22:00:00', '%Y-%m-%d %H:%M:%S')})

    args.update(
        {'colors': ['#FFA500', '#FF4500',  '#FF6347', '#FFC0CB', '#FF0000', '#B22222',
                    '#800080', '#9400D3', '#4B0082', '#000080', '#00CED1', '#00FF00', '#008000',
                    '#7FFF00', '#FFA07A', '#FF69B4', '#32CD32', '#40E0D0',
                    '#20B2AA', '#7FFFAA', '#00FFFF', '#008080', '#6495ED', '#4169E1', '#6A5ACD',
                    '#708090', '#9400D3', '#DA70D6',  '#ADD8E6', '#9370DB',
                    '#BA55D3', '#C71585', '#D2B48C', '#DEB887']})

    # 地铁电机最大牵引力 N
    # args.update({'p1': 30 * 1000})
    # 地铁列车最大制动力 N
    # args.update({'p3': 50 * 1000})
    # 地铁电机最大牵引功率 Kw
    args.update({'pr': 190})
    # 转换效率
    args.update({'fbeta1': 0.93})  # 牵引效率
    args.update({'fbeta2': 0.94})  # 制动效率
    args.update({'fbeta3': 0.4})  # 再生能利用效率
    args.update({'metro_mass': 337.8 * 1000})  # 列车质量
    # # 打印参数
    # print("训练参数如下：")
    # print(''.join(['=']*80))
    # tplt = "{:^20}\t{:^20}\t{:^20}"
    # print(tplt.format("参数名", "参数值", "参数类型"))
    # for k, v in args.items():
    #     print(tplt.format(k, v, str(type(v))))
    # print(''.join(['=']*80))

    args.update({'test_iterations': 10})
    return args
