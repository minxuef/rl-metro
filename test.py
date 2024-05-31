import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import time
import torch
import numpy as np
from config import get_args, get_env_args
from environment.metro_env_eventbased import MetroEnvEventbased
from stable_baselines3 import PPO


def test():
    args = get_args()
    params = get_env_args(args)
    metro_env = MetroEnvEventbased(params)
    iterations = params['test_iterations']

    '''
    # 随机动作test]
    iter = 0
    time1 = time.time()
    rew_mean1 = 0
    energy_traction_mean1 = 0
    energy_re_mean1 = 0
    energy_total_mean1 = 0

    while iter < iterations:
        rew_li = []
        energy_traction_li = []
        energy_re_li = []
        energy_total_li = []

        steps = 0
        next_obs = metro_env.reset()
        for _ in range(100):
            action = torch.FloatTensor([np.random.random() * 2 - 1, np.random.random() * 2 - 1])
            next_obs, reward, done, _ = metro_env.step(action)
            rew_li.append(reward)
            tra_energy1,re_energy1= metro_env.caculate_energy_total()

            energy_traction_li.append(tra_energy1)
            energy_re_li.append(re_energy1)
            energy_total_li.append(tra_energy1-re_energy1)

            steps += 1
            if done:
                print(f"iter: {iter} | ep_len:{steps} | ep_rew:{sum(rew_li)}")
                print(f"重叠时间: {metro_env.overlap_time[-1]}|牵引时间：{metro_env.total_traction_time_episodes[-1]} ")
                print(f"总延误时间偏差: {metro_env.total_deviation_time[-1]} ")
                break
        iter += 1
        rew_mean1 += sum(rew_li) / iterations
        energy_traction_mean1 += sum(energy_traction_li) / iterations
        energy_re_mean1 += sum(energy_re_li) / iterations
        energy_total_mean1 += sum(energy_total_li) / iterations

    print(f"time：{time.time() - time1}")
    print(f"rew_mean1：{rew_mean1}")
    print(f"牵引能耗：{energy_traction_mean1}")
    print(f"再生能耗：{energy_re_mean1}")
    print(f"总能耗：{energy_total_mean1}")
    metro_env.render()
    print('test over')
    '''


    agent = PPO.load('./models/2024-0524-10-05-57-PPO-4-7/rl_model_56000_steps.zip')
    iter = 0
    time2 = time.time()
    rew_mean2 = 0
    energy_traction_mean2 = 0
    energy_re_mean2 = 0
    energy_total_mean2 = 0

    while iter < iterations:
        rew_li = []
        energy_traction_li = []
        energy_re_li = []
        energy_total_li = []

        steps = 0
        next_obs = metro_env.reset()
        for _ in range(1000):
            action = agent.predict(next_obs)[0]
            print(action)
            next_obs, reward, done, _ = metro_env.step(action)

            rew_li.append(reward)
            tra_energy2,re_energy2= metro_env.caculate_energy_total()

            energy_traction_li.append(tra_energy2)
            energy_re_li.append(re_energy2)
            energy_total_li.append(tra_energy2 - re_energy2)

            steps += 1
            if done:
                print(f"iter: {iter} | ep_len:{steps} | ep_rew:{sum(rew_li)}")
                print(f"重叠时间: {metro_env.overlap_time[-1]}|牵引时间：{metro_env.total_traction_time_episodes[-1]} ")
                print(f"总偏差时间: {metro_env.total_deviation_time[-1]} ")
                break
        iter += 1
        rew_mean2 += sum(rew_li) / iterations
        energy_traction_mean2 += sum(energy_traction_li) / iterations
        energy_re_mean2 += sum(energy_re_li) / iterations
        energy_total_mean2 += sum(energy_total_li) / iterations
    print(f"time：{time.time() - time2}")
    print(f"rew_mean2：{rew_mean2}")
    print(f"牵引能耗：{energy_traction_mean2}")
    print(f"再生能耗：{energy_re_mean2}")
    print(f"总能耗：{energy_total_mean2}")
    metro_env.render()
    print('finish')


if __name__ == "__main__":
    test()
