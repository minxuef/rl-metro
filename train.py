import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import datetime as dt
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env  # 只能用于gym环境
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from config import get_args, get_env_args
from stable_baselines3 import PPO, SAC, DDPG,A2C
from environment.metro_env_eventbased import MetroEnvEventbased
from stable_baselines3.common.env_checker import check_env


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose >= 1:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True


def smooth(data, weight=0.7):
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


args = get_args()
params = get_env_args(args)
# model保存目录
time_str = dt.datetime.strftime(dt.datetime.now(), '%Y-%m%d-%H-%M-%S')
model_name = f"{time_str}-{params['algo_name']}-{params['num_metros']}-{params['num_metro_stations']}"
log_name = f"{time_str}-{params['algo_name']}-{params['num_metros']}-{params['num_metro_stations']}"
model_dir = os.path.join('./models', model_name)
logs_dir = os.path.join('./logs', log_name)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

env = MetroEnvEventbased(params)
check_env(env=env, warn=True, skip_render_check=False)  # 检查环境是否符合stable_baseline3的要求

env = Monitor(env, logs_dir)  # 监控环境

# 设置评估回调函数
# eval_callback = EvalCallback(env, best_model_save_path=model_dir,
# log_path=logs_dir, n_eval_episodes=params['num_metros'] * params['num_metro_stations'] ,

# deterministic=True, render=False)

checkpoint_callback = CheckpointCallback(
    save_freq=params['num_metros'] * params['num_metro_stations'] * 100,
    save_path=model_dir,
    name_prefix="rl_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
)  # 50个episode保存一次模型

callback = [checkpoint_callback, SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=logs_dir)]

# 设置tensorboard参数

model_ppo = PPO("MlpPolicy", env, n_steps=params['num_metros'] * params['num_metro_stations'], verbose=0,
                learning_rate=0.03, gamma=0.99, gae_lambda=0.95, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                batch_size=params['num_metros'] * params['num_metro_stations'], n_epochs=10, clip_range=0.2,
                clip_range_vf=None, normalize_advantage=True,
                policy_kwargs=None,
                tensorboard_log=os.path.join(logs_dir, "PPO_tensorboard"))

#model_a2c = A2C("MlpPolicy", env, n_steps=2048, verbose=1, tensorboard_log=os.path.join(logs_dir, "A2C_tensorboard"))
#model_ddpg = DDPG("MlpPolicy", env, learning_rate=0.001,verbose=0, tensorboard_log=os.path.join(logs_dir, "DDPG_tensorboard"))
# 加载模型用的代码
# model = PPO.load('./models/2024-0319-16-49-08-PPO-8-24/rl_model_1920000_steps.zip', env, verbose=1, tensorboard_log=logs_dir)
#model_sac = SAC("MlpPolicy",env, verbose=0, tensorboard_log=os.path.join(logs_dir, "SAC_tensorboard"),learning_rate=0.003)

time_steps = params['num_metros'] * params['num_metro_stations'] * params['train_eps']  # 每次训练循环中智能体与环境交互次数

model_ppo.learn(total_timesteps=time_steps, tb_log_name='PPO', reset_num_timesteps=False, callback=callback)
#model_a2c.learn(total_timesteps=time_steps, tb_log_name='A2C', reset_num_timesteps=False,callback=callback)
#model_ddpg.learn(total_timesteps=time_steps, tb_log_name='DDPG', reset_num_timesteps=False,callback=callback)
#model_sac.learn(total_timesteps=time_steps, tb_log_name='SAC', reset_num_timesteps=False,callback=callback)
print('finished ,saving and plotting results...')
# model_a2c.save(f'{model_dir}/A2C/{time_steps * iters}')
#model_ddpg.save(f'{model_dir}/DDPG/{time_steps * iters}')

plt.rcParams["font.sans-serif"] = ['simhei']  # 设置字体
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['font.size'] = 14
# render回合奖励
plot_results([logs_dir], num_timesteps=time_steps, x_axis='timesteps', task_name='PPO_rewards', figsize=(10, 5))
plt.savefig('./results/episode_rewards_timesteps.svg',format='svg')
plt.grid(True)
plt.show()
plot_results([logs_dir], num_timesteps=time_steps, x_axis='episodes', task_name='PPO_rewards', figsize=(10, 5))
plt.savefig('./results/回合奖励变化.svg',format='svg')
plt.grid(True)
plt.show()

# render回合奖励，观察动作
# plt.figure(figsize=(10, 10))
plt.plot(np.arange(len(env.episode_rewards)), smooth(env.episode_rewards))
plt.xlabel('回合')
plt.ylabel('奖励')
plt.title('奖励变化')
plt.grid(True)
plt.savefig('./results/episode_rewards_plot.svg',format='svg')
plt.show()


# Save rewards to a DataFrame
reward_df = pd.DataFrame(env.episode_rewards, columns=['Reward'])
# Save rewards to a CSV file
reward_df.to_csv(os.path.join(model_dir, 'episodes_reward_data.csv'), index=True)


# render重叠时间延误偏差变化
plt.plot(np.arange(len(env.overlap_time_episodes)), env.overlap_time_episodes)
plt.xlabel('回合')
plt.ylabel('重叠时间/s')
plt.title('总重叠时间变化')
plt.grid(True)
plt.savefig('./results/Overlap_time_plot.svg',format='svg')
plt.show()

plt.plot(np.arange(len(env.total_deviation_time_episodes)), env.total_deviation_time_episodes)
plt.xlabel('回合')
plt.ylabel('总到达偏差时间/s')
plt.title('总到达偏差时间变化')
plt.grid(True)
plt.savefig('./results/Total_deviation_time_plot.svg',format='svg')
plt.show()

plt.plot(np.arange(len(env.total_traction_time_episodes)), env.total_traction_time_episodes)
plt.xlabel('回合')
plt.ylabel('牵引时间/s')
plt.title('总牵引时间变化')
plt.grid(True)
plt.savefig('./results/Total_traction_time_plot.svg',format='svg')
plt.show()

plt.plot(np.arange(len(env.tr_energy_episodes)), env.tr_energy_episodes)
plt.xlabel('回合')
plt.ylabel('牵引能量/x10kWh')
plt.title('牵引能量变化')
plt.grid(True)
plt.savefig('./results/Traction_energy.svg',format='svg')
plt.show()

plt.plot(np.arange(len(env.re_energy_episodes)), env.re_energy_episodes)
plt.xlabel('回合')
plt.ylabel('回馈能量/x10kWh')
plt.title('回馈能量变化')
plt.grid(True)
plt.savefig('./results/Re_energy.svg',format='svg')
plt.show()

plt.plot(np.arange(len(env.energy_episodes)), env.energy_episodes)
plt.xlabel('回合')
plt.ylabel('净牵引能耗/x10kWh')
plt.title('净牵引能耗变化')
plt.grid(True)
plt.savefig('./results/Energy.svg',format='svg')
plt.show()