import numpy as np
from envs.multiagentenv import MultiAgentEnv  # 确保 pymarl 的接口兼容
from grid_maif.grid_maif import POMAPFEnv  # 引入 grid_maif 环境
from gymnasium import spaces
import torch


class GridMaifWrapper(MultiAgentEnv):
    """
    Wrapper 将 grid_maif 环境封装为适配 pymarl 的接口。
    """

    def __init__(self, **kwargs):
        """
        初始化 GridMaifWrapper 环境。
        所有环境配置均通过 kwargs 传入。
        """
        self.env_args = kwargs  # 保存环境参数
        self.episode_limit = kwargs.get("episode_limit", 100)  # 设置最大时间步
        self.obs_radius = kwargs.get("obs_radius", 1)  # 观测范围
        self.num_agents = kwargs.get("num_agents", 4)  # 智能体数量
        self.render_bool = kwargs.get("render", False) 

        # 初始化 grid_maif 环境
        self.env = POMAPFEnv(
            action_mapping=kwargs.get("action_mapping", None),
            default_env_setting=kwargs.get("default_env_setting", (self.num_agents, 8)),
            obs_radius=self.obs_radius,
            reward_fn=kwargs.get("reward_fn", None),
            use_predefined=kwargs.get("use_predefined", False),
            benchmark_path=kwargs.get("benchmark_path", None),
        )
        self.num_agents = self.env.num_agents  # 更新智能体数量
        self.obs_radius = self.env.obs_radius  # 更新观测范围

        # 获取动作空间
        self.action_space = spaces.Discrete(len(self.env.action_mapping))  # 动作数量
        self.action_spaces = [self.action_space for _ in range(self.num_agents)]  # 每个智能体的动作空间

        # 获取观测空间

        self.observation_space = self.env.single_observations_space  # 单个智能体观测空间
        self.observation_spaces=self.env.observation_spaces

        # 初始化时间步
        self.current_step = 0

        # 保存最近一次的观测和状态
        self._obs = None
        self._state = None

    def step(self, actions):
        """
        执行一步动作。
        :param actions: 动作列表 [a1, a2, ..., an]，n 为智能体数量。
        :return: obs, reward, done, truncated, info
        """
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        elif isinstance(actions,list):
            actions=np.array(actions)
        elif not isinstance(actions, np.ndarray):
            raise TypeError(f"Unsupported action type: {type(actions)}")
        
        # 调用原始环境的 step 函数
        obs, rewards, done, terminated, info = self.env.step(actions)

        # 更新时间步
        self.current_step += 1

        # 检查是否达到最大时间步
        time_limited_bool = self.current_step >= self.episode_limit
        done = done or time_limited_bool  # 如果达到时间限制，标记为结束

        # 保存观测
        self._obs = obs

        # 处理奖励为列表形式或标量形式
        total_reward = sum(rewards)  # 总奖励
        reward = total_reward if self.env_args.get("common_reward", True) else np.array(rewards)
        # print(actions)
        # 构建返回信息
        info["episode_limit"] = time_limited_bool  # 添加时间限制信息
        if self.render_bool==True:
            print(actions)
            self.render()

        return obs, reward, done, terminated, info

    def reset(self):
        """
        重置环境。
        :return: 初始观测。
        """
        self.current_step = 0  # 重置时间步
        obs = self.env.reset()
        self._obs = obs
        self._state = self.get_state()
        return self._obs

    def get_obs(self):
        """
        获取所有智能体的观测。
        :return: 观测列表 [obs1, obs2, ..., obsn]。
        """
        return self._obs

    def get_obs_agent(self, agent_id):
        """
        获取单个智能体的观测。
        :param agent_id: 智能体 ID。
        :return: 单个智能体的观测。
        """
        return self._obs[agent_id]  # obs[0] 包含观测

    def get_obs_size(self):
        """
        获取单个智能体的观测空间大小（总元素数量）。
        :return: 观测空间大小（标量）。
        """
        return int(np.prod(self.observation_space))

    def get_state(self):
        """
        获取全局状态。
        :return: 全局状态。
        """
        # 将所有智能体的观测拼接为全局状态
        return np.concatenate(self._obs, axis=0)
        # return np.concatenate([obs.flatten() for obs in self._obs])

    def get_state_size(self):
        """
        获取全局状态空间大小。
        :return: 状态空间大小。
        """
        return int(self.get_obs_size() * self.num_agents)

    def get_avail_actions(self):
        """
        获取所有智能体的可用动作。
        :return: 可用动作列表 [[a1_1, a1_2, ...], [a2_1, a2_2, ...], ...]。
        """
        return [[1] * self.action_space.n for _ in range(self.num_agents)]

    def get_avail_agent_actions(self, agent_id):
        """
        获取单个智能体的可用动作。
        :param agent_id: 智能体 ID。
        :return: 可用动作列表 [a1, a2, ...]。
        """
        return [1] * self.action_space.n

    def get_total_actions(self):
        """
        获取动作总数。
        :return: 动作总数。
        """
        return self.action_space.n

    def render(self, **kwargs):
        """
        渲染环境。
        """
        self.env.render(**kwargs)

    def close(self):
        """
        关闭环境。
        """
        pass

    def seed(self, seed=None):
        """
        设置环境随机种子。
        :param seed: 随机种子。
        """
        np.random.seed(seed)

    def save_replay(self):
        """
        此方法保留为 pymarl 兼容性，当前不实现。
        """
        pass

    def get_stats(self):
        """
        返回环境统计信息。
        :return: 统计信息字典。
        """
        return {}

    def get_env_info(self):
        """
        返回环境的关键信息。
        :return: 环境信息字典。
        """
        return {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.num_agents,
            "episode_limit": self.episode_limit,
            "action_spaces": self.action_spaces,
        }