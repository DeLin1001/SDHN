from envs.multiagentenv import MultiAgentEnv
import numpy as np
import torch  # 添加导入 torch
from .gym_art.quadrotor_multi.quadrotor_multi import QuadrotorEnvMulti
from collections.abc import Iterable
from gymnasium import spaces
import warnings
class QuadrotorMultiWrapper(MultiAgentEnv):
    def __init__(self, **kwargs):
        # 提取 env_args，如果未提供则为空字典
        # env_args = kwargs.pop('env_args', {})
        env_args=kwargs
        # 从 env_args 或 kwargs 中获取 common_reward
        self.common_reward = env_args.pop('common_reward', kwargs.pop('common_reward', True))
        # 从 env_args 或 kwargs 中获取 reward_scalarisation
        self.reward_scalarisation = env_args.pop('reward_scalarisation', kwargs.pop('reward_scalarisation', 'sum'))
        print(env_args)
        self.render_bool=env_args['render']
        self.render_mode=env_args['render_mode']
        self.episode_limit = env_args['ep_time']
        self.room_dims=env_args['room_dims']
        self.normalise_actions=env_args['normalise_actions']
        # 获取控制模式和动作空间类型
        self.control_input = env_args.get('control_input', 'pos')
        self.if_discrete = env_args.get('if_discrete', False)
        # 过滤 env_args，移除 QuadrotorEnvMulti 未定义的参数
        allowed_params = [
            'num_agents','rew_coeff', 'obs_repr', 'neighbor_visible_num',
            'neighbor_obs_type', 'collision_hitbox_radius', 'collision_falloff_radius',
            'use_obstacles', 'obst_density', 'obst_size', 'obst_spawn_area',
            'use_downwash', 'use_numba', 'quads_mode', 'room_dims', 'use_replay_buffer',
            'quads_view_mode', 'render', 'dynamics_params', 'raw_control','control_input','if_discrete',
            'zero_action_middle', 'dynamics_randomize_every', 'dynamics_change',
            'dyn_sampler_1', 'sense_noise', 'init_random_state', 'render_mode','sim_freq'
        ]
        env_args = {key: env_args[key] for key in env_args if key in allowed_params}
        
        # 将剩余的参数传递给 QuadrotorEnvMulti
        self.env = QuadrotorEnvMulti(**env_args)
        self.n_agents = self.env.num_agents
        
        # 修改部分：将 action_space 和 action_spaces 的范围重设为 [-1, 1]
        self.original_action_spaces = self.env.action_spaces  # 保存原始动作空间
        if self.if_discrete:
            self.action_spaces=self.original_action_spaces
        else:
            if self.normalise_actions:
                self.action_spaces = [
                    self._create_normalized_action_space(space)
                    for space in self.original_action_spaces
                ]
            else:
                self.action_spaces=self.original_action_spaces
        self.action_space = self.action_spaces[0]  # 单一动作空间
        self.observation_space=self.env.observation_space
        self.observation_spaces=self.env.observation_spaces
        

        self.current_step = 0  # 初始化当前时间步计数器
        #复数形式是list，表示所有智能体空间，单数形式是单个智能体的（第一个agent的空间）

        self._obs = None
        self._state = None
        if self.common_reward:
            if self.reward_scalarisation == "sum":
                self.reward_agg_fn = lambda rewards: sum(rewards)
            elif self.reward_scalarisation == "mean":
                self.reward_agg_fn = lambda rewards: sum(rewards) / len(rewards)
            else:
                raise ValueError(
                    f"Invalid reward_scalarisation: {self.reward_scalarisation} (only support 'sum' or 'mean')"
                )
    
    # 修改部分：创建 [-1, 1] 范围的动作空间
    def _create_normalized_action_space(self, original_space):
        if hasattr(original_space, 'shape'):  # 连续动作空间
            low = np.full(original_space.shape, -1.0)
            high = np.full(original_space.shape, 1.0)
            return spaces.Box(low=low, high=high, dtype=np.float32)
        elif hasattr(original_space, 'n'):  # 离散动作空间，暂不处理
            raise NotImplementedError("当前代码仅支持连续动作空间映射")
        else:
            raise ValueError(f"无法识别的动作空间类型: {type(original_space)}")

    def step(self, actions):
        # print(actions)
        reward, terminated, env_infos=[],None,{}
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        elif isinstance(actions,list):
            actions=np.array(actions)
        elif not isinstance(actions, np.ndarray):
            raise TypeError(f"Unsupported action type: {type(actions)}")
        
        # 修改部分：将 [-1, 1] 范围的动作映射回原动作空间范围
        original_actions = []
        if self.normalise_actions and not self.if_discrete:
            for agent_id, action in enumerate(actions):
                original_space = self.original_action_spaces[agent_id]
                action_low = original_space.low
                action_high = original_space.high
                original_action = (action + 1) / 2 * (action_high - action_low) + action_low
                original_actions.append(original_action)
        else:
            original_actions=actions
        # 执行环境步
        obs, rewards, dones, infos = self.env.step(original_actions)  # obs: [n_agents x obs_dim], rewards: [n_agents], dones: [n_agents], infos: [n_agents x dict]
        
        # 更新时间步计数器
        self.current_step += 1

        # 检查是否达到 episode_limit

        time_limited_bool = False  # 您原有的结束条件
        if self.current_step >= self.episode_limit:
            time_limited_bool = True

        # 返回值中的 dones 列表需要包含每个智能体的 done 标志，或者单个 done 标志
        # 例如，如果 dones 是每个智能体的列表：
        # dones = [done] * self.num_agents
        # 如果 dones 是单个布尔值：
        # dones = done
        # 将智能体的奖励列表转换为 numpy 数组
        reward_n = np.array(rewards)

        # 如果是公共奖励，根据设置计算总奖励
        if self.common_reward:
            reward = float(self.reward_agg_fn(rewards))
        else:
            reward = reward_n

        # 终止条件
        terminated = all(dones)
        if terminated or time_limited_bool: #时间达到限制或者仿真本身停止，都会导致terminated=true
            terminated=True
        
        # 构建扁平的 env_infos

        # 遍历每个智能体的 info，累加统计信息
        for info in infos:
            for k, v in info.items():
                if isinstance(v, dict):
                    # 如果值是字典，继续展开
                    for sub_k, sub_v in v.items():
                        key = f"{k}_{sub_k}"  # 使用下划线连接键名
                        env_infos[key] = env_infos.get(key, 0) + sub_v
                else:
                    env_infos[k] = env_infos.get(k, 0) + v
        env_infos['episode_limit']=time_limited_bool
        # info_n = {}  # 可以根据需要填充

        # 更新观测和状态
        self._obs = obs
        self._state = self.get_state()
        
        if self.render_bool==True:
            self.render()
     
        return obs,reward, terminated,terminated, env_infos

    def reset(self):
        self.current_step = 0  # 重置当前时间步
        obs = self.env.reset()
        self._obs = obs
        self._state = self.get_state()
        # return self._obs  # 移除返回值

    def get_obs(self):
        # print(self._obs[0][0:3])#just for test
        return self._obs

    def get_obs_agent(self, agent_id):
        return self._obs[agent_id]

    def get_obs_size(self):
        return self.env.observation_space.shape[0]

    def get_state(self):
        # 拼接所有智能体的状态，可以根据需要修改
        return np.concatenate(self._obs, axis=0)

    def get_state_size(self):
        return self.get_obs_size() * self.n_agents


    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions
    # 修改 get_avail_agent_actions 方法，支持离散动作空间
    def get_avail_agent_actions(self, agent_id):
        action_space = self.action_spaces[agent_id]
        if isinstance(action_space, spaces.Discrete):
            return [1] * action_space.n  # 所有动作都可用
        elif isinstance(action_space, spaces.Box):
            action_dim = action_space.shape[0]
            return [1] * action_dim
        else:
            raise NotImplementedError("未知的动作空间类型")




    # def get_total_actions(self):
    #     action_space = self.action_spaces[0]  # 假设所有智能体的动作空间相同
    #     if hasattr(action_space, 'shape'):
    #         return action_space.shape[0]
    #     else:
    #         raise NotImplementedError("未知的动作空间类型")
        
    def get_total_actions(self):
        # 假设所有智能体的动作空间相同
        action_space = self.action_spaces[0]  # 修改为取第一个智能体的动作空间，如果动作空间相同
        if hasattr(action_space, 'n'):
            return action_space.n
        elif hasattr(action_space, 'shape'):
            return action_space.shape[0]
        else:
            raise NotImplementedError("未知的动作空间类型")

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)

    def save_replay(self):
        pass

    def get_stats(self):
        return {}

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
            "action_spaces": self.action_spaces,  # 添加这一行
            "actions_dtype":np.float32,
            "normalise_actions": self.normalise_actions
        }
        return env_info