import matplotlib.pyplot as plt  # 用于绘制地图
import imageio  # 用于生成 GIF 动图
import random
from typing import List
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from PIL import Image
import pickle  # 用于加载 .pth 文件
import math
from scipy.ndimage import label

# config
import yaml



class POMAPFEnv:
    def __init__(self, 
                 action_mapping=None,
                 default_env_setting=None,
                 obs_radius=1,
                 reward_fn=None,
                 use_predefined=False,
                 benchmark_path=None,
                 **kwargs):
        """
        使用 kwargs 的方式获取配置，而不再从 config.yaml 读取。
        
        参数示例：
            action_mapping: dict, e.g. {0: [0,0], 1: [-1,0], 2: [1,0], ...}
            default_env_setting: tuple or list, e.g. ( num_agents, map_size )
            obs_radius: int, e.g. 1
            reward_fn: dict, 包含各种动作与状态对应的奖励
            use_predefined: bool, 是否使用预定义数据
            benchmark_path: str, 若使用预定义，则可以存放 pth 文件路径
            ...
        """
        # 1) 记录下传入的关键配置
        # 如果外部未显式传入，就使用一些默认值
        self.FREE_SPACE=kwargs.get("FREE_SPACE", 0)
        self.OBSTACLE=kwargs.get("OBSTACLE", 1)
        self.action_mapping = action_mapping if action_mapping is not None else {
            0: np.array([0,0]),
            1: np.array([-1,0]),
            2: np.array([1,0]),
            3: np.array([0,-1]),
            4: np.array([0,1]),
        }
        self.default_env_setting = default_env_setting if default_env_setting is not None else (4, 8)
        self.obs_radius = obs_radius
        self.reward_fn = reward_fn if reward_fn is not None else {
            "move": -0.1,
            "collision": -1,
            "stay_on_goal": 0.0,
            "stay_off_goal": -0.05,
            "reach_goal": 10.0,
            "formation_scale": 1.0,
            "approach": 0.1,
        }

        self.use_predefined = use_predefined
        self.benchmark_path = benchmark_path

        # 3. 以下这些属性均需要在环境中使用，为了避免未定义错误，先给它们设置占位值
        #    在后续的逻辑中对其进行真正赋值
        self.instances = None       # 用于存储已加载的预定义场景
        self.map = None             # 当前环境地图
        self.map_size = (0, 0)      # 当前地图尺寸
        self.num_agents = 0         # 当前环境中智能体数量
        self.agents_pos = None      # 智能体初始位置
        self.goals_pos = None       # 智能体目标位置
        self.obstacle_density = None
        self.current_rewards = None
        self.last_actions = None
        self.heuri_map = None
        self.render_frames = []
        self.fig = None
        self.ax = None
        self.img = None
        self.agent_colors = []
        self.goal_colors = []
        # 设置一个默认的渲染尺寸，以免后续使用时出错
        self.render_size = 1080  
        self.num_steps = 0

        # === Modification Here ===
        # 4. 若使用预定义场景，则直接在__init__中载入instances，并调用 self.reset_load_random()
        if self.use_predefined:
            if not self.benchmark_path:
                raise ValueError("use_predefined = True，但未在配置文件中提供有效的 benchmark_path。")
            with open(self.benchmark_path, 'rb') as f:
                self.instances = pickle.load(f)
            self.reset_load_random()   # 直接随机选一个场景加载
        else:
            # 否则，仍使用“随机生成地图”方式进行初始化
            self._init_random_env()

        self.reset()

    # === Modification Here ===
    def _init_random_env(self):
        """
        当不使用预定义场景时，执行本函数，初始化一个随机生成的环境。
        将其从原来 __init__ 中抽出来，避免重复。
        """
        self.num_agents = self.default_env_setting[0]
        size = self.default_env_setting[1]
        self.map_size = (size, size)

        # 随机生成地图
        self.obstacle_density = np.random.triangular(0, 0.33, 0.5)
        self.map = np.random.choice(2, self.map_size,
                                    p=[1 - self.obstacle_density, self.obstacle_density]).astype(int)

        # 分区并确保有足够位置
        partition_list = self.map_partition(self.map)
        partition_list = [p for p in partition_list if len(p) >= 2]
        while len(partition_list) == 0:
            self.map = np.random.choice(2, self.map_size,
                                        p=[1 - self.obstacle_density, self.obstacle_density]).astype(int)
            partition_list = self.map_partition(self.map)
            partition_list = [p for p in partition_list if len(p) >= 2]

        # 为 num_agents 个体随机分配起始位置和目标位置
        self.agents_pos = np.empty((self.num_agents, 2), dtype=int)
        self.goals_pos = np.empty((self.num_agents, 2), dtype=int)
        pos_num = sum([len(p) for p in partition_list])

        for i in range(self.num_agents):
            pos_idx = random.randint(0, pos_num - 1)
            partition_idx = 0
            for partition in partition_list:
                if pos_idx >= len(partition):
                    pos_idx -= len(partition)
                    partition_idx += 1
                else:
                    break

            pos = random.choice(partition_list[partition_idx])
            partition_list[partition_idx].remove(pos)
            self.agents_pos[i] = np.asarray(pos, dtype=int)

            pos = random.choice(partition_list[partition_idx])
            partition_list[partition_idx].remove(pos)
            self.goals_pos[i] = np.asarray(pos, dtype=int)

            partition_list = [p for p in partition_list if len(p) >= 2]
            pos_num = sum([len(p) for p in partition_list])

        self.num_steps = 0
        # self.get_heuristic_map()
        self.last_actions = np.zeros(
            (self.num_agents, 5, 2*self.obs_radius+1, 2*self.obs_radius+1),
            dtype=bool
        )
        self.current_rewards = None
        self.agent_colors = self.generate_agent_colors(self.num_agents)
        self.goal_colors = self.generate_goal_colors(self.num_agents)

    def move(self, loc, d):
        return loc[0] + self.action_mapping[d][0], loc[1] + self.action_mapping[d][1]
    
    # def map_partition(self,grid_map):
    #     empty_spots = np.argwhere(np.array(grid_map)==self.FREE_SPACE).tolist()
    #     empty_spots = [tuple(pos) for pos in empty_spots]
    #     partitions = []
    #     while empty_spots:
    #         start_loc = empty_spots.pop()
    #         open_list = [start_loc]
    #         close_list = []
    #         while open_list:
    #             loc = open_list.pop(0)
    #             for d in range(4):
    #                 child_loc = self.move(loc, d)
    #                 if child_loc[0] < 0 or child_loc[0] >= len(grid_map) \
    #                     or child_loc[1] < 0 or child_loc[1] >= len(grid_map[0]):
    #                     continue
    #                 if grid_map[child_loc[0]][child_loc[1]] == self.OBSTACLE:
    #                     continue
    #                 if child_loc in empty_spots:
    #                     empty_spots.remove(child_loc)
    #                     open_list.append(child_loc)
    #             close_list.append(loc)
    #         partitions.append(close_list)
    #     return partitions



    def map_partition(grid_map):
        """
        使用 scipy.ndimage.label 来获取地图中连通的空地分区。
        """
        # grid_map == self.FREE_SPACE(0) 表示空地
        # 先得到只包含 True/False 的二值数组：True代表空地，False代表障碍
        free_area = (grid_map == 0)

        # 调用 label 获取连通分区
        labeled_array, num_features = label(free_area)

        # 根据 label 的结果，把同一个 label 归类到同一个 partition
        partitions = [[] for _ in range(num_features)]
        # (x, y) 各自的 label
        xs, ys = np.where(labeled_array > 0)
        for x, y in zip(xs, ys):
            label_id = labeled_array[x, y]  # 注意 label_id 从 1 开始
            partitions[label_id - 1].append((x, y))

        return partitions

    def generate_agent_colors(self, num_agents):
        """
        根据智能体数量，生成不同的颜色。
        使用HSV色彩空间均匀分布颜色，然后转换为RGB。
        """
        colors = []
        for i in range(num_agents):
            hue = i / num_agents  # hue在[0,1)之间均匀分布
            saturation = 0.8  # 饱和度固定
            value = 0.9  # 亮度固定
            color = plt.cm.hsv(hue)
            colors.append(color[:3])  # 取RGB部分
        return colors

    def generate_goal_colors(self, num_agents):
        """
        生成目标位置的颜色，采用比智能体颜色浅的颜色。
        """
        goal_colors = []
        for color in self.agent_colors:
            # 增加亮度
            light_color = np.clip(np.array(color) + 0.3, 0, 1)
            goal_colors.append(light_color)
        return goal_colors

    def update_env_setting_set(self, new_env_setting_set):
        self.env_setting_set = new_env_setting_set

    def reset(self):
        if self.use_predefined:
            self.reset_load_random()
        else:
            self.reset_random()
        
        return self.observe()

    def reset_random(self):

        self.render_frames = []  # 用于保存渲染帧的列表
        self.fig = None  # 用于保存 matplotlib 图像窗口
        self.ax = None  # 保存绘图区域
        self.img = None  # 保存图像对象
        self.render_frames = []  # 保存渲染帧列表
        rand_env_setting = random.choice(self.env_setting_set)
        self.num_agents = rand_env_setting[0]
        self.map_size = (rand_env_setting[1], rand_env_setting[1])
        self.obstacle_density = np.random.triangular(0, 0.33, 0.5)
        
        self.map = np.random.choice(2, self.map_size, p=[1-self.obstacle_density, self.obstacle_density]).astype(np.float32)
        partition_list = self.map_partition(self.map)
        partition_list = [ partition for partition in partition_list if len(partition) >= 2 ]
        while len(partition_list) == 0:
            self.map = np.random.choice(2, self.map_size, p=[1-self.obstacle_density, self.obstacle_density]).astype(np.float32)
            partition_list = self.map_partition(self.map)
            partition_list = [ partition for partition in partition_list if len(partition) >= 2 ]
        self.agents_pos = np.empty((self.num_agents, 2), dtype=int)
        self.goals_pos = np.empty((self.num_agents, 2), dtype=int)
        pos_num = sum([ len(partition) for partition in partition_list ])
        
        for i in range(self.num_agents):
            pos_idx = random.randint(0, pos_num-1)
            partition_idx = 0
            for partition in partition_list:
                if pos_idx >= len(partition):
                    pos_idx -= len(partition)
                    partition_idx += 1
                else:
                    break 
            pos = random.choice(partition_list[partition_idx])
            partition_list[partition_idx].remove(pos)
            self.agents_pos[i] = np.asarray(pos, dtype=int)
            pos = random.choice(partition_list[partition_idx])
            partition_list[partition_idx].remove(pos)
            self.goals_pos[i] = np.asarray(pos, dtype=int)
            partition_list = [ partition for partition in partition_list if len(partition) >= 2 ]
            pos_num = sum([ len(partition) for partition in partition_list ])
        self.num_steps = 0
        # self.get_heuristic_map()
        self.last_actions = np.zeros((self.num_agents, 5, 2*self.obs_radius+1, 2*self.obs_radius+1), dtype=bool)

        # 重置当前奖励
        self.current_rewards = None

        # 重新生成颜色，以防智能体数量变化
        self.agent_colors = self.generate_agent_colors(self.num_agents)
        self.goal_colors = self.generate_goal_colors(self.num_agents)

        return self.observe()


    def reset_load_random(self):
        """
        当 use_predefined=True 时，从 self.instances (已读入的数据) 中随机选取一个场景进行加载。
        与原先的示例不同，不再需要传入 instances 这个参数。
        """
        if not self.instances or 'maps' not in self.instances or 'forms' not in self.instances:
            raise ValueError("instances 数据不完整或加载失败，请检查 benchmark_path 对应的文件格式。")

        self.last_actions = np.zeros((self.num_agents, 5, 2*self.obs_radius+1, 2*self.obs_radius+1), dtype=bool)

        self.num_steps = 0
        # 随机选择一个 i
        i = random.randint(0, len(self.instances['maps']) - 1)

        selected_map = self.instances['maps'][i]         # np.ndarray
        init_positions = self.instances['forms'][i][0]   # 列表[(x, y), ...]
        goal_positions = self.instances['forms'][i][1]   # 列表[(x, y), ...]

        # 直接调用 environment 内部已有的 load 函数
        self.load(
            map=np.array(selected_map),
            agents_pos=np.array(init_positions),
            goals_pos=np.array(goal_positions)
        )
        return self.observe()

    def load(self, map:np.ndarray, agents_pos:np.ndarray, goals_pos:np.ndarray):
        self.map = np.copy(map)
        self.agents_pos = np.copy(agents_pos)
        self.goals_pos = np.copy(goals_pos)
        self.num_agents = agents_pos.shape[0]
        self.map_size = (self.map.shape[0], self.map.shape[1])
        self.num_steps = 0
        # self.get_heuristic_map()
        self.last_actions = np.zeros((self.num_agents, 5, 2*self.obs_radius+1, 2*self.obs_radius+1), dtype=bool)

        # 重置当前奖励
        self.current_rewards = None

        # 重新生成颜色，以防智能体数量变化
        self.agent_colors = self.generate_agent_colors(self.num_agents)
        self.goal_colors = self.generate_goal_colors(self.num_agents)

    def get_heuristic_map(self):
        dist_map = np.ones((self.num_agents, *self.map_size), dtype=int) * float('inf')
        for i in range(self.num_agents):
            open_list = list()
            x, y = tuple(self.goals_pos[i])
            open_list.append((x, y))
            dist_map[i, x, y] = 0

            while open_list:
                x, y = open_list.pop(0)
                dist = dist_map[i, x, y]
                up = x-1, y
                if up[0] >= 0 and self.map[up]==0 and dist_map[i, x-1, y] > dist+1:
                    dist_map[i, x-1, y] = dist+1
                    if up not in open_list:
                        open_list.append(up)
                down = x+1, y
                if down[0] < self.map_size[0] and self.map[down]==0 and dist_map[i, x+1, y] > dist+1:
                    dist_map[i, x+1, y] = dist+1
                    if down not in open_list:
                        open_list.append(down)
                left = x, y-1
                if left[1] >= 0 and self.map[left]==0 and dist_map[i, x, y-1] > dist+1:
                    dist_map[i, x, y-1] = dist+1
                    if left not in open_list:
                        open_list.append(left)
                right = x, y+1
                if right[1] < self.map_size[1] and self.map[right]==0 and dist_map[i, x, y+1] > dist+1:
                    dist_map[i, x, y+1] = dist+1
                    if right not in open_list:
                        open_list.append(right)
        self.heuri_map = np.zeros((self.num_agents, 4, *self.map_size), dtype=bool)

        for x in range(self.map_size[0]):
            for y in range(self.map_size[1]):
                if self.map[x, y] == 0:
                    for i in range(self.num_agents):
                        if x > 0 and dist_map[i, x-1, y] < dist_map[i, x, y]:
                            assert dist_map[i, x-1, y] == dist_map[i, x, y]-1
                            self.heuri_map[i, 0, x, y] = 1
                        if x < self.map_size[0]-1 and dist_map[i, x+1, y] < dist_map[i, x, y]:
                            assert dist_map[i, x+1, y] == dist_map[i, x, y]-1
                            self.heuri_map[i, 1, x, y] = 1
                        if y > 0 and dist_map[i, x, y-1] < dist_map[i, x, y]:
                            assert dist_map[i, x, y-1] == dist_map[i, x, y]-1
                            self.heuri_map[i, 2, x, y] = 1
                        if y < self.map_size[1]-1 and dist_map[i, x, y+1] < dist_map[i, x, y]:
                            assert dist_map[i, x, y+1] == dist_map[i, x, y]-1
                            self.heuri_map[i, 3, x, y] = 1
        self.heuri_map = np.pad(self.heuri_map, ((0, 0), (0, 0), (self.obs_radius, self.obs_radius), (self.obs_radius, self.obs_radius)))

    def step(self, actions: List[int]):
        terminated=False
        assert len(actions) == self.num_agents, 'only {} actions as input while {} agents in environment'.format(len(actions), self.num_agents)
        assert all([action_idx<5 and action_idx>=0 for action_idx in actions]), 'action index out of range'

        checking_list = [i for i in range(self.num_agents)]

        rewards = []
        next_pos = np.copy(self.agents_pos)
        # ---------  首先记录各agent与目标的旧距离(欧几里得，也可改曼哈顿) ----------
        old_distances = np.linalg.norm(self.agents_pos - self.goals_pos, axis=1)
        # remove unmoving agent id
        for agent_id in checking_list.copy():
            if actions[agent_id] == 0:
                # unmoving
                if np.array_equal(self.agents_pos[agent_id], self.goals_pos[agent_id]):
                    rewards.append(self.reward_fn['stay_on_goal'])
                else:
                    rewards.append(self.reward_fn['stay_off_goal'])
                checking_list.remove(agent_id)
            else:
                # move
                
                next_pos[agent_id] += self.action_mapping[actions[agent_id]]
                rewards.append(self.reward_fn['move'])

        # first round check, these two conflicts have the heightest priority
        for agent_id in checking_list.copy():
            if np.any(next_pos[agent_id]<0) or np.any(next_pos[agent_id]>=self.map_size[0]):
                # agent out of map range
                rewards[agent_id] = self.reward_fn['collision']
                next_pos[agent_id] = self.agents_pos[agent_id]
                checking_list.remove(agent_id)
            elif self.map[tuple(next_pos[agent_id])] == 1:
                # collide obstacle
                rewards[agent_id] = self.reward_fn['collision']
                next_pos[agent_id] = self.agents_pos[agent_id]
                checking_list.remove(agent_id)

        # second round check, agent swapping conflict
        no_conflict = False
        while not no_conflict:
            no_conflict = True
            for agent_id in checking_list:
                target_agent_id = np.where(np.all(next_pos[agent_id]==self.agents_pos, axis=1))[0]
                if target_agent_id:
                    target_agent_id = target_agent_id.item()
                    assert target_agent_id != agent_id, 'logic bug'
                    if np.array_equal(next_pos[target_agent_id], self.agents_pos[agent_id]):
                        assert target_agent_id in checking_list, 'target_agent_id should be in checking list'
                        next_pos[agent_id] = self.agents_pos[agent_id]
                        rewards[agent_id] = self.reward_fn['collision']
                        next_pos[target_agent_id] = self.agents_pos[target_agent_id]
                        rewards[target_agent_id] = self.reward_fn['collision']
                        checking_list.remove(agent_id)
                        checking_list.remove(target_agent_id)
                        no_conflict = False
                        break

        # third round check, agent collision conflict
        no_conflict = False
        while not no_conflict:
            no_conflict = True
            for agent_id in checking_list:
                collide_agent_id = np.where(np.all(next_pos==next_pos[agent_id], axis=1))[0].tolist()
                if len(collide_agent_id) > 1:
                    # collide agent
                    # if all agents in collide agent are in checking list
                    all_in_checking = True
                    for id in collide_agent_id.copy():
                        if id not in checking_list:
                            all_in_checking = False
                            collide_agent_id.remove(id)
                    if all_in_checking:
                        collide_agent_pos = next_pos[collide_agent_id].tolist()
                        for pos, id in zip(collide_agent_pos, collide_agent_id):
                            pos.append(id)
                        collide_agent_pos.sort(key=lambda x: x[0]*self.map_size[0]+x[1])
                        collide_agent_id.remove(collide_agent_pos[0][2])
                    next_pos[collide_agent_id] = self.agents_pos[collide_agent_id]
                    for id in collide_agent_id:
                        rewards[id] = self.reward_fn['collision']
                    for id in collide_agent_id:
                        checking_list.remove(id)
                    no_conflict = False
                    break

        self.agents_pos = np.copy(next_pos)
        self.num_steps += 1

        # ---------- 计算“接近目标”额外奖励 -----------
        # 现在计算新距离
        new_distances = np.linalg.norm(self.agents_pos - self.goals_pos, axis=1)
        approach_rew = self.reward_fn.get("approach", 0.1)  # 默认给个 0.1
        for i in range(self.num_agents):
            distance_diff = old_distances[i] - new_distances[i]
            # 如果真正更靠近，则给正奖励
            if distance_diff > 0:
                rewards[i] += approach_rew

        # 计算编队队形奖励
        formation_reward = self.compute_formation_reward()
        for i in range(self.num_agents):
            rewards[i] += formation_reward[i]

        # check done
        if np.array_equal(self.agents_pos, self.goals_pos):
            done = True
            rewards = [self.reward_fn['reach_goal']+rewards[i] for i in range(self.num_agents)]
        else:
            done = False
        # info = {'step': self.num_steps-1}
        info = {}

        # make sure no overlapping agents
        if np.unique(self.agents_pos, axis=0).shape[0] < self.num_agents:
            print(self.num_steps)
            print(self.map)
            print(self.agents_pos)
            self.reset()
            terminated=True
            reward=-50
            raise RuntimeError('unique')
            return self.observe(), rewards, done, terminated, info
            

        # update last actions
        self.last_actions = np.zeros((self.num_agents, 5, 2*self.obs_radius+1, 2*self.obs_radius+1), dtype=bool)
        self.last_actions[np.arange(self.num_agents), np.array(actions)] = 1
        # 保存当前步的奖励
        self.current_rewards = rewards
        return self.observe(), rewards, done,terminated, info


    # def observe(self):
    #     obs = np.zeros((self.num_agents, 6, 2*self.obs_radius+1, 2*self.obs_radius+1), dtype=bool)
    #     # 说明：6 个通道分别表示智能体位置、目标位置、障碍物分布、其他智能体位置、行动历史、启发式地图
    #     obstacle_map = np.pad(self.map, self.obs_radius, 'constant', constant_values=0)
    #     agent_map = np.zeros((self.map_size), dtype=bool)
    #     agent_map[self.agents_pos[:,0], self.agents_pos[:,1]] = 1
    #     agent_map = np.pad(agent_map, self.obs_radius, 'constant', constant_values=0)
    #     for i, agent_pos in enumerate(self.agents_pos):
    #         x, y = agent_pos
    #         obs[i, 0] = agent_map[x:x+2*self.obs_radius+1, y:y+2*self.obs_radius+1]
    #         obs[i, 0, self.obs_radius, self.obs_radius] = 0
    #         obs[i, 1] = obstacle_map[x:x+2*self.obs_radius+1, y:y+2*self.obs_radius+1]
    #         obs[i, 2:] = self.heuri_map[i, :, x:x+2*self.obs_radius+1, y:y+2*self.obs_radius+1]

    #     return obs                #, np.copy(self.agents_pos)   
    # 
    def observe(self):
        """
        获取每个智能体的观测，返回形状为 [num_agents, single_obs]。
        single_obs 包括：
        - 自身位置：2 个元素 [x, y]。
        - formation 当前平均位置：2 个元素 [formation_mean_x, formation_mean_y]。
        - 目标位置：2 个元素 [goal_x, goal_y]。
        - goals 平均位置：2 个元素 [goals_mean_x, goals_mean_y]。
        - 观测范围展开：(2 * obs_radius + 1) ^ 2 个元素。
        """
        num_agents = self.num_agents
        obs_radius = self.obs_radius
        obs_size = (2 * obs_radius + 1) ** 2  # 展开的观测范围大小
        single_obs_length = 2 + 2 + 2 + 2 + obs_size  # 自身 + formation_mean + goal + goals_mean + 展开观测范围
        self.single_observations_space = (1, single_obs_length)
        self.observation_spaces = [self.single_observations_space for _ in range(self.num_agents)]
        observations = np.zeros((num_agents, single_obs_length), dtype=int)

        # 计算 formation 当前平均位置和 goals 的平均位置
        formation_mean = np.mean(self.agents_pos, axis=0).astype(int)
        goals_mean = np.mean(self.goals_pos, axis=0).astype(int)

        # 获取扩展后的障碍物地图和智能体地图
        padded_map = np.pad(self.map, obs_radius, mode="constant", constant_values=1)  # 1 表示障碍物
        padded_agent_map = np.zeros_like(padded_map, dtype=int)
        for i, pos in enumerate(self.agents_pos):
            padded_agent_map[obs_radius + pos[0], obs_radius + pos[1]] = 2  # 其他智能体标记为 2

        for i, (agent_pos, goal_pos) in enumerate(zip(self.agents_pos, self.goals_pos)):
            # 当前智能体的位置
            x, y = agent_pos
            goal_x, goal_y = goal_pos

            # 将自身位置存入观测
            observations[i, 0:2] = x, y  # 自身位置

            # 将 formation 当前平均位置存入观测
            observations[i, 2:4] = formation_mean[0], formation_mean[1]  # formation 平均位置

            # 将目标位置存入观测
            observations[i, 4:6] = goal_x, goal_y  # 目标位置

            # 将 goals 平均位置存入观测
            observations[i, 6:8] = goals_mean[0], goals_mean[1]  # goals 平均位置

            # 切片获取观测范围内的障碍物和其他智能体信息
            x_padded, y_padded = x + obs_radius, y + obs_radius
            local_map = padded_map[x_padded - obs_radius:x_padded + obs_radius + 1,
                                y_padded - obs_radius:y_padded + obs_radius + 1]
            local_agents = padded_agent_map[x_padded - obs_radius:x_padded + obs_radius + 1,
                                            y_padded - obs_radius:y_padded + obs_radius + 1]

            # 移除自身位置的数据（设置为 0）
            local_agents[obs_radius, obs_radius] = 0

            # 合并障碍物和其他智能体信息
            local_obs = local_map + local_agents

            # 将观测范围展平并存入观测
            observations[i, 8:] = local_obs.flatten()

        return observations
    def render(self, save_path=None, play=True):
        """
        渲染当前环境的栅格地图，支持实时显示和保存为 GIF。
        :param save_path: 如果指定路径，则保存当前帧为 GIF 文件的一部分。
        :param play: 如果为 True，则实时显示当前帧。
        """
        # 创建一个高分辨率的 RGB 图像
        grid_map = np.ones((self.map_size[0], self.map_size[1], 3), dtype=np.float32)  # 默认白色

        # 设置障碍物为黑色
        grid_map[self.map == 1] = [0, 0, 0]

        # 绘制目标位置：浅色
        for i, goal in enumerate(self.goals_pos):
            grid_map[goal[0], goal[1]] = self.goal_colors[i]

        # 绘制智能体位置：深色
        for i, pos in enumerate(self.agents_pos):
            grid_map[pos[0], pos[1]] = self.agent_colors[i]

        # 将图像放大到1080x1080
        pil_image = Image.fromarray((grid_map * 255).astype(np.uint8))
        pil_image = pil_image.resize((self.render_size, self.render_size), Image.NEAREST)
        grid_map_resized = np.array(pil_image)

        # 保存当前帧到渲染帧列表并即时保存为GIF（如果指定save_path）
        if save_path is not None:
            self.render_frames.append(grid_map_resized)
            # 立即保存当前帧到GIF
            imageio.mimsave(save_path, self.render_frames, fps=5)
            self.render_frames = []  # 清空帧列表以避免重复保存

        # 实时显示当前帧
        if play:
            if self.fig is None or self.ax is None:
                self.fig, self.ax = plt.subplots(figsize=(8, 8))
                self.ax.set_xticks([])  # 隐藏 x 轴
                self.ax.set_yticks([])  # 隐藏 y 轴
                self.img = self.ax.imshow(grid_map_resized, origin='upper')
                plt.ion()  # 开启交互模式
            else:
                self.img.set_data(grid_map_resized)

            # 在界面上显示当前步各项奖励
            if self.current_rewards is not None:
                reward_text = " | ".join([f"A{i + 1}: {r}" for i, r in enumerate(self.current_rewards)])
                self.ax.set_title(f"Step: {self.num_steps} | Rewards: {reward_text}", fontsize=12)
            else:
                self.ax.set_title(f"Step: {self.num_steps}", fontsize=12)

            plt.draw()
            plt.pause(0.001)  # 减小暂停时间以加快显示速度

    def save_gif(self, save_path, fps=5):
        """
        保存渲染帧为GIF文件。
        :param save_path: GIF文件的保存路径。
        :param fps: 每秒帧数。
        """
        if len(self.render_frames) == 0:
            print("没有帧可保存。")
            return
        imageio.mimsave(save_path, self.render_frames, fps=fps)
        print(f"渲染已保存到 {save_path}")
        
    def compute_formation_reward(self):
        """
        计算编队队形的奖励。奖励基于智能体的相对位置与目标队形的匹配程度。
        使用智能体群的中心作为参考，不依赖于任意单个智能体。
        """
        formation_reward = [0.0 for _ in range(self.num_agents)]
        if self.num_agents <= 1:
            return formation_reward  # 单个智能体无需编队

        # 计算目标队形的中心
        goal_centroid = np.mean(self.goals_pos, axis=0)
        # 计算当前智能体的中心
        current_centroid = np.mean(self.agents_pos, axis=0)

        # 计算目标相对于中心的相对位置
        target_rel_positions = self.goals_pos - goal_centroid

        # 计算当前智能体相对于中心的相对位置
        current_rel_positions = self.agents_pos - current_centroid

        # 计算相对位置的偏差（曼哈顿距离）
        for i in range(self.num_agents):
            distance = abs(current_rel_positions[i][0] - target_rel_positions[i][0]) + \
                       abs(current_rel_positions[i][1] - target_rel_positions[i][1])
            # 奖励根据距离进行线性映射，距离越小，奖励越高
            formation_scale = self.reward_fn.get('formation_scale', 1.0)
            formation_reward[i] += (math.exp(-distance)-0.5)*formation_scale

        return formation_reward