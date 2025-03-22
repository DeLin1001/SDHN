import sys
import os
# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
# 获取项目的根目录
project_root = os.path.dirname(current_dir)
project_root = os.path.dirname(project_root)
# 将项目的根目录添加到 sys.path
sys.path.append(project_root)
#将gym_art的上级目录加入了sys.path，后续就可以直接通过gym_art.来import了
import copy
import time
from collections import deque
from copy import deepcopy

import gymnasium as gym
import numpy as np

from gym_art.quadrotor_multi.aerodynamics.downwash import perform_downwash
from gym_art.quadrotor_multi.collisions.obstacles import perform_collision_with_obstacle
from gym_art.quadrotor_multi.collisions.quadrotors import calculate_collision_matrix, \
    calculate_drone_proximity_penalties, perform_collision_between_drones
from gym_art.quadrotor_multi.collisions.room import perform_collision_with_wall, perform_collision_with_ceiling
from gym_art.quadrotor_multi.obstacles.utils import get_cell_centers
from gym_art.quadrotor_multi.quad_utils import QUADS_OBS_REPR, QUADS_NEIGHBOR_OBS_TYPE

from gym_art.quadrotor_multi.obstacles.obstacles import MultiObstacles
from gym_art.quadrotor_multi.quadrotor_multi_visualization import Quadrotor3DSceneMulti
from gym_art.quadrotor_multi.quadrotor_single import QuadrotorSingle
from gym_art.quadrotor_multi.scenarios.mix import create_scenario

"""num_agents（智能体数量）： 定义了环境中无人机的数量，即智能体的数量。
rew_coeff（奖励系数）： 一个字典，定义了奖励函数中各个组成部分的权重。例如，{'pos': 1.0, 'effort': 0.05}，表示位置误差的权重为 1.0，控制努力的权重为 0.05。
obs_repr（观测表示）： 定义了观测向量中包含的内容，以字符串形式指定，用下划线分隔。例如，"xyz_vxyz_R_omega" 表示观测包括位置、速度、旋转矩阵和角速度。
neighbor_visible_num（可见邻居数量）： 定义了每个智能体可以观测到的邻居数量。-1 表示可以观测到所有其他智能体。
neighbor_obs_type（邻居观测类型）： 指定了邻居观测的类型，如仅位置（"pos"）、位置和速度（"pos_vel"）等。
collision_hitbox_radius（碰撞检测半径）： 定义了用于碰撞检测的碰撞箱（hitbox）半径，当两个智能体的距离小于该半径时，认为发生了碰撞。
collision_falloff_radius（碰撞惩罚衰减半径）： 定义了碰撞惩罚开始衰减的距离，当智能体之间的距离大于 collision_hitbox_radius 但小于 collision_falloff_radius 时，碰撞惩罚会逐渐减小。
use_obstacles（使用障碍物）： 布尔值，指示是否在环境中添加障碍物。
obst_density（障碍物密度）： 定义障碍物在空间中的密度，用于确定生成多少障碍物。
obst_size（障碍物大小）： 定义障碍物的尺寸。
obst_spawn_area（障碍物生成区域）： 三维空间中障碍物生成的区域范围。
use_downwash（使用下洗气流效应）： 布尔值，指示是否模拟无人机之间的下洗气流效应，这是多旋翼无人机在接近时产生的气动干扰。
use_numba（使用 Numba 加速）： 布尔值，指示是否使用 Numba 库对数值计算进行加速。
quads_mode（无人机模式）： 指定无人机群的行为模式或场景，例如 "mix" 模式可能表示混合的任务或编队。
room_dims（房间尺寸）： 定义环境的尺寸，三维元组，表示房间的长度、宽度和高度。
use_replay_buffer（使用重放缓冲区）： 布尔值，指示是否使用重放缓冲区，通常在某些强化学习训练库中使用，用于经验回放。
quads_view_mode（无人机视角模式）： 列表，指定渲染时的视角，如 "chase"（追逐视角）、"top_down"（俯视视角）等。
quads_render（渲染环境）： 布尔值，指示是否渲染环境。
dynamics_params（动力学参数）： 字典，定义了无人机的物理动力学参数，如质量、惯性矩、最大推力等。
raw_control（原始控制）： 布尔值，指示是否使用原始控制，即直接控制电机推力。如果为 False，可能使用高级控制策略。
zero_action_middle（控制输入零中心化）： 布尔值，指示控制输入是否以零为中心，即控制输入范围是否为 [-1, 1]（零中心）或 [0, 1]。
dynamics_randomize_every（动力学参数随机化频率）： 整数，表示每隔多少个 episode 对动力学参数进行一次随机化。
dynamics_change（动力学参数修改）： 字典，用于在初始动力学参数的基础上进行修改，便于测试不同的动力学配置。
dyn_sampler_1（动力学参数采样器）： 定义了第一个动力学参数采样器，可用于在训练过程中随机化或扰动动力学参数。
sense_noise（传感器噪声）： 指定传感器观测中的噪声参数，可用于模拟真实世界中的传感器不准确性。
init_random_state（随机初始状态）： 布尔值，指示智能体的初始状态是否随机化。
render_mode（渲染模式）： 字符串，指定渲染模式，如 "human" 表示用于人类观看的渲染。"""

class QuadrotorEnvMulti(gym.Env):
    def __init__(self, num_agents=1,  rew_coeff=None, obs_repr='xyz_vxyz_R_omega',
                 # Neighbor
                 neighbor_visible_num=-1, neighbor_obs_type='pos_vel', collision_hitbox_radius=1.0, collision_falloff_radius=1.2,

                 # Obstacle
                 use_obstacles=False, obst_density=0.0, obst_size=[0.15, 0.15, 0.75], obst_spawn_area=[5.0, 5.0],

                 # Aerodynamics, Numba Speed Up, Scenarios, Room, Replay Buffer, Rendering
                 use_downwash=False, use_numba=False, quads_mode='static_same_goal', room_dims=(10.0, 10.0, 10.0), use_replay_buffer=False, quads_view_mode=['topdown'],
                 render=False,

                 # Quadrotor Specific (Do Not Change)
                 dynamics_params='Crazyflie',  zero_action_middle=True, control_input='pos', if_discrete=False,
                 dynamics_randomize_every=None, dynamics_change=dict(noise=dict(thrust_noise_ratio=0.05), damp=dict(vel=0, omega_quadratic=0)), 
                 dyn_sampler_1= None,
                 sense_noise='default', init_random_state=True,
                 # Rendering
                 render_mode='human', sim_freq=200
                 ):
        super().__init__()

        # Predefined Parameters
        self.num_agents = num_agents
        obs_self_size = QUADS_OBS_REPR[obs_repr]
        if neighbor_visible_num == -1:
            self.num_use_neighbor_obs = self.num_agents - 1
        else:
            self.num_use_neighbor_obs = neighbor_visible_num

        # Set to True means that sample_factory will treat it as a multi-agent vectorized environment even with
        # num_agents=1. More info, please look at sample-factory: envs/quadrotors/wrappers/reward_shaping.py
        self.is_multiagent = True
        self.room_dims = room_dims
        self.quads_view_mode = quads_view_mode

        # Generate All Quadrotors
        self.envs = []
        for i in range(self.num_agents):
            e = QuadrotorSingle(
                # Quad Parameters
                dynamics_params=dynamics_params, dynamics_change=dynamics_change,
                dynamics_randomize_every=dynamics_randomize_every, dyn_sampler_1=dyn_sampler_1,
                zero_action_middle=zero_action_middle, sense_noise=sense_noise,
                init_random_state=init_random_state, obs_repr=obs_repr,  room_dims=room_dims,
                use_numba=use_numba, sim_freq=sim_freq,
                # Neighbor
                num_agents=num_agents,
                neighbor_obs_type=neighbor_obs_type, num_use_neighbor_obs=self.num_use_neighbor_obs,
                # Obstacle
                use_obstacles=use_obstacles,
                control_input=control_input, if_discrete=if_discrete,
            )
            self.envs.append(e)

        # Set Obs & Act
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.observation_spaces = [e.observation_space for e in self.envs]
        self.action_spaces = [e.action_space for e in self.envs]

        # Aux variables
        self.quad_arm = self.envs[0].dynamics.arm  # 无人机臂长
        self.control_freq = self.envs[0].control_freq  # 控制频率
        self.control_dt = 1.0 / self.control_freq  # 控制时间步长
        self.pos = np.zeros([self.num_agents, 3])  # 所有无人机的位置
        self.vel = np.zeros([self.num_agents, 3])  # 所有无人机的速度
        self.omega = np.zeros([self.num_agents, 3])  # 所有无人机的角速度
        self.rel_pos = np.zeros((self.num_agents, self.num_agents, 3))  # 相对位置
        self.rel_vel = np.zeros((self.num_agents, self.num_agents, 3))  # 相对速度
        

        # Reward
        self.rew_coeff = dict(
            pos=1., effort=0.05, action_change=0., crash=1., orient=1., yaw=0., rot=0., attitude=0., spin=0.1, vel=0.,
            quadcol_bin=0., quadcol_bin_smooth_max=0, quadcol_bin_obst=0, time=0.01
        )
        rew_coeff_orig = copy.deepcopy(self.rew_coeff)

        if rew_coeff is not None:
            assert isinstance(rew_coeff, dict)
            assert set(rew_coeff.keys()).issubset(set(self.rew_coeff.keys()))
            self.rew_coeff.update(rew_coeff)# 更新奖励系数
        for key in self.rew_coeff.keys():
            self.rew_coeff[key] = float(self.rew_coeff[key])

        orig_keys = list(rew_coeff_orig.keys())
        # Checking to make sure we didn't provide some false rew_coeffs (for example by misspelling one of the params)
        assert np.all([key in orig_keys for key in self.rew_coeff.keys()])

        # Neighbors
        neighbor_obs_size = QUADS_NEIGHBOR_OBS_TYPE[neighbor_obs_type]

        self.clip_neighbor_space_length = self.num_use_neighbor_obs * neighbor_obs_size
        self.clip_neighbor_space_min_box = self.observation_space.low[
                                           obs_self_size:obs_self_size + self.clip_neighbor_space_length]
        self.clip_neighbor_space_max_box = self.observation_space.high[
                                           obs_self_size:obs_self_size + self.clip_neighbor_space_length]

        # 障碍物
        self.use_obstacles = use_obstacles
        self.obstacles = None
        self.num_obstacles = 0
        if self.use_obstacles:
            self.prev_obst_quad_collisions = []  # 之前的无人机与障碍物碰撞
            self.obst_quad_collisions_per_episode = 0  # 每个 episode 中无人机与障碍物的碰撞次数
            self.obst_quad_collisions_after_settle = 0  # 稳定后（起飞后）无人机与障碍物的碰撞次数
            self.curr_quad_col = []  # 当前的无人机与障碍物碰撞
            self.obst_density = obst_density  # 障碍物密度
            self.obst_spawn_area = obst_spawn_area  # 障碍物生成区域
            self.num_obstacles = int(obst_density * obst_spawn_area[0] * obst_spawn_area[1])  # 计算障碍物数量
            self.obst_map = None  # 障碍物地图
            self.obst_size = obst_size  # 障碍物尺寸

            # 记录更多信息
            self.distance_to_goal_3_5 = 0  # 距离目标超过3.5米的次数
            self.distance_to_goal_5 = 0  # 距离目标超过5米的次数

        # Scenarios
        self.quads_mode = quads_mode
        self.scenario = create_scenario(quads_mode=quads_mode, envs=self.envs, num_agents=num_agents,
                                        room_dims=room_dims)

        # Collisions
        # # Collisions: Neighbors
        self.collisions_per_episode = 0 # 每个 episode 的总碰撞次数
        # # # Ignore collisions because of spawn
        self.collisions_after_settle = 0# 稳定后（起飞后）的碰撞次数
        self.collisions_grace_period_steps = 1.5 * self.control_freq# 起飞后1.5秒的宽限期, if control_freq is hz? or ms?
        self.collisions_grace_period_seconds = 1.5# 起飞后1.5秒的宽限期
        self.prev_drone_collisions = []# 最后5秒的碰撞次数

        self.collisions_final_grace_period_steps = 5.0 * self.control_freq # 结束前5秒的宽限期
        self.collisions_final_5s = 0  

        # # # Dense reward info
        self.collision_threshold = collision_hitbox_radius * self.quad_arm# 碰撞阈值（实际半径）
        self.collision_falloff_threshold = collision_falloff_radius * self.quad_arm # 碰撞衰减阈值

        # # Collisions: Room
        self.collisions_room_per_episode = 0  # 与房间的碰撞次数
        self.collisions_floor_per_episode = 0  # 与地板的碰撞次数
        self.collisions_wall_per_episode = 0  # 与墙壁的碰撞次数
        self.collisions_ceiling_per_episode = 0  # 与天花板的碰撞次数

        self.prev_crashed_walls = []  # 之前的墙壁碰撞列表
        self.prev_crashed_ceiling = []  # 之前的天花板碰撞列表
        self.prev_crashed_room = []  # 之前的房间碰撞列表

        # Replay
        self.use_replay_buffer = use_replay_buffer
        # # only start using the buffer after the drones learn how to fly
        self.activate_replay_buffer = False
        # # since the same collisions happen during replay, we don't want to keep resaving the same event
        self.saved_in_replay_buffer = False
        self.last_step_unique_collisions = False
        self.crashes_in_recent_episodes = deque([], maxlen=100)
        self.crashes_last_episode = 0

        # Numba
        self.use_numba = use_numba

        # Aerodynamics
        self.use_downwash = use_downwash

        # Rendering
        # # set to true whenever we need to reset the OpenGL scene in render()
        self.render_mode =render_mode
        self.quads_render = render
        self.scenes = []
        if self.quads_render:
            self.reset_scene = False
            self.simulation_start_time = 0
            self.frames_since_last_render = self.render_skip_frames = 0
            self.render_every_nth_frame = 1
            # # Use this to control rendering speed
            self.render_speed = 1.0
            self.quads_formation_size = 2.0
            self.all_collisions = {}

        # Log
        self.distance_to_goal = [[] for _ in range(len(self.envs))]
        self.reached_goal = [False for _ in range(len(self.envs))]

        # Log metric
        self.agent_col_agent = np.ones(self.num_agents)  # 记录每个智能体是否与其他智能体碰撞（1：未碰撞，0：碰撞）
        self.agent_col_obst = np.ones(self.num_agents)  # 记录每个智能体是否与障碍物碰撞（1：未碰撞，0：碰撞）

        # Others
        self.apply_collision_force = False # 是否应用碰撞力 无人机之间的碰撞力处理

    def all_dynamics(self):
        return tuple(e.dynamics for e in self.envs)

    def get_rel_pos_vel_item(self, env_id, indices=None):
        i = env_id

        if indices is None:
            # if not specified explicitly, consider all neighbors
            indices = [j for j in range(self.num_agents) if j != i]

        cur_pos = self.pos[i]
        cur_vel = self.vel[i]
        pos_neighbor = np.stack([self.pos[j] for j in indices])
        vel_neighbor = np.stack([self.vel[j] for j in indices])
        pos_rel = pos_neighbor - cur_pos
        vel_rel = vel_neighbor - cur_vel
        return pos_rel, vel_rel

    def get_obs_neighbor_rel(self, env_id, closest_drones):
        i = env_id
        pos_neighbors_rel, vel_neighbors_rel = self.get_rel_pos_vel_item(env_id=i, indices=closest_drones[i])
        obs_neighbor_rel = np.concatenate((pos_neighbors_rel, vel_neighbors_rel), axis=1)
        return obs_neighbor_rel

    def extend_obs_space(self, obs, closest_drones):
        obs_neighbors = []
        for i in range(len(self.envs)):
            obs_neighbor_rel = self.get_obs_neighbor_rel(env_id=i, closest_drones=closest_drones)
            obs_neighbors.append(obs_neighbor_rel.reshape(-1))
        obs_neighbors = np.stack(obs_neighbors)

        # clip observation space of neighborhoods
        obs_neighbors = np.clip(
            obs_neighbors, a_min=self.clip_neighbor_space_min_box, a_max=self.clip_neighbor_space_max_box,
        )
        obs_ext = np.concatenate((obs, obs_neighbors), axis=1)
        return obs_ext

    def neighborhood_indices(self):
        """Return a list of closest drones for each drone in the swarm."""
        # indices of all the other drones except us
        indices = [[j for j in range(self.num_agents) if i != j] for i in range(self.num_agents)]
        indices = np.array(indices)

        if self.num_use_neighbor_obs == self.num_agents - 1:
            return indices
        elif 1 <= self.num_use_neighbor_obs < self.num_agents - 1:
            close_neighbor_indices = []

            for i in range(self.num_agents):
                rel_pos, rel_vel = self.get_rel_pos_vel_item(env_id=i, indices=indices[i])
                rel_dist = np.linalg.norm(rel_pos, axis=1)
                rel_dist = np.maximum(rel_dist, 0.01)
                rel_pos_unit = rel_pos / rel_dist[:, None]

                # new relative distance is a new metric that combines relative position and relative velocity
                # the smaller the new_rel_dist, the closer the drones
                new_rel_dist = rel_dist + np.sum(rel_pos_unit * rel_vel, axis=1)

                rel_pos_index = new_rel_dist.argsort()
                rel_pos_index = rel_pos_index[:self.num_use_neighbor_obs]
                close_neighbor_indices.append(indices[i][rel_pos_index])

            return close_neighbor_indices
        else:
            raise RuntimeError("Incorrect number of neigbors")

    def add_neighborhood_obs(self, obs):
        indices = self.neighborhood_indices()
        obs_ext = self.extend_obs_space(obs, closest_drones=indices)
        return obs_ext

    def can_drones_fly(self):
        """
        Here we count the average number of collisions with the walls and ground in the last N episodes
        Returns: True if drones are considered proficient at flying
        """
        res = abs(np.mean(self.crashes_in_recent_episodes)) < 1 and len(self.crashes_in_recent_episodes) >= 10
        return res

    def calculate_room_collision(self):
        floor_collisions = np.array([env.dynamics.crashed_floor for env in self.envs])
        wall_collisions = np.array([env.dynamics.crashed_wall for env in self.envs])
        ceiling_collisions = np.array([env.dynamics.crashed_ceiling for env in self.envs])

        floor_crash_list = np.where(floor_collisions >= 1)[0]

        cur_wall_crash_list = np.where(wall_collisions >= 1)[0]
        wall_crash_list = np.setdiff1d(cur_wall_crash_list, self.prev_crashed_walls)

        cur_ceiling_crash_list = np.where(ceiling_collisions >= 1)[0]
        ceiling_crash_list = np.setdiff1d(cur_ceiling_crash_list, self.prev_crashed_ceiling)

        return floor_crash_list, wall_crash_list, ceiling_crash_list

    def obst_generation_given_density(self, grid_size=1.0):
        obst_area_length, obst_area_width = int(self.obst_spawn_area[0]), int(self.obst_spawn_area[1])
        num_room_grids = obst_area_length * obst_area_width

        cell_centers = get_cell_centers(obst_area_length=obst_area_length, obst_area_width=obst_area_width,
                                        grid_size=grid_size)

        room_map = [i for i in range(0, num_room_grids)]

        obst_index = np.random.choice(a=room_map, size=int(num_room_grids * self.obst_density), replace=False)

        obst_pos_arr = []
        # 0: No Obst, 1: Obst
        obst_map = np.zeros([obst_area_length, obst_area_width])
        for obst_id in obst_index:
            rid, cid = obst_id // obst_area_width, obst_id - (obst_id // obst_area_width) * obst_area_width
            obst_map[rid, cid] = 1
            obst_item = list(cell_centers[rid + int(obst_area_length / grid_size) * cid])
            obst_item.append(self.room_dims[2] / 2.)
            obst_pos_arr.append(obst_item)

        return obst_map, obst_pos_arr, cell_centers

    def init_scene_multi(self):
        models = tuple(e.dynamics.model for e in self.envs)
        for i in range(len(self.quads_view_mode)):
            self.scenes.append(Quadrotor3DSceneMulti(
                models=models,
                w=600, h=480, resizable=True, viewpoint=self.quads_view_mode[i],
                room_dims=self.room_dims, num_agents=self.num_agents,
                render_speed=self.render_speed, formation_size=self.quads_formation_size, obstacles=self.obstacles,
                vis_vel_arrows=False, vis_acc_arrows=True, viz_traces=25, viz_trace_nth_step=1,
                num_obstacles=self.num_obstacles, scene_index=i
            ))

    def reset(self, obst_density=None, obst_size=None):
        obs, rewards, dones, infos = [], [], [], []

        if obst_density:
            self.obst_density = obst_density
        if obst_size:
            self.obst_size = obst_size

        # Scenario reset
        if self.use_obstacles:
            self.obstacles = MultiObstacles(obstacle_size=self.obst_size, quad_radius=self.quad_arm)
            self.obst_map, obst_pos_arr, cell_centers = self.obst_generation_given_density()
            self.scenario.reset(obst_map=self.obst_map, cell_centers=cell_centers)
        else:
            self.scenario.reset()

        # Replay buffer
        if self.use_replay_buffer and not self.activate_replay_buffer:
            self.crashes_in_recent_episodes.append(self.crashes_last_episode)
            self.activate_replay_buffer = self.can_drones_fly()
            self.crashes_last_episode = 0

        for i, e in enumerate(self.envs):
            e.goal = self.scenario.goals[i]
            if self.scenario.spawn_points is None:
                e.spawn_point = self.scenario.goals[i]
            else:
                e.spawn_point = self.scenario.spawn_points[i]
            e.rew_coeff = self.rew_coeff

            observation = e.reset()
            obs.append(observation)
            self.pos[i, :] = e.dynamics.pos

        # Neighbors
        if self.num_use_neighbor_obs > 0:
            obs = self.add_neighborhood_obs(obs)

        # Obstacles
        if self.use_obstacles:
            quads_pos = np.array([e.dynamics.pos for e in self.envs])
            obs = self.obstacles.reset(obs=obs, quads_pos=quads_pos, pos_arr=obst_pos_arr)
            self.obst_quad_collisions_per_episode = self.obst_quad_collisions_after_settle = 0
            self.prev_obst_quad_collisions = []
            self.distance_to_goal_3_5 = 0
            self.distance_to_goal_5 = 0

        # Collision
        # # Collision: Neighbor
        self.collisions_per_episode = self.collisions_after_settle = self.collisions_final_5s = 0
        self.prev_drone_collisions = []

        # # Collision: Room
        self.collisions_room_per_episode = 0
        self.collisions_floor_per_episode = self.collisions_wall_per_episode = self.collisions_ceiling_per_episode = 0
        self.prev_crashed_walls = []
        self.prev_crashed_ceiling = []
        self.prev_crashed_room = []

        # Log
        # # Final Distance (1s / 3s / 5s)
        self.distance_to_goal = [[] for _ in range(len(self.envs))]
        self.agent_col_agent = np.ones(self.num_agents)
        self.agent_col_obst = np.ones(self.num_agents)
        self.reached_goal = [False for _ in range(len(self.envs))]

        # Rendering
        if self.quads_render:
            self.reset_scene = True
            self.quads_formation_size = self.scenario.formation_size
            self.all_collisions = {val: [0.0 for _ in range(len(self.envs))] for val in ['drone', 'ground', 'obstacle']}

        return obs

    def step(self, actions):
        obs, rewards, dones, infos = [], [], [], []

        for i, a in enumerate(actions):
            self.envs[i].rew_coeff = self.rew_coeff

            observation, reward, done, info = self.envs[i].step(a)
            obs.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

            self.pos[i, :] = self.envs[i].dynamics.pos

        # 1. Calculate collisions: 1) between drones 2) with obstacles 3) with room
        # 1) Collisions between drones
        drone_col_matrix, curr_drone_collisions, distance_matrix = \
            calculate_collision_matrix(positions=self.pos, collision_threshold=self.collision_threshold)
        # drone_col_matrix：无人机碰撞矩阵（对称矩阵）
        # curr_drone_collisions：当前碰撞的无人机对列表
        # distance_matrix：无人机之间的距离信息

        # 过滤 curr_drone_collisions，去除无效项[-1000,-1000],which supposed to be init num
        curr_drone_collisions = curr_drone_collisions.astype(int)
        curr_drone_collisions = np.delete(curr_drone_collisions, np.unique(
            np.where(curr_drone_collisions == [-1000, -1000])[0]), axis=0)

        old_quad_collision = set(map(tuple, self.prev_drone_collisions))
        new_quad_collision = np.array([x for x in curr_drone_collisions if tuple(x) not in old_quad_collision])

        self.last_step_unique_collisions = np.setdiff1d(curr_drone_collisions, self.prev_drone_collisions)
        # 过滤距离矩阵，只保留距离小于碰撞衰减阈值的无人机对
        # # Filter distance_matrix; Only contains quadrotor pairs with distance <= self.collision_threshold
        near_quad_ids = np.where(distance_matrix[:, 2] <= self.collision_falloff_threshold)
        distance_matrix = distance_matrix[near_quad_ids]

        # Collision between 2 drones counts as a single collision
        # # Calculate collisions (i) All collisions (ii) collisions after grace period
        collisions_curr_tick = len(self.last_step_unique_collisions) // 2 # 每次碰撞涉及两个无人机，因此除以2
        self.collisions_per_episode += collisions_curr_tick # 更新总碰撞次数
                # 如果碰撞发生在稳定后（起飞后）
        if collisions_curr_tick > 0 and self.envs[0].tick >= self.collisions_grace_period_steps:
            self.collisions_after_settle += collisions_curr_tick
            for agent_id in self.last_step_unique_collisions:
                self.agent_col_agent[agent_id] = 0 # 标记发生碰撞的智能体


        # # Aux: Neighbor Collisions
        # 更新之前的碰撞信息
        self.prev_drone_collisions = curr_drone_collisions

        # 2) Collisions with obstacles
        if self.use_obstacles:
            rew_obst_quad_collisions_raw = np.zeros(self.num_agents)
            obst_quad_col_matrix, quad_obst_pair = self.obstacles.collision_detection(pos_quads=self.pos)
            # obst_quad_col_matrix：与障碍物发生碰撞的无人机ID列表
            # quad_obst_pair：无人机与障碍物的对应关系
            # We assume drone can only collide with one obstacle at the same time.
            # Given this setting, in theory, the gap between obstacles should >= 0.1 (drone diameter: 0.46*2 = 0.92)
            self.curr_quad_col = np.setdiff1d(obst_quad_col_matrix, self.prev_obst_quad_collisions)
            collisions_obst_curr_tick = len(self.curr_quad_col)
            self.obst_quad_collisions_per_episode += collisions_obst_curr_tick

            if collisions_obst_curr_tick > 0 and self.envs[0].tick >= self.collisions_grace_period_steps:
                self.obst_quad_collisions_after_settle += collisions_obst_curr_tick
                for qid in self.curr_quad_col:
                    q_rel_dist = np.linalg.norm(obs[qid][0:3])
                    if q_rel_dist > 3.5:
                        self.distance_to_goal_3_5 += 1
                    if q_rel_dist > 5.0:
                        self.distance_to_goal_5 += 1
                    # Used for log agent_success
                    self.agent_col_obst[qid] = 0

            # # Aux: Obstacle Collisions
            self.prev_obst_quad_collisions = obst_quad_col_matrix
            # 对发生碰撞的无人机赋予奖励惩罚
            if len(obst_quad_col_matrix) > 0:
                # We assign penalties to the drones which collide with the obstacles
                # And obst_quad_last_step_unique_collisions only include drones' id
                rew_obst_quad_collisions_raw[self.curr_quad_col] = -1.0

        # 3) Collisions with room
        floor_crash_list, wall_crash_list, ceiling_crash_list = self.calculate_room_collision()
        room_crash_list = np.unique(np.concatenate([floor_crash_list, wall_crash_list, ceiling_crash_list]))
        room_crash_list = np.setdiff1d(room_crash_list, self.prev_crashed_room)
        # # Aux: Room Collisions
        self.prev_crashed_walls = wall_crash_list
        self.prev_crashed_ceiling = ceiling_crash_list
        self.prev_crashed_room = room_crash_list

        # 2. Calculate rewards and infos for collision
        # 1) Between drones
        rew_collisions_raw = np.zeros(self.num_agents)
        if self.last_step_unique_collisions.any():
            rew_collisions_raw[self.last_step_unique_collisions] = -1.0 # 碰撞惩罚为 -1.0
        rew_collisions = self.rew_coeff["quadcol_bin"] * rew_collisions_raw

        # penalties for being too close to other drones
        if len(distance_matrix) > 0:
            rew_proximity = -1.0 * calculate_drone_proximity_penalties(
                distance_matrix=distance_matrix, collision_falloff_threshold=self.collision_falloff_threshold,
                dt=self.control_dt, max_penalty=self.rew_coeff["quadcol_bin_smooth_max"], num_agents=self.num_agents,
            )
        else:
            rew_proximity = np.zeros(self.num_agents)

        # 2) With obstacles
        rew_collisions_obst_quad = np.zeros(self.num_agents)
        if self.use_obstacles:
            rew_collisions_obst_quad = self.rew_coeff["quadcol_bin_obst"] * rew_obst_quad_collisions_raw

        # 3) With room
        # # TODO: reward penalty
        

        # 更新奖励和信息
        if self.envs[0].tick >= self.collisions_grace_period_steps:
            self.collisions_room_per_episode += len(room_crash_list)
            self.collisions_floor_per_episode += len(floor_crash_list)
            self.collisions_wall_per_episode += len(wall_crash_list)
            self.collisions_ceiling_per_episode += len(ceiling_crash_list)

        # Reward & Info
        for i in range(self.num_agents):
            rewards[i] += rew_collisions[i]  # 添加无人机间碰撞的奖励
            rewards[i] += rew_proximity[i]  # 添加接近惩罚

            infos[i]["rewards"]["rew_quadcol"] = rew_collisions[i]  # 无人机间碰撞奖励
            infos[i]["rewards"]["rew_proximity"] = rew_proximity[i]  # 接近惩罚
            infos[i]["rewards"]["rewraw_quadcol"] = rew_collisions_raw[i]  # 原始碰撞奖励

            if self.use_obstacles:
                rewards[i] += rew_collisions_obst_quad[i]  # 添加与障碍物碰撞的奖励
                infos[i]["rewards"]["rew_quadcol_obstacle"] = rew_collisions_obst_quad[i]  # 障碍物碰撞奖励
                infos[i]["rewards"]["rewraw_quadcol_obstacle"] = rew_obst_quad_collisions_raw[i]  # 原始障碍物碰撞奖励

            # 记录到目标的距离，用于计算到达目标的情况
            self.distance_to_goal[i].append(-infos[i]["rewards"]["rewraw_pos"])  # 假设 rewraw_pos 是位置误差的负值
            if len(self.distance_to_goal[i]) >= 5 and \
                    np.mean(self.distance_to_goal[i][-5:]) / self.envs[0].dt < self.scenario.approch_goal_metric \
                    and not self.reached_goal[i]:
                self.reached_goal[i] = True  # 标记达到目标

        # 3. 应用随机力：1）气动效应 2）无人机之间的碰撞 3）与障碍物的碰撞 4）与房间的碰撞

        # 3. Applying random forces: 1) aerodynamics 2) between drones 3) obstacles 4) room
        self_state_update_flag = False

        # # 1) aerodynamics 气动效应（如下洗气流）
        if self.use_downwash:
            envs_dynamics = [env.dynamics for env in self.envs]
            applied_downwash_list = perform_downwash(drones_dyn=envs_dynamics, dt=self.control_dt)
            downwash_agents_list = np.where(applied_downwash_list == 1)[0]
            if len(downwash_agents_list) > 0:
                self_state_update_flag = True

        # # 2) Drones无人机之间的碰撞力处理
        if self.apply_collision_force:
            if len(new_quad_collision) > 0:
                self_state_update_flag = True
                for val in new_quad_collision:
                    dyn1, dyn2 = self.envs[val[0]].dynamics, self.envs[val[1]].dynamics
                    dyn1.vel, dyn1.omega, dyn2.vel, dyn2.omega = perform_collision_between_drones(
                        pos1=dyn1.pos, vel1=dyn1.vel, omega1=dyn1.omega, pos2=dyn2.pos, vel2=dyn2.vel, omega2=dyn2.omega)

            # # 3) Obstacles 与障碍物的碰撞力处理
            if self.use_obstacles:
                if len(self.curr_quad_col) > 0:
                    self_state_update_flag = True
                    for val in self.curr_quad_col:
                        obstacle_id = quad_obst_pair[int(val)]
                        obstacle_pos = self.obstacles.pos_arr[int(obstacle_id)]
                        perform_collision_with_obstacle(drone_dyn=self.envs[int(val)].dynamics,
                                                        obstacle_pos=obstacle_pos,
                                                        obstacle_size=self.obst_size)

            # # 4) Room 与房间（墙壁、天花板）的碰撞力处理
            if len(wall_crash_list) > 0 or len(ceiling_crash_list) > 0:
                self_state_update_flag = True

                for val in wall_crash_list:
                    perform_collision_with_wall(drone_dyn=self.envs[val].dynamics, room_box=self.envs[0].room_box)

                for val in ceiling_crash_list:
                    perform_collision_with_ceiling(drone_dyn=self.envs[val].dynamics)

        # 4. Run the scenario passed to self.quads_mode
        self.scenario.step()

        # 5. Collect final observations
        # Collect positions after physical interaction
        for i in range(self.num_agents):
            self.pos[i, :] = self.envs[i].dynamics.pos
            self.vel[i, :] = self.envs[i].dynamics.vel

        if self_state_update_flag:
            obs = [e.state_vector(e) for e in self.envs]

        # Concatenate observations of neighbor drones
        if self.num_use_neighbor_obs > 0:
            obs = self.add_neighborhood_obs(obs)

        # Concatenate obstacle observations
        if self.use_obstacles:
            obs = self.obstacles.step(obs=obs, quads_pos=self.pos)

        # 6. Update info for replay buffer
        # Once agent learns how to take off, activate the replay buffer
        if self.use_replay_buffer and not self.activate_replay_buffer:
            self.crashes_last_episode += infos[0]["rewards"]["rew_crash"]

        # Rendering
        if self.quads_render:
            # Collisions with room
            ground_collisions = [1.0 if env.dynamics.on_floor else 0.0 for env in self.envs]
            if self.use_obstacles:
                obst_coll = [1.0 if i < 0 else 0.0 for i in rew_obst_quad_collisions_raw]
            else:
                obst_coll = [0.0 for _ in range(self.num_agents)]
            self.all_collisions = {'drone': drone_col_matrix, 'ground': ground_collisions,
                                   'obstacle': obst_coll}

        # 7. DONES
        if any(dones):
            scenario_name = self.scenario.name()[9:]
            for i in range(len(infos)):
                if self.saved_in_replay_buffer:
                    infos[i]['episode_extra_stats'] = {
                        'num_collisions_replay': self.collisions_per_episode,
                        'num_collisions_obst_replay': self.obst_quad_collisions_per_episode,
                    }
                else:
                    self.distance_to_goal = np.array(self.distance_to_goal)
                    self.reached_goal = np.array(self.reached_goal)
                    infos[i]['episode_extra_stats'] = {
                        'num_collisions': self.collisions_per_episode,
                        'num_collisions_with_room': self.collisions_room_per_episode,
                        'num_collisions_with_floor': self.collisions_floor_per_episode,
                        'num_collisions_with_wall': self.collisions_wall_per_episode,
                        'num_collisions_with_ceiling': self.collisions_ceiling_per_episode,
                        'num_collisions_after_settle': self.collisions_after_settle,
                        f'{scenario_name}/num_collisions': self.collisions_after_settle,

                        'num_collisions_final_5_s': self.collisions_final_5s,
                        f'{scenario_name}/num_collisions_final_5_s': self.collisions_final_5s,

                        'distance_to_goal_1s': (1.0 / self.envs[0].dt) * np.mean(
                            self.distance_to_goal[i, int(-1 * self.control_freq):]),
                        'distance_to_goal_3s': (1.0 / self.envs[0].dt) * np.mean(
                            self.distance_to_goal[i, int(-3 * self.control_freq):]),
                        'distance_to_goal_5s': (1.0 / self.envs[0].dt) * np.mean(
                            self.distance_to_goal[i, int(-5 * self.control_freq):]),

                        f'{scenario_name}/distance_to_goal_1s': (1.0 / self.envs[0].dt) * np.mean(
                            self.distance_to_goal[i, int(-1 * self.control_freq):]),
                        f'{scenario_name}/distance_to_goal_3s': (1.0 / self.envs[0].dt) * np.mean(
                            self.distance_to_goal[i, int(-3 * self.control_freq):]),
                        f'{scenario_name}/distance_to_goal_5s': (1.0 / self.envs[0].dt) * np.mean(
                            self.distance_to_goal[i, int(-5 * self.control_freq):]),
                    }

                    if self.use_obstacles:
                        infos[i]['episode_extra_stats']['num_collisions_obst_quad'] = \
                            self.obst_quad_collisions_per_episode
                        infos[i]['episode_extra_stats']['num_collisions_obst_quad_after_settle'] = \
                            self.obst_quad_collisions_after_settle
                        infos[i]['episode_extra_stats'][f'{scenario_name}/num_collisions_obst'] = \
                            self.obst_quad_collisions_per_episode

                        infos[i]['episode_extra_stats']['num_collisions_obst_quad_3_5'] = \
                            self.distance_to_goal_3_5
                        infos[i]['episode_extra_stats'][f'{scenario_name}/num_collisions_obst_quad_3_5'] = \
                            self.distance_to_goal_3_5

                        infos[i]['episode_extra_stats']['num_collisions_obst_quad_5'] = \
                            self.distance_to_goal_5
                        infos[i]['episode_extra_stats'][f'{scenario_name}/num_collisions_obst_quad_5'] = \
                            self.distance_to_goal_5

            if not self.saved_in_replay_buffer:
                # agent_success_rate: base_success_rate, based on per agent
                # 0: collision; 1: no collision
                agent_col_flag_list = np.logical_and(self.agent_col_agent, self.agent_col_obst)
                agent_success_flag_list = np.logical_and(agent_col_flag_list, self.reached_goal)
                agent_success_ratio = 1.0 * np.sum(agent_success_flag_list) / self.num_agents

                # agent_deadlock_rate
                # Doesn't approach to the goal while no collisions with other objects
                agent_deadlock_list = np.logical_and(agent_col_flag_list, 1 - self.reached_goal)
                agent_deadlock_ratio = 1.0 * np.sum(agent_deadlock_list) / self.num_agents

                # agent_col_rate
                # Collide with other drones and obstacles
                agent_col_ratio = 1.0 - np.sum(agent_col_flag_list) / self.num_agents

                # agent_neighbor_col_rate
                agent_neighbor_col_ratio = 1.0 - np.sum(self.agent_col_agent) / self.num_agents
                # agent_obst_col_rate
                agent_obst_col_ratio = 1.0 - np.sum(self.agent_col_obst) / self.num_agents

                for i in range(len(infos)):
                    # agent_success_rate
                    infos[i]['episode_extra_stats']['metric/agent_success_rate'] = agent_success_ratio
                    infos[i]['episode_extra_stats'][f'{scenario_name}/agent_success_rate'] = agent_success_ratio
                    # agent_deadlock_rate
                    infos[i]['episode_extra_stats']['metric/agent_deadlock_rate'] = agent_deadlock_ratio
                    infos[i]['episode_extra_stats'][f'{scenario_name}/agent_deadlock_rate'] = agent_deadlock_ratio
                    # agent_col_rate
                    infos[i]['episode_extra_stats']['metric/agent_col_rate'] = agent_col_ratio
                    infos[i]['episode_extra_stats'][f'{scenario_name}/agent_col_rate'] = agent_col_ratio
                    # agent_neighbor_col_rate
                    infos[i]['episode_extra_stats']['metric/agent_neighbor_col_rate'] = agent_neighbor_col_ratio
                    infos[i]['episode_extra_stats'][f'{scenario_name}/agent_neighbor_col_rate'] = agent_neighbor_col_ratio
                    # agent_obst_col_rate
                    infos[i]['episode_extra_stats']['metric/agent_obst_col_rate'] = agent_obst_col_ratio
                    infos[i]['episode_extra_stats'][f'{scenario_name}/agent_obst_col_rate'] = agent_obst_col_ratio

            obs = self.reset()
            # terminate the episode for all "sub-envs"
            dones = [True] * len(dones)
        # infos={} # reduce computation
        return obs, rewards, dones, infos

    def render(self, verbose=False):
        models = tuple(e.dynamics.model for e in self.envs)

        if len(self.scenes) == 0:
            self.init_scene_multi()

        if self.reset_scene:
            for i in range(len(self.scenes)):
                self.scenes[i].update_models(models)
                self.scenes[i].formation_size = self.quads_formation_size
                self.scenes[i].update_env(self.room_dims)

                self.scenes[i].reset(tuple(e.goal for e in self.envs), self.all_dynamics(), self.obstacles,
                                     self.all_collisions)

            self.reset_scene = False

        if self.quads_mode == "mix":
            for i in range(len(self.scenes)):
                self.scenes[i].formation_size = self.scenario.scenario.formation_size
        else:
            for i in range(len(self.scenes)):
                self.scenes[i].formation_size = self.scenario.formation_size
        self.frames_since_last_render += 1

        if self.render_skip_frames > 0:
            self.render_skip_frames -= 1
            return None

        # this is to handle the 1st step of the simulation that will typically be very slow
        if self.simulation_start_time > 0:
            simulation_time = time.time() - self.simulation_start_time
        else:
            simulation_time = 0

        realtime_control_period = 1 / self.control_freq

        render_start = time.time()
        goals = tuple(e.goal for e in self.envs)
        frames = []
        first_spawn = None
        for i in range(len(self.scenes)):
            frame, first_spawn = self.scenes[i].render_chase(all_dynamics=self.all_dynamics(), goals=goals,
                                                             collisions=self.all_collisions,
                                                             mode=self.render_mode, obstacles=self.obstacles,
                                                             first_spawn=first_spawn)
            frames.append(frame)
        # Update the formation size of the scenario
        if self.quads_mode == "mix":
            for i in range(len(self.scenes)):
                self.scenario.scenario.update_formation_size(self.scenes[i].formation_size)
        else:
            for i in range(len(self.scenes)):
                self.scenario.update_formation_size(self.scenes[i].formation_size)

        render_time = time.time() - render_start

        desired_time_between_frames = realtime_control_period * self.frames_since_last_render / self.render_speed
        time_to_sleep = desired_time_between_frames - simulation_time - render_time

        # wait so we don't simulate/render faster than realtime
        if self.render_mode == "human" and time_to_sleep > 0:
            time.sleep(time_to_sleep)

        if simulation_time + render_time > desired_time_between_frames:
            self.render_every_nth_frame += 1
            if verbose:
                print(f"Last render + simulation time {render_time + simulation_time:.3f}")
                print(f"Rendering does not keep up, rendering every {self.render_every_nth_frame} frames")
        elif simulation_time + render_time < realtime_control_period * (
                self.frames_since_last_render - 1) / self.render_speed:
            self.render_every_nth_frame -= 1
            if verbose:
                print(f"We can increase rendering framerate, rendering every {self.render_every_nth_frame} frames")

        if self.render_every_nth_frame > 5:
            self.render_every_nth_frame = 5
            if self.envs[0].tick % 20 == 0:
                print(f"Rendering cannot keep up! Rendering every {self.render_every_nth_frame} frames")

        self.render_skip_frames = self.render_every_nth_frame - 1
        self.frames_since_last_render = 0

        self.simulation_start_time = time.time()

        if self.render_mode == "rgb_array":
            return frame

    def __deepcopy__(self, memo):
        """OpenGL scene can't be copied naively."""

        cls = self.__class__
        copied_env = cls.__new__(cls)
        memo[id(self)] = copied_env

        # this will actually break the reward shaping functionality in PBT, but we need to fix it in SampleFactory, not here
        skip_copying = {"scene", "reward_shaping_interface"}

        for k, v in self.__dict__.items():
            if k not in skip_copying:
                setattr(copied_env, k, deepcopy(v, memo))

        # warning! deep-copied env has its scene uninitialized! We need to reuse one from the existing env
        # to avoid creating tons of windows
        copied_env.scene = None

        return copied_env
