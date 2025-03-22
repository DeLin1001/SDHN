#!/usr/bin/env python
"""
Quadrotor simulation for OpenAI Gym, with components reusable elsewhere.
Also see: D. Mellinger, N. Michael, V.Kumar. 
Trajectory Generation and Control for Precise Aggressive Maneuvers with Quadrotors
http://journals.sagepub.com/doi/pdf/10.1177/0278364911434236

Developers:
James Preiss, Artem Molchanov, Tao Chen 

References:
[1] RotorS: https://www.researchgate.net/profile/Fadri_Furrer/publication/309291237_RotorS_-_A_Modular_Gazebo_MAV_Simulator_Framework/links/5a0169c4a6fdcc82a3183f8f/RotorS-A-Modular-Gazebo-MAV-Simulator-Framework.pdf
[2] CrazyFlie modelling: http://mikehamer.info/assets/papers/Crazyflie%20Modelling.pdf
[3] HummingBird: http://www.asctec.de/en/uav-uas-drones-rpas-roav/asctec-hummingbird/
[4] CrazyFlie thrusters transition functions: https://www.bitcraze.io/2015/02/measuring-propeller-rpm-part-3/
[5] HummingBird modelling: https://digitalrepository.unm.edu/cgi/viewcontent.cgi?referer=https://www.google.com/&httpsredir=1&article=1189&context=ece_etds
[6] Rotation b/w matrices: http://www.boris-belousov.net/2016/12/01/quat-dist/#using-rotation-matrices
[7] Rodrigues' rotation formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
"""
import copy

from gymnasium.utils import seeding

import gym_art.quadrotor_multi.get_state as get_state
import gym_art.quadrotor_multi.quadrotor_randomization as quad_rand
from gym_art.quadrotor_multi.quadrotor_control import *
from gym_art.quadrotor_multi.quadrotor_dynamics import QuadrotorDynamics
from gym_art.quadrotor_multi.sensor_noise import SensorNoise

GRAV = 9.81  # default gravitational constant

def rotation_matrix_to_euler_angles(R):
    """
    将旋转矩阵转换为欧拉角（roll、pitch、yaw）
    参数：
        R：3x3的旋转矩阵
    返回：
        欧拉角数组 [roll, pitch, yaw]，单位为弧度
    """
    assert R.shape == (3, 3)
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        # 在奇异情况下，近似处理
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    return np.array([roll, pitch, yaw])
# reasonable reward function for hovering at a goal and not flying too high
def compute_reward_weighted(dynamics, goal, action, dt, rew_coeff, action_prev, on_floor=False):
    
    # Distance to the goal
    dist = np.linalg.norm(goal - dynamics.pos)
    cost_pos_raw = dist*dt
    cost_pos = rew_coeff["pos"] * cost_pos_raw

    cost_time=rew_coeff["time"]*dt

    # Penalize amount of control effort
    cost_effort_raw = np.linalg.norm(action)*dt
    cost_effort = rew_coeff["effort"] * cost_effort_raw

    # Loss orientation
    if on_floor:
        cost_orient_raw = 1.0*dt
    else:
        cost_orient_raw = -dynamics.rot[2, 2]*dt

    cost_orient = rew_coeff["orient"] * cost_orient_raw

    # Loss for constant uncontrolled rotation around vertical axis
    cost_spin_raw = ((dynamics.omega[0] ** 2 + dynamics.omega[1] ** 2 + dynamics.omega[2] ** 2) ** 0.5)*dt
    cost_spin = rew_coeff["spin"] * cost_spin_raw

    # Loss crash for staying on the floor
    cost_crash_raw = float(on_floor)*dt
    cost_crash = rew_coeff["crash"] * cost_crash_raw

     # 姿态角限制
    euler_angles = rotation_matrix_to_euler_angles(dynamics.rot)
    roll_angle = euler_angles[0]
    pitch_angle = euler_angles[1]
    max_angle = np.radians(30)  # 最大允许姿态角

    roll_exceed = max(abs(roll_angle) - max_angle, 0)
    pitch_exceed = max(abs(pitch_angle) - max_angle, 0)
    cost_attitude_raw = (roll_exceed + pitch_exceed)*dt*57.3# hudu to jiaodu
    cost_attitude = rew_coeff["attitude"] * cost_attitude_raw

    # 计算总奖励
    reward = -np.sum([
        cost_pos,
        cost_effort,
        cost_crash,
        cost_orient,
        cost_spin,
        cost_attitude,  # 新增的姿态角惩罚项
        cost_time, #work as soon as possible
    ])

    rew_info = {
        'rew_pos': -cost_pos,
        'rew_action': -cost_effort,
        'rew_crash': -cost_crash,
        "rew_orient": -cost_orient,
        "rew_spin": -cost_spin,
        "rew_attitude": -cost_attitude,
        "rew_time": -cost_time,
        "rewraw_attitude": -cost_attitude_raw,
        'rewraw_pos': -cost_pos_raw,
        'rewraw_action': -cost_effort_raw,
        'rewraw_crash': -cost_crash_raw,
        "rewraw_orient": -cost_orient_raw,
        "rewraw_spin": -cost_spin_raw,
    }



    if np.isnan(reward) or not np.isfinite(reward):
        for key, value in locals().items():
            print('%s: %s \n' % (key, str(value)))
        raise ValueError('QuadEnv: reward is Nan')

    return reward, rew_info


# ENV Gym environment for quadrotor seeking the origin with no obstacles and full state observations. NOTES: - room
# size of the env and init state distribution are not the same ! It is done for the reason of having static (and
# preferably short) episode length, since for some distance it would be impossible to reach the goal
class QuadrotorSingle:
    def __init__(self, dynamics_params="DefaultQuad", dynamics_change=None,
                 dynamics_randomize_every=None, dyn_sampler_1=None, dyn_sampler_2=None,
                  zero_action_middle=True, dim_mode='3D', tf_control=False, sim_freq=200.,
                 sim_steps=2, obs_repr="xyz_vxyz_R_omega_goal",  room_dims=(10.0, 10.0, 10.0),
                 init_random_state=False, sense_noise=None, verbose=False, gravity=GRAV,
                 t2w_std=0.005, t2t_std=0.0005, excite=False, dynamics_simplification=False, use_numba=False,
                 neighbor_obs_type='none', num_agents=1, num_use_neighbor_obs=0, use_obstacles=False,control_input='pos', if_discrete=False):
        np.seterr(under='ignore')
        """
        Args:
            dynamics_params: [str or dict] loading dynamics params by name or by providing a dictionary. 
                If "random": dynamics will be randomized completely (see sample_dyn_parameters() )
                If dynamics_randomize_every is None: it will be randomized only once at the beginning.
                One can randomize dynamics during the end of any episode using resample_dynamics()
                WARNING: randomization during an episode is not supported yet. Randomize ONLY before calling reset().
            dynamics_change: [dict] update to dynamics parameters relative to dynamics_params provided
            
            dynamics_randomize_every: [int] how often (trajectories) perform randomization dynamics_sampler_1: [dict] 
            the first sampler to be applied. Dict must contain type (see quadrotor_randomization) and whatever params 
            requires 
            dynamics_sampler_2: [dict] the second sampler to be applied. Convenient if you need to 
                fix some params after sampling.
            

            zero_action_middle: [bool] meaning that control will be [-1 .. 1] rather than [0 .. 1]
            dim_mode: [str] Dimensionality of the env. 
            Options: 1D(just a vertical stabilization), 2D(vertical plane), 3D(normal)
            tf_control: [bool] creates Mellinger controller using TensorFlow
            sim_freq (float): frequency of simulation
            sim_steps: [int] how many simulation steps for each control step
            obs_repr: [str] options: xyz_vxyz_rot_omega, xyz_vxyz_quat_omega
            room_size: [int] env room size. Not the same as the initialization box to allow shorter episodes
            init_random_state: [bool] use random state initialization or horizontal initialization with 0 velocities
            rew_coeff: [dict] weights for different reward components (see compute_weighted_reward() function)
            sens_noise (dict or str): sensor noise parameters. If None - no noise. If "default" then the default params 
                are loaded. Otherwise one can provide specific params.
            excite: [bool] change the set point at the fixed frequency to perturb the quad
        """
        # Numba Speed Up
        self.use_numba = use_numba

        # Room
        self.room_length = room_dims[0]
        self.room_width = room_dims[1]
        self.room_height = room_dims[2]
        self.room_box = np.array([[-self.room_length / 2., -self.room_width / 2, 0.],
                                  [self.room_length / 2., self.room_width / 2., self.room_height]])

        self.init_random_state = init_random_state
        

        # Preset parameters
        self.obs_repr = obs_repr
        self.rew_coeff = None
        # EPISODE PARAMS
        self.sim_steps = sim_steps
        self.dt = 1.0 / sim_freq
        self.tick = 0
        self.control_freq = sim_freq / sim_steps
        self.traj_count = 0

        # Self dynamics
        self.control_input = control_input
        self.if_discrete = if_discrete
        self.dim_mode = dim_mode
        self.zero_action_middle = zero_action_middle
        self.tf_control = tf_control
        self.dynamics_randomize_every = dynamics_randomize_every
        self.verbose = verbose
        self.gravity = gravity
        self.update_sense_noise(sense_noise=sense_noise)
        self.t2w_std = t2w_std
        self.t2w_min = 1.5
        self.t2w_max = 10.0

        self.t2t_std = t2t_std
        self.t2t_min = 0.005
        self.t2t_max = 1.0
        self.excite = excite
        self.dynamics_simplification = dynamics_simplification
        self.max_init_vel = 1.  # m/s
        self.max_init_omega = 2 * np.pi  # rad/s

        # DYNAMICS (and randomization)
        # Could be dynamics of a specific quad or a random dynamics (i.e. randomquad)
        self.dyn_base_sampler = getattr(quad_rand, dynamics_params)()
        self.dynamics_change = copy.deepcopy(dynamics_change)
        self.dynamics_params = self.dyn_base_sampler.sample()
        # Now, updating if we are providing modifications
        if self.dynamics_change is not None:
            dict_update_existing(self.dynamics_params, self.dynamics_change)

        self.dyn_sampler_1 = dyn_sampler_1
        if dyn_sampler_1 is not None:
            sampler_type = dyn_sampler_1["class"]
            self.dyn_sampler_1_params = copy.deepcopy(dyn_sampler_1)
            del self.dyn_sampler_1_params["class"]
            self.dyn_sampler_1 = getattr(quad_rand, sampler_type)(params=self.dynamics_params,
                                                                  **self.dyn_sampler_1_params)

        self.dyn_sampler_2 = dyn_sampler_2
        if dyn_sampler_2 is not None:
            sampler_type = dyn_sampler_2["class"]
            self.dyn_sampler_2_params = copy.deepcopy(dyn_sampler_2)
            del self.dyn_sampler_2_params["class"]
            self.dyn_sampler_2 = getattr(quad_rand, sampler_type)(params=self.dynamics_params,
                                                                  **self.dyn_sampler_2_params)

        # Updating dynamics
        self.action_space = None
        self.resample_dynamics()

        # Self info
        self.state_vector = self.state_vector = getattr(get_state, "state_" + self.obs_repr)
        if use_obstacles:
            self.box = 0.1
        else:
            self.box = 2.0
        self.box_scale = 1.0
        self.goal = None
        self.spawn_point = None

        # Neighbor info
        self.num_agents = num_agents
        self.neighbor_obs_type = neighbor_obs_type
        self.num_use_neighbor_obs = num_use_neighbor_obs

        # Obstacles info
        self.use_obstacles = use_obstacles

        # Make observation space
        self.observation_space = self.make_observation_space()

        self._seed()

    def update_sense_noise(self, sense_noise):
        if isinstance(sense_noise, dict):
            self.sense_noise = SensorNoise(**sense_noise)
        elif isinstance(sense_noise, str):
            if sense_noise == "default":
                self.sense_noise = SensorNoise(bypass=False, use_numba=self.use_numba)
            else:
                ValueError("ERROR: QuadEnv: sense_noise parameter is of unknown type: " + str(sense_noise))
        elif sense_noise is None:
            self.sense_noise = SensorNoise(bypass=True)
        else:
            raise ValueError("ERROR: QuadEnv: sense_noise parameter is of unknown type: " + str(sense_noise))

    def update_dynamics(self, dynamics_params):
        # DYNAMICS
        # Then loading the dynamics
        self.dynamics_params = dynamics_params
        self.dynamics = QuadrotorDynamics(model_params=dynamics_params,
                                          dynamics_steps_num=self.sim_steps, room_box=self.room_box,
                                          dim_mode=self.dim_mode, gravity=self.gravity,
                                          dynamics_simplification=self.dynamics_simplification,
                                          use_numba=self.use_numba, dt=self.dt)

        # CONTROL
        self.update_controller()
        

        # ACTIONS
        self.action_space = self.create_action_space()

        # STATE VECTOR FUNCTION
        self.state_vector = getattr(get_state, "state_" + self.obs_repr)


    def update_controller(self):
        # 根据控制输入类型和离散动作空间标志，选择合适的控制器
        if self.control_input == 'pos':
            self.controller = NonlinearPositionController(self.dynamics)
        elif self.control_input == 'vel':
            self.controller = VelocityControl(self.dynamics)
        elif self.control_input == 'raw':
            if self.dim_mode == '1D':  # Z axis only
                self.controller = VerticalControl(self.dynamics, zero_action_middle=self.zero_action_middle)
            elif self.dim_mode == '2D':  # X and Z axes only
                self.controller = VertPlaneControl(self.dynamics, zero_action_middle=self.zero_action_middle)
            elif self.dim_mode == '3D':
                self.controller = RawControl(self.dynamics, zero_action_middle=self.zero_action_middle)
            else:
                raise ValueError("Unsupported control dim_mode type")
        else:
            raise ValueError("Unsupported control input type")


    def create_action_space(self):
        if self.if_discrete:
            # 定义离散动作空间
            if self.control_input == 'vel':
                return spaces.Discrete(27)  # 3^3([3 channel][3 action:stay\up\down])
            elif self.control_input == 'pos':
                return spaces.Discrete(27)  # # 3^3([3 channel][3 action:stay\up\down])
            elif self.control_input =='raw':
                return spaces.Discrete(81)  # 3^4([4 channel][3 action:stay\up\down])
            else:
                # 对于其他控制模式，可以根据需要定义
                raise NotImplementedError("Discrete action space not implemented for this control input")
        else:
            # 使用控制器定义的连续动作空间
            return self.controller.action_space(self.dynamics)

    def discrete_action_to_continuous(self, action):
        # 将离散动作映射到连续动作
        if self.control_input == 'vel':
            delta_v = 0.001  # 速度增量，可根据需要调整
            if self.continuous_action is None:
                # 初始化为当前速度
                self.continuous_action = self.dynamics.vel.copy()

            # Decode the action into individual channel changes
            dx, dy, dz = self.decode_action(action, 3)
            self.continuous_action += np.array([dx * delta_v, dy * delta_v, dz * delta_v])

            # 确保连续动作在限制范围内
            self.continuous_action = np.clip(self.continuous_action, self.controller.low, self.controller.high)
            # print("target_pos: ",self.continuous_action)
            return self.continuous_action

        elif self.control_input == 'pos':
            delta_p = 0.1  # 位置增量，可根据需要调整
            if self.continuous_action is None:
                # 初始化为当前位置
                self.continuous_action = self.dynamics.pos.copy()

            # Decode the action into individual channel changes
            dx, dy, dz = self.decode_action(action, 3)
            self.continuous_action += np.array([dx * delta_p, dy * delta_p, dz * delta_p])

            # 确保连续动作在限制范围内
            self.continuous_action = np.clip(self.continuous_action, self.controller.low, self.controller.high)
            return self.continuous_action

        elif self.control_input == 'raw':
            delta_thrust = 0.05  # 推力增量，可根据需要调整
            if self.continuous_action is None:
                # 初始化为当前电机命令
                self.continuous_action = self.controller.action.copy() if self.controller.action is not None else np.ones(4) * 0.5

            # Decode the action into individual channel changes
            d1, d2, d3, d4 = self.decode_action(action, 4)
            self.continuous_action += np.array([d1 * delta_thrust, d2 * delta_thrust, d3 * delta_thrust, d4 * delta_thrust])

            # 确保连续动作在限制范围内
            self.continuous_action = np.clip(self.continuous_action, self.controller.low, self.controller.high)
            return self.continuous_action
        else:
            raise ValueError("Unsupported control input type")

    def decode_action(self, action, num_channels):
        """
        Decode a single integer action into individual channel changes.
        :param action: The integer action to decode.
        :param num_channels: The number of channels to decode.
        :return: A tuple of channel changes.
        """
        channel_changes = []
        for i in range(num_channels):
            channel_changes.append(action % 3 - 1)  # -1 for decrease, 0 for no change, 1 for increase
            action //= 3
        return tuple(channel_changes)


    def make_observation_space(self):
        room_range = self.room_box[1] - self.room_box[0]
        self.obs_space_low_high = {
            "xyz": [-room_range, room_range],
            "xyzr": [-room_range, room_range],
            "vxyz": [-self.dynamics.vxyz_max * np.ones(3), self.dynamics.vxyz_max * np.ones(3)],
            "vxyzr": [-self.dynamics.vxyz_max * np.ones(3), self.dynamics.vxyz_max * np.ones(3)],
            "acc": [-self.dynamics.acc_max * np.ones(3), self.dynamics.acc_max * np.ones(3)],
            "R": [-np.ones(9), np.ones(9)],
            "omega": [-self.dynamics.omega_max * np.ones(3), self.dynamics.omega_max * np.ones(3)],
            "t2w": [0. * np.ones(1), 5. * np.ones(1)],
            "t2t": [0. * np.ones(1), 1. * np.ones(1)],
            "h": [0. * np.ones(1), self.room_box[1][2] * np.ones(1)],
            "act": [np.zeros(4), np.ones(4)],
            "quat": [-np.ones(4), np.ones(4)],
            "euler": [-np.pi * np.ones(3), np.pi * np.ones(3)],
            "rxyz": [-room_range, room_range],  # rxyz stands for relative pos between quadrotors
            "rvxyz": [-2.0 * self.dynamics.vxyz_max * np.ones(3), 2.0 * self.dynamics.vxyz_max * np.ones(3)],
            # rvxyz stands for relative velocity between quadrotors
            "roxyz": [-room_range, room_range],  # roxyz stands for relative pos between quadrotor and obstacle
            "rovxyz": [-20.0 * np.ones(3), 20.0 * np.ones(3)],
            # rovxyz stands for relative velocity between quadrotor and obstacle
            "osize": [np.zeros(3), 20.0 * np.ones(3)],  # obstacle size, [[0., 0., 0.], [20., 20., 20.]]
            "otype": [np.zeros(1), 20.0 * np.ones(1)],
            # obstacle type, [[0.], [20.]], which means we can support 21 types of obstacles
            "goal": [-room_range, room_range],
            "wall": [np.zeros(6), 5.0 * np.ones(6)],
            "floor": [np.zeros(1), self.room_box[1][2] * np.ones(1)],
            "octmap": [-10 * np.ones(9), 10 * np.ones(9)],
        }
        self.obs_comp_names = list(self.obs_space_low_high.keys())
        self.obs_comp_sizes = [self.obs_space_low_high[name][1].size for name in self.obs_comp_names]

        obs_comps = self.obs_repr.split("_")
        if self.neighbor_obs_type == 'pos_vel' and self.num_use_neighbor_obs > 0:
            obs_comps = obs_comps + (['rxyz'] + ['rvxyz']) * self.num_use_neighbor_obs

        if self.use_obstacles:
            obs_comps = obs_comps + ["octmap"]

        print("Observation components:", obs_comps)
        obs_low, obs_high = [], []
        for comp in obs_comps:
            obs_low.append(self.obs_space_low_high[comp][0])
            obs_high.append(self.obs_space_low_high[comp][1])
        obs_low = np.concatenate(obs_low)
        obs_high = np.concatenate(obs_high)

        self.obs_comp_sizes_dict, self.obs_space_comp_indx, self.obs_comp_end = {}, {}, []
        end_indx = 0
        for obs_i, obs_name in enumerate(self.obs_comp_names):
            end_indx += self.obs_comp_sizes[obs_i]
            self.obs_comp_sizes_dict[obs_name] = self.obs_comp_sizes[obs_i]
            self.obs_space_comp_indx[obs_name] = obs_i
            self.obs_comp_end.append(end_indx)

        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        return self.observation_space

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        self.actions[1] = copy.deepcopy(self.actions[0])
        self.actions[0] = copy.deepcopy(action)

        self.controller.step_func(dynamics=self.dynamics, action=action,  dt=self.dt, observation=None)

        reward, rew_info = compute_reward_weighted(
            dynamics=self.dynamics, goal=self.goal, action=action, dt=self.dt, 
            rew_coeff=self.rew_coeff, action_prev=self.actions[1], on_floor=self.dynamics.on_floor)

        self.tick += 1
        done = False
        sv = self.state_vector(self)
        self.traj_count += int(done)
        env_info = {
            'rewards': rew_info
        }

        return sv, reward, done, env_info

    def resample_dynamics(self):
        """
        Allows manual dynamics resampling when needed.
        WARNING: 
            - Randomization dyring an episode is not supported
            - MUST call reset() after this function
        """
        # Getting base parameters (could also be random parameters)
        self.dynamics_params = self.dyn_base_sampler.sample()

        # Now, updating if we are providing modifications
        if self.dynamics_change is not None:
            dict_update_existing(self.dynamics_params, self.dynamics_change)

        # Applying sampler 1
        if self.dyn_sampler_1 is not None:
            self.dynamics_params = self.dyn_sampler_1.sample(self.dynamics_params)

        # Applying sampler 2
        if self.dyn_sampler_2 is not None:
            self.dynamics_params = self.dyn_sampler_2.sample(self.dynamics_params)

        # Checking that quad params make sense
        quad_rand.check_quad_param_limits(self.dynamics_params)

        # Updating params
        self.update_dynamics(dynamics_params=self.dynamics_params)

    def _reset(self):
        # DYNAMICS RANDOMIZATION AND UPDATE
        if self.dynamics_randomize_every is not None and (self.traj_count + 1) % self.dynamics_randomize_every == 0:
            self.resample_dynamics()

        if self.box < 10:
            self.box = self.box * self.box_scale
        x, y, z = self.np_random.uniform(-self.box, self.box, size=(3,)) + self.spawn_point

        if self.dim_mode == '1D':
            x, y = self.goal[0], self.goal[1]
        elif self.dim_mode == '2D':
            y = self.goal[1]
        # Since being near the groud means crash we have to start above
        if z < 0.75:
            z = 0.75
        pos = npa(x, y, z)

        # INIT STATE
        # Initializing rotation and velocities
        if self.init_random_state:
            if self.dim_mode == '1D':
                omega, rotation = np.zeros(3, dtype=np.float64), np.eye(3)
                vel = np.array([0, 0, self.max_init_vel * np.random.rand()])
            elif self.dim_mode == '2D':
                omega = npa(0, self.max_init_omega * np.random.rand(), 0)
                vel = self.max_init_vel * np.random.rand(3)
                vel[1] = 0.
                theta = np.pi * np.random.rand()
                c, s = np.cos(theta), np.sin(theta)
                rotation = np.array(((c, 0, -s), (0, 1, 0), (s, 0, c)))
            else:
                # It already sets the state internally
                _, vel, rotation, omega = self.dynamics.random_state(
                    box=(self.room_length, self.room_width, self.room_height), vel_max=self.max_init_vel,
                    omega_max=self.max_init_omega
                )
        else:
            # INIT HORIZONTALLY WITH 0 VEL and OMEGA
            vel, omega = np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

            if self.dim_mode == '1D' or self.dim_mode == '2D':
                rotation = np.eye(3)
            else:
                # make sure we're sort of pointing towards goal (for mellinger controller)
                rotation = randyaw()
                while np.dot(rotation[:, 0], to_xyhat(-pos)) < 0.5:
                    rotation = randyaw()

        self.init_state = [pos, vel, rotation, omega]
        self.dynamics.set_state(pos, vel, rotation, omega)
        self.dynamics.reset()
        self.dynamics.on_floor = False
        self.dynamics.crashed_floor = self.dynamics.crashed_wall = self.dynamics.crashed_ceiling = False

        # Reseting some internal state (counters, etc)
        self.tick = 0
        self.actions = [np.zeros([4, ]), np.zeros([4, ])]

        state = self.state_vector(self)

        # COntrol param
        self.continuous_action=None #target pos\vel
        return state

    def reset(self):
        return self._reset()

    def render(self, **kwargs):
        """This class is only meant to be used as a component of QuadMultiEnv."""
        raise NotImplementedError()

    def step(self, action):
        # 处理离散动作空间
        if self.if_discrete:
            action = self.discrete_action_to_continuous(action)
        return self._step(action)
