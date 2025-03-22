import time
from unittest import TestCase
import numpy as np

from gym_art.quadrotor_multi.quad_experience_replay import ExperienceReplayWrapper
from gym_art.quadrotor_multi.quadrotor_multi import QuadrotorEnvMulti


def create_env(num_agents, use_numba=False, use_replay_buffer=False, episode_duration=7, local_obs=-1):
    quad = 'Crazyflie'
    dyn_randomize_every = dyn_randomization_ratio = None

    episode_duration = episode_duration  # seconds

    raw_control = raw_control_zero_middle = True

    sampler_1 = None
    if dyn_randomization_ratio is not None:
        sampler_1 = dict(type="RelativeSampler", noise_ratio=dyn_randomization_ratio, sampler="normal")

    sense_noise = 'default'

    dynamics_change = dict(noise=dict(thrust_noise_ratio=0.05), damp=dict(vel=0, omega_quadratic=0))

    # 新增的缺少参数及其默认值
    rew_coeff = None  # 可以设置为None，类内部会使用默认值
    obs_repr = 'xyz_vxyz_R_omega'  # 观测表示方式
    neighbor_visible_num = -1  # -1表示可见所有邻居
    neighbor_obs_type = 'pos_vel'  # 邻居观测类型
    collision_hitbox_radius = 1.0  # 碰撞半径
    collision_falloff_radius = 1.2  # 碰撞衰减半径
    use_obstacles = False  # 是否使用障碍物
    obst_density = 0.0  # 障碍物密度
    obst_size = [0.15, 0.15, 0.75]  # 障碍物尺寸
    obst_spawn_area = [5.0, 5.0]  # 障碍物生成区域
    use_downwash = False  # 是否使用下洗效应
    quads_mode = 'static_same_goal'  # 无人机模式，可以根据需要修改
    room_dims = (10.0, 10.0, 10.0)  # 房间尺寸
    quads_view_mode = ['topdown']  # 视角模式
    quads_render = False  # 是否渲染
    render_mode="rgb_array"#'human' or "rgb_array"


    env = QuadrotorEnvMulti(
        num_agents=num_agents,
        ep_time=episode_duration,
        rew_coeff=rew_coeff,
        obs_repr=obs_repr,

        neighbor_visible_num=neighbor_visible_num,
        neighbor_obs_type=neighbor_obs_type,
        collision_hitbox_radius=collision_hitbox_radius,
        collision_falloff_radius=collision_falloff_radius,

        use_obstacles=use_obstacles,
        obst_density=obst_density,
        obst_size=obst_size,
        obst_spawn_area=obst_spawn_area,

        use_downwash=use_downwash,
        use_numba=use_numba,
        quads_mode=quads_mode,
        room_dims=room_dims,
        use_replay_buffer=use_replay_buffer,
        quads_view_mode=quads_view_mode,
        quads_render=quads_render,

        dynamics_params=quad,
        raw_control=raw_control,
        raw_control_zero_middle=raw_control_zero_middle,
        dynamics_randomize_every=dyn_randomize_every,
        dynamics_change=dynamics_change,
        dyn_sampler_1=sampler_1,
        sense_noise=sense_noise,
        init_random_state=True,
        render_mode=render_mode
    )
    """rew_coeff: 奖励系数，可以设置为None，因为类内部有默认值。
        obs_repr: 观测的表示方式，根据代码中的QUADS_OBS_REPR定义，这里选择'xyz_vxyz_R_omega'。
        neighbor_visible_num: 可见的邻居数量，-1表示所有邻居都可见。
        neighbor_obs_type: 邻居的观测类型，设为'pos_vel'表示位置和速度。
        collision_hitbox_radius: 碰撞检测的半径倍数，设置为1.0。
        collision_falloff_radius: 碰撞惩罚衰减的半径倍数，设置为1.2。
        use_obstacles: 是否使用障碍物，测试时可以设置为False。
        obst_density: 障碍物密度，设置为0.0表示没有障碍物。
        obst_size: 障碍物的尺寸，单位为米。
        obst_spawn_area: 障碍物生成区域的尺寸，单位为米。
        use_downwash: 是否使用下洗效应，测试时可以设置为False。
        quads_mode: 无人机的模式，测试时可以设置为'static_same_goal'。
        room_dims: 房间的尺寸，单位为米。
        quads_view_mode: 渲染时的视角模式，设置为['top_down']（俯视图）。
        'chase' 'side':'global':'topdown''topdownfollow''corner':
        quads_render: 是否启用渲染，测试时可以设置为False。"""

    return env


class TestMultiEnv(TestCase):
    def test_basic(self):
        num_agents = 2
        env = create_env(num_agents, use_numba=False)

        self.assertTrue(hasattr(env, 'num_agents'))
        self.assertEqual(env.num_agents, num_agents)

        obs = env.reset()
        self.assertIsNotNone(obs)

        for i in range(100):
            env.action_space.sample()
            print(env.action_space.sample())
            obs, rewards, dones, infos = env.step([env.action_space.sample() for i in range(num_agents)])
            print(f"got obs {obs}")
            print(f"got rewards {rewards}")
            print(f"got dones {dones}")
            print(f"got infos {infos}")
            try:
                self.assertIsInstance(obs, list)
            except:
                self.assertIsInstance(obs, np.ndarray)

            self.assertIsInstance(rewards, list)
            self.assertIsInstance(dones, list)
            self.assertIsInstance(infos, list)

        env.close()

    def test_render(self):
        num_agents = 16
        env = create_env(num_agents, use_numba=False, local_obs=8)
        env.render_speed = 1.0

        env.reset()
        time.sleep(0.1)

        num_steps = 0
        render_n_frames = 100

        render_start = None
        while num_steps < render_n_frames:
            obs, rewards, dones, infos = env.step([env.action_space.sample() for _ in range(num_agents)])
            num_steps += 1
            # print('Rewards: ', rewards, "\nCollisions: \n", env.collisions, "\nDistances: \n", env.dist)
            env.render()

            if num_steps <= 1:
                render_start = time.time()

        render_took = time.time() - render_start
        print(f"Rendering of {render_n_frames} frames took {render_took:.3f} sec")

        env.close()

    def test_local_info(self):
        num_agents = 16
        env = create_env(num_agents, use_numba=False, local_obs=8)

        env.reset()

        for i in range(100):
            obs, rewards, dones, infos = env.step([env.action_space.sample() for i in range(num_agents)])

        env.close()


class TestReplayBuffer(TestCase):
    def test_replay(self):
        num_agents = 16
        replay_buffer_sample_prob = 1.0
        env = create_env(num_agents, use_numba=False, use_replay_buffer=replay_buffer_sample_prob > 0, episode_duration=5)
        env.render_speed = 1.0
        env = ExperienceReplayWrapper(env, replay_buffer_sample_prob=replay_buffer_sample_prob)

        env.reset()
        time.sleep(0.01)

        num_steps = 0
        render_n_frames = 150

        while num_steps < render_n_frames:
            obs, rewards, dones, infos = env.step([env.action_space.sample() for _ in range(num_agents)])
            num_steps += 1
            # print('Rewards: ', rewards, "\nCollisions: \n", env.collisions, "\nDistances: \n", env.dist)
            env.render()
            # this env self-resets

        env.close()
