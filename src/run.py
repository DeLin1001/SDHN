import datetime
import os
from os.path import dirname, abspath
import pprint
import shutil
import time
import threading
from types import SimpleNamespace as SN

import torch as th

from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from utils.general_reward_support import test_alg_config_supports_reward
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from gym import spaces as gym_spaces
from gymnasium import spaces as gymn_spaces
from math import ceil
import numpy as np

def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    assert test_alg_config_supports_reward(
        args
    ), "The specified algorithm does not support the general reward setup. Please choose a different algorithm or set `common_reward=True`."

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    # unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    try:
        map_name = _config["env_args"]["map_name"]
    except:
        map_name = _config["env_args"]["key"]
    unique_token = (
        f"{_config['name']}_seed{_config['seed']}_{map_name}_{datetime.datetime.now()}"
    )

    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(
            dirname(dirname(abspath(__file__))), "results", "tb_logs"
        )
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    if args.use_wandb:
        logger.setup_wandb(
            _config, args.wandb_team, args.wandb_project, args.wandb_mode
        )

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Finish logging
    logger.finish()

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    # os._exit(os.EX_OK)


def evaluate_sequential(args, runner):
    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, logger):
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)
    if "quadrotor_multi"  in args.env and args.env_args["if_discrete"]!=True:  # need to rewrite
        env_info = runner.get_env_info()
        args.n_agents = env_info["n_agents"]
        args.n_actions = env_info["n_actions"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        args.action_spaces = env_info["action_spaces"]
        args.actions_dtype = env_info["actions_dtype"]
        args.normalise_actions = env_info.get("normalise_actions", False) # if true, action vectors need to sum to one

        # 判断动作空间的类型，使用对应的 spaces 模块
        first_action_space = args.action_spaces[0]
        if isinstance(first_action_space, gymn_spaces.Space):
            # 使用 gymnasium 的 spaces
            spaces_module = gymn_spaces
        elif isinstance(first_action_space, gym_spaces.Space):
            # 使用 gym 的 spaces
            spaces_module = gym_spaces
        else:
            raise NotImplementedError("Unknown action space type: {}".format(type(first_action_space)))

        # 初始化张量，避免使用过时的构造函数
        device = th.device("cuda" if args.use_cuda else "cpu")
        mult_coef_tensor = th.zeros(args.n_agents, args.n_actions, device=device)
        action_min_tensor = th.zeros(args.n_agents, args.n_actions, device=device)

        # 根据动作空间的类型，处理连续或离散动作空间
        if all([isinstance(act_space, spaces_module.Box) for act_space in args.action_spaces]):
            for _aid in range(args.n_agents):
                for _actid in range(args.action_spaces[_aid].shape[0]):
                    _action_min = args.action_spaces[_aid].low[_actid]
                    _action_max = args.action_spaces[_aid].high[_actid]
                    mult_coef_tensor[_aid, _actid] = (_action_max - _action_min).item()
                    action_min_tensor[_aid, _actid] = (_action_min).item()
        elif all([isinstance(act_space, spaces_module.Tuple) for act_space in args.action_spaces]):
            for _aid in range(args.n_agents):
                # 处理 Tuple 类型的动作空间
                total_dims = 0
                for space in args.action_spaces[_aid].spaces:
                    for _actid in range(space.shape[0]):
                        _action_min = space.low[_actid]
                        _action_max = space.high[_actid]
                        mult_coef_tensor[_aid, total_dims + _actid] = (_action_max - _action_min).item()
                        action_min_tensor[_aid, total_dims + _actid] = (_action_min).item()
                    total_dims += space.shape[0]
        else:
            raise NotImplementedError("Action spaces are of mixed types or unknown types.")

        args.actions2unit_coef = mult_coef_tensor
        args.actions2unit_coef_cpu = mult_coef_tensor.cpu()
        args.actions2unit_coef_numpy = mult_coef_tensor.cpu().numpy()
        args.actions_min = action_min_tensor
        args.actions_min_cpu = action_min_tensor.cpu()
        args.actions_min_numpy = action_min_tensor.cpu().numpy()

        def actions_to_unit_box(actions):
            if isinstance(actions, np.ndarray):
                return args.actions2unit_coef_numpy * actions + args.actions_min_numpy
            elif actions.is_cuda:
                return args.actions2unit_coef * actions + args.actions_min
            else:
                return args.args.actions2unit_coef_cpu  * actions + args.actions_min_cpu

        def actions_from_unit_box(actions):
            if isinstance(actions, np.ndarray):
                return th.div((actions - args.actions_min_numpy), args.actions2unit_coef_numpy)
            elif actions.is_cuda:
                return th.div((actions - args.actions_min), args.actions2unit_coef)
            else:
                return th.div((actions - args.actions_min_cpu), args.actions2unit_coef_cpu)

        # make conversion functions globally available
        args.actions2unit = actions_to_unit_box
        args.unit2actions = actions_from_unit_box
        print("Action spaces:", args.action_spaces)
        print("Type of action_spaces[0]:", type(args.action_spaces[0]))
        actions_vshape=None
        action_dtype = th.long if not args.actions_dtype == np.float32 else th.float



        # 确定 actions_vshape
        if all([isinstance(act_space, spaces_module.Box) for act_space in args.action_spaces]):
            if args.actions_dtype != np.float32:
                actions_vshape = 1
            else:
                actions_vshape = max([act_space.shape[0] for act_space in args.action_spaces])
        elif all([isinstance(act_space, spaces_module.Tuple) for act_space in args.action_spaces]):
            if args.actions_dtype != np.float32:
                actions_vshape = 1
            else:
                actions_vshape = max([sum([space.shape[0] for space in act_space.spaces]) for act_space in args.action_spaces])
        else:
            # 处理其他类型的动作空间，例如 Discrete
            first_act_space = args.action_spaces[0]
            if hasattr(first_act_space, 'n'):
                actions_vshape = 1
            elif hasattr(first_act_space, 'shape'):
                actions_vshape = first_act_space.shape[0]
            else:
                raise NotImplementedError("Unknown action space type: {}".format(type(first_act_space)))

        print("actions_vshape:", actions_vshape)
        # Default/Base scheme
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (actions_vshape,), "group": "agents", "dtype": action_dtype},
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        groups = {
            "agents": args.n_agents
        }

        if not args.actions_dtype == np.float32:
            preprocess = {
                "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
            }
        else:
            preprocess = {}
    else:
        # Set up schemes and groups here
        env_info = runner.get_env_info()
        args.n_agents = env_info["n_agents"]
        args.n_actions = env_info["n_actions"]
        args.state_shape = env_info["state_shape"]
        # print(env_info)

        # Default/Base scheme
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {
                "vshape": (env_info["n_actions"],),
                "group": "agents",
                "dtype": th.int,
            },
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        # For individual rewards in gymmai reward is of shape (1, n_agents)
        if args.common_reward:
            scheme["reward"] = {"vshape": (1,)}
        else:
            scheme["reward"] = {"vshape": (args.n_agents,)}
        groups = {"agents": args.n_agents}
        preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(args.checkpoint_path)
            )
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            runner.log_train_stats_t = runner.t_env
            evaluate_sequential(args, runner)
            logger.log_stat("episode", runner.t_env, runner.t_env)
            logger.print_recent_stats()
            logger.console_logger.info("Finished Evaluation")
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:
        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)
            # learner.train(episode_sample,max_ep_t, runner.t_env, episode)   #GACG

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max)
            )
            # print(th.cuda.memory_stats())
            # print(th.cuda.memory_summary())
            th.cuda.empty_cache()
            # print(th.cuda.memory_stats())
            # print(th.cuda.memory_summary())
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max),
                    time_str(time.time() - start_time),
                )
            )
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval
            or model_save_time == 0
        ):
            model_save_time = runner.t_env
            save_path = os.path.join(
                args.local_results_path, "models", args.unique_token, str(runner.t_env)
            )
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

            if args.use_wandb and args.wandb_save_model:
                wandb_save_dir = os.path.join(
                    logger.wandb.dir, "models", args.unique_token, str(runner.t_env)
                )
                os.makedirs(wandb_save_dir, exist_ok=True)
                for f in os.listdir(save_path):
                    shutil.copyfile(
                        os.path.join(save_path, f), os.path.join(wandb_save_dir, f)
                    )

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config
