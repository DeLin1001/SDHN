# Imports
import numpy as np
from numpy.linalg import norm
from gymnasium import spaces
from gym_art.quadrotor_multi.quad_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F

GRAV = 9.81  # Gravity acceleration in m/s^2

# ShiftedMotorControl class
class ShiftedMotorControl(object):
    def __init__(self, dynamics):
        pass  # Initialization function; currently does nothing

    def action_space(self, dynamics):
        # Define action space so that zero action corresponds to hovering
        low = -1.0 * np.ones(4)  # Minimum values are -1
        high = (dynamics.thrust_to_weight - 1.0) * np.ones(4)  # Maximum values depend on thrust-to-weight ratio
        return spaces.Box(low, high, dtype=np.float32)

    def step(self, dynamics, action, dt):
        # Adjust action to fit the thrust-to-weight ratio
        action = (action + 1.0) / dynamics.thrust_to_weight
        action = np.clip(action, 0, 1)  # Clip actions to between 0 and 1
        dynamics.step(action, dt)  # Update dynamics

# RawControl class
class RawControl(object):
    def __init__(self, dynamics, zero_action_middle=True):
        self.zero_action_middle = zero_action_middle
        self.action = None
        self.step_func = self.step  # Bind step function
        self.action_space(dynamics)
    def action_space(self, dynamics):
        # Define action space
        if not self.zero_action_middle:
            # Action range 0 .. 1
            self.low = np.zeros(4)
            self.bias = 0.0
            self.scale = 1.0
        else:
            # Action range -1 .. 1
            self.low = -np.ones(4)
            self.bias = 1.0
            self.scale = 0.5
        self.high = np.ones(4)
        return spaces.Box(self.low, self.high, dtype=np.float32)

    def step(self, dynamics, action,  dt, observation=None):
        # Execute action and update dynamics
        action = np.clip(action, a_min=self.low, a_max=self.high)
        action = self.scale * (action + self.bias)  # Scale and bias the action
        dynamics.step(action, dt)
        self.action = action.copy()  # Save current action

# VerticalControl class
class VerticalControl(object):
    def __init__(self, dynamics, zero_action_middle=True, dim_mode="3D"):
        self.zero_action_middle = zero_action_middle
        self.dim_mode = dim_mode
        if self.dim_mode == '1D':
            self.step = self.step1D
        elif self.dim_mode == '3D':
            self.step = self.step3D
        else:
            raise ValueError('QuadEnv: Unknown dimensionality mode %s' % self.dim_mode)
        self.step_func = self.step  # Bind step function

    def action_space(self, dynamics):
        # Define action space
        if not self.zero_action_middle:
            # Action range 0 .. 1
            self.low = np.zeros(1)
            self.bias = 0
            self.scale = 1.0
        else:
            # Action range -1 .. 1
            self.low = -np.ones(1)
            self.bias = 1.0
            self.scale = 0.5
        self.high = np.ones(1)
        return spaces.Box(self.low, self.high, dtype=np.float32)

    def step3D(self, dynamics, action, dt, observation=None):
        # 3D vertical control
        action = self.scale * (action + self.bias)
        action = np.clip(action, a_min=self.low, a_max=self.high)
        dynamics.step(np.array([action[0]] * 4), dt)  # Apply same thrust to all four motors

    def step1D(self, dynamics, action,  dt, observation=None):
        # 1D vertical control
        action = self.scale * (action + self.bias)
        action = np.clip(action, a_min=self.low, a_max=self.high)
        dynamics.step(np.array([action[0]]), dt)

# VertPlaneControl class
class VertPlaneControl(object):
    def __init__(self, dynamics, zero_action_middle=True, dim_mode="3D"):
        self.zero_action_middle = zero_action_middle
        self.dim_mode = dim_mode
        if self.dim_mode == '2D':
            self.step = self.step2D
        elif self.dim_mode == '3D':
            self.step = self.step3D
        else:
            raise ValueError('QuadEnv: Unknown dimensionality mode %s' % self.dim_mode)
        self.step_func = self.step

    def action_space(self, dynamics):
        # Define action space
        if not self.zero_action_middle:
            self.low = np.zeros(2)
            self.bias = 0
            self.scale = 1.0
        else:
            self.low = -np.ones(2)
            self.bias = 1.0
            self.scale = 0.5
        self.high = np.ones(2)
        return spaces.Box(self.low, self.high, dtype=np.float32)

    def step3D(self, dynamics, action, dt, observation=None):
        # 3D vertical plane control
        action = self.scale * (action + self.bias)
        action = np.clip(action, a_min=self.low, a_max=self.high)
        # Apply thrust symmetrically
        dynamics.step(np.array([action[0], action[0], action[1], action[1]]), dt)

    def step2D(self, dynamics, action, dt, observation=None):
        # 2D vertical plane control
        action = self.scale * (action + self.bias)
        action = np.clip(action, a_min=self.low, a_max=self.high)
        dynamics.step(np.array(action), dt)

# Function to compute quadrotor Jacobian
def quadrotor_jacobian(dynamics):
    # Compute torque and thrust matrices
    torque = dynamics.thrust_max * dynamics.prop_crossproducts.T
    torque[2, :] = dynamics.torque_max * dynamics.prop_ccw
    thrust = dynamics.thrust_max * np.ones((1, 4))
    dw = (1.0 / dynamics.inertia)[:, None] * torque
    dv = thrust / dynamics.mass
    J = np.vstack([dv, dw])  # Stack acceleration and angular acceleration Jacobians
    J_cond = np.linalg.cond(J)  # Compute condition number
    if J_cond > 50:
        print("WARN: Jacobian conditioning is high: ", J_cond)
    return J

# OmegaThrustControl class
class OmegaThrustControl(object):
    def __init__(self, dynamics):
        jacobian = quadrotor_jacobian(dynamics)
        self.Jinv = np.linalg.inv(jacobian)  # Compute inverse of Jacobian

    def action_space(self, dynamics):
        # Define action space
        circle_per_sec = 2 * np.pi
        max_rp = 5 * circle_per_sec  # Max roll and pitch rate
        max_yaw = 1 * circle_per_sec  # Max yaw rate
        min_g = -1.0
        max_g = dynamics.thrust_to_weight - 1.0
        low = np.array([min_g, -max_rp, -max_rp, -max_yaw])
        high = np.array([max_g, max_rp, max_rp, max_yaw])
        return spaces.Box(low, high, dtype=np.float32)

    def step(self, dynamics, action, dt):
        # P controller to control angular velocity
        kp = 5.0  # Proportional gain
        omega_err = dynamics.omega - action[1:]  # Angular velocity error
        dw_des = -kp * omega_err  # Desired angular acceleration
        acc_des = GRAV * (action[0] + 1.0)  # Desired acceleration
        des = np.append(acc_des, dw_des)  # Combine desired acceleration and angular acceleration
        thrusts = np.matmul(self.Jinv, des)  # Compute motor thrusts
        thrusts = np.clip(thrusts, 0, 1)
        dynamics.step(thrusts, dt)

# quadrotor_control.py

class VelocityControl(object):
    def __init__(self, dynamics):
        self.action = None
        self.kp_v = 2  # 速度控制的比例增益，可以根据需要调整
        self.kp_a = 200.0
        self.kd_a = 50.0
        self.rot_des = np.eye(3)
        self.step_func = self.step

        # 初始化雅可比矩阵的逆
        jacobian = quadrotor_jacobian(dynamics)
        self.Jinv = np.linalg.inv(jacobian)
        self.action_space(dynamics)
    def action_space(self, dynamics):
        # 定义动作空间
        max_velocity = 5.0  # 最大速度，可调整

        # 动作范围为 -max_velocity .. max_velocity
        self.low = np.full((3,), -max_velocity, dtype=np.float32)
        self.high = np.full((3,), max_velocity, dtype=np.float32)

        return spaces.Box(self.low, self.high, dtype=np.float32)
    

    def step(self, dynamics, action, dt, observation=None):
        # 执行速度控制
        action = np.clip(action, self.low, self.high)

        # 计算速度误差
        vel_error = action - dynamics.vel
        # 计算所需加速度
        acc_des = self.kp_v * vel_error + np.array([0, 0, GRAV])  # 考虑重力补偿

        # 计算期望的旋转矩阵
        xc_des = self.rot_des[:, 0]
        zb_des, _ = normalize(acc_des)
        yb_des, _ = normalize(np.cross(zb_des, xc_des))
        xb_des = np.cross(yb_des, zb_des)
        R_des = np.column_stack((xb_des, yb_des, zb_des))
        R = dynamics.rot

        def vee(R):
            return np.array([R[2, 1], R[0, 2], R[1, 0]])

        e_R = 0.5 * vee(np.matmul(R_des.T, R) - np.matmul(R.T, R_des))  # 旋转误差
        e_R[2] *= 0.2  # 减慢偏航动态
        e_w = dynamics.omega  # 角速度误差

        # 计算所需角加速度
        dw_des = -self.kp_a * e_R - self.kd_a * e_w  # 所需角加速度
        thrust_mag = np.dot(acc_des, R[:, 2])  # 推力大小

        des = np.append(thrust_mag, dw_des)
        thrusts = np.matmul(self.Jinv, des)
        thrusts = np.clip(thrusts, 0.0, 1.0)

        dynamics.step(thrusts, dt)
        self.action = thrusts.copy()




# NonlinearPositionController class using PyTorch
# 定义 NonlinearPositionController
class NonlinearPositionController(object):
    def __init__(self, dynamics, use_torch=True):
        # 初始化雅可比矩阵的逆
        jacobian = quadrotor_jacobian(dynamics)
        self.Jinv = np.linalg.inv(jacobian)
        self.action = None

        # 定义控制增益为 PyTorch 参数
        self.kp_p = nn.Parameter(torch.tensor(4.5))
        self.kd_p = nn.Parameter(torch.tensor(3.5))
        self.kp_a = nn.Parameter(torch.tensor(200.0))
        self.kd_a = nn.Parameter(torch.tensor(50.0))

        self.rot_des = np.eye(3)  # 期望的旋转矩阵

        self.action_space(dynamics)
        self.use_torch = use_torch
        if use_torch:
            self.step_func = self.step_torch
        else:
            self.step_func = self.step  # 使用原始的 step 函数


    def action_space(self, dynamics):
        # 动作为位置指令
        self.low=dynamics.room_box[0]
        self.high=dynamics.room_box[1]
        # low = np.array([-self.room_dim[0]/2, -self.room_dim[1]/2, 0], dtype=np.float32)
        # high = np.array([self.room_dim[0]/2, self.room_dim[1]/2, self.room_dim[2]], dtype=np.float32)
        return spaces.Box(self.low, self.high, dtype=np.float32)

    def step_torch(self, dynamics, action, dt, observation=None):
        # action 是期望的位置指令
        desired_pos = torch.from_numpy(action.astype(np.float32)).unsqueeze(0)  # (1, 3)

        # 转换无人机的当前状态为 PyTorch 张量
        xyz = torch.from_numpy(dynamics.pos.astype(np.float32)).unsqueeze(0)  # (1, 3)
        Vxyz = torch.from_numpy(dynamics.vel.astype(np.float32)).unsqueeze(0)  # (1, 3)
        Omega = torch.from_numpy(dynamics.omega.astype(np.float32)).unsqueeze(0)  # (1, 3)
        R = torch.from_numpy(dynamics.rot.astype(np.float32)).unsqueeze(0)  # (1, 3, 3)

        # 计算位置和速度误差
        to_goal = desired_pos - xyz  # (batch_size, 3)
        e_p_norm = torch.norm(to_goal, dim=1, keepdim=True)  # (batch_size, 1)
        max_norm = 4.0
        e_p = -to_goal * torch.clamp(max_norm / (e_p_norm + 1e-8), max=1.0)  # (batch_size, 3)
        e_v = Vxyz

        # 计算期望加速度
        acc_des = -self.kp_p * e_p - self.kd_p * e_v + torch.tensor([[0, 0, GRAV]])

        # 计算期望的旋转矩阵
        def project_xy(x):
            return x * torch.tensor([[1., 1., 0.]])  # (batch_size, 3)

        xc_des = project_xy(R[:, :, 0])  # (batch_size, 3)

        zb_des = F.normalize(acc_des, dim=1)  # (batch_size, 3)
        yb_des = F.normalize(torch.cross(zb_des, xc_des, dim=1), dim=1)  # (batch_size, 3)
        xb_des = torch.cross(yb_des, zb_des, dim=1)  # (batch_size, 3)

        R_des = torch.stack((xb_des, yb_des, zb_des), dim=2)  # (batch_size, 3, 3)

        # 计算旋转误差
        R_transpose = R.transpose(1, 2)  # (batch_size, 3, 3)
        R_des_transpose = R_des.transpose(1, 2)  # (batch_size, 3, 3)

        Rdiff = torch.matmul(R_des_transpose, R) - torch.matmul(R_transpose, R_des)  # (batch_size, 3, 3)

        def tf_vee(R):
            return torch.stack([R[:, 2, 1], R[:, 0, 2], R[:, 1, 0]], dim=1)  # (batch_size, 3)

        e_R = 0.5 * tf_vee(Rdiff)  # (batch_size, 3)
        e_R[:, 2] *= 0.2  # 减缓偏航动态
        e_w = Omega  # (batch_size, 3)

        # 计算所需的角加速度
        dw_des = -self.kp_a * e_R - self.kd_a * e_w  # (batch_size, 3)

        acc_cur = R[:, :, 2]  # (batch_size, 3)

        acc_dot = (acc_des * acc_cur).sum(dim=1, keepdim=True)  # (batch_size, 1)
        thrust_mag = acc_dot  # (batch_size, 1)

        # 拼接推力和角加速度
        des = torch.cat([thrust_mag, dw_des], dim=1)  # (batch_size, 4)
        des_np = des.detach().numpy().squeeze()  # Shape: (4,)

        thrusts = np.matmul(self.Jinv, des_np)  # (4,)
        thrusts = np.clip(thrusts, 0.0, 1.0)

        dynamics.step(thrusts, dt)
        self.action = thrusts.copy()

    def step(self, dynamics, action, dt, observation=None):
        # action 是期望的位置指令
        desired_pos = action  # (3,)
        to_goal = desired_pos - dynamics.pos  # 目标位置的向量
        goal_dist = np.linalg.norm(to_goal)
        e_p = -clamp_norm(to_goal, 4.0)  # 位置误差，限幅
        e_v = dynamics.vel  # 速度误差
        acc_des = -self.kp_p.detach().numpy() * e_p - self.kd_p.detach().numpy() * e_v + np.array([0, 0, GRAV])  # 所需加速度

        # 计算期望的旋转矩阵
        xc_des = self.rot_des[:, 0]
        zb_des, _ = normalize(acc_des)
        yb_des, _ = normalize(np.cross(zb_des, xc_des))
        xb_des = np.cross(yb_des, zb_des)
        R_des = np.column_stack((xb_des, yb_des, zb_des))
        R = dynamics.rot

        def vee(R):
            return np.array([R[2, 1], R[0, 2], R[1, 0]])

        e_R = 0.5 * vee(np.matmul(R_des.T, R) - np.matmul(R.T, R_des))  # 旋转误差
        e_R[2] *= 0.2  # 减慢偏航动态
        e_w = dynamics.omega  # 角速度误差

        dw_des = -self.kp_a.detach().numpy() * e_R - self.kd_a.detach().numpy() * e_w  # 所需角加速度
        thrust_mag = np.dot(acc_des, R[:, 2])  # 推力大小

        des = np.append(thrust_mag, dw_des)
        thrusts = np.matmul(self.Jinv, des)
        thrusts = np.clip(thrusts, 0.0, 1.0)

        dynamics.step(thrusts, dt)
        self.action = thrusts.copy()
# Note: Ensure that the 'normalize' and 'clamp_norm' functions are properly imported or defined.

# TODO:
# class AttitudeControl,
# refactor common parts of VelocityYaw and NonlinearPosition
