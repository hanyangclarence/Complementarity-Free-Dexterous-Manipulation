import casadi as cs
import numpy as np

import envs.allegro_fkin as allegro_fk
from utils import rotations


class ExplicitMPCParams:
    def __init__(self, rand_seed=1, target_type='rotation'):
        # ---------------------------------------------------------------------------------------------
        #      simulation parameters 
        # ---------------------------------------------------------------------------------------------
        self.model_path_ = 'envs/xmls/env_allegro_cube.xml'
        self.object_names_ = ['obj']

        self.h_ = 0.1
        self.frame_skip_ = int(50)

        # system dimensions:
        self.n_palm_qpos_ = 7  # palm translation(3) + rotation quaternion(4)
        self.n_palm_qvel_ = 6  # palm linear velocity(3) + angular velocity(3)
        self.n_robot_qpos_ = 16  # finger joints
        self.n_qpos_ = 30  # obj(7) + palm(7) + fingers(16)
        self.n_qvel_ = 28  # obj(6) + palm(6) + fingers(16)
        self.n_cmd_ = 22  # palm(6) + fingers(16)

        # ---------------------------------------------------------------------------------------------
        #      initial state and target state
        # ---------------------------------------------------------------------------------------------
        # np.random.seed(100 + rand_seed)

        self.init_robot_qpos_ = np.array([
            0.125, 1.13, 1.45, 1.24,
            -0.02, 0.445, 1.17, 1.5,
            -0.459, 1.54, 1.11, 1.23,
            0.638, 1.85, 1.5, 1.26
        ])

        # palm optimization flag (set to False for stable baseline behavior)
        self.optimize_palm_ = True  # TODO: Debug instability with palm optimization

        # random init and target pose for object
        if target_type == 'rotation':
            init_obj_xy = np.array([-0.03, -0.01]) + 0.005 * np.random.randn(2)
            init_obj_pos = np.hstack([init_obj_xy, 0.04])
            init_yaw_angle = np.pi * np.random.rand(1) - np.pi / 2
            init_obj_quat_rand = rotations.rpy_to_quaternion(np.hstack([init_yaw_angle, 0, 0]))
            self.init_obj_qpos_ = np.hstack((init_obj_pos, init_obj_quat_rand))

            # Target with randomized translation in all directions
            # Base position near hand center, with random offset
            target_xy_offset = 0.1 * (np.random.rand(2) - 0.5)  # ±1cm in x,y
            target_z_offset = 0.0 * np.random.rand(1)   # ±1cm in z
            self.target_p_ = np.array([-0.01, -0.0, 0.05]) + np.hstack([target_xy_offset, target_z_offset])

            yaw_angle = init_yaw_angle + np.random.choice([-np.pi / 2, np.pi / 2])
            self.target_q_ = rotations.rpy_to_quaternion(np.hstack([yaw_angle, 0, 0]))

        else:
            raise ValueError(f'Target type {target_type} not supported')

        # Palm at world origin with zero rotation (like original fixed-base version)
        # Palm qpos in MuJoCo: [x, y, z, roll, pitch, yaw]
        # Will be converted to [x, y, z, qw, qx, qy, qz] for MPC
        self.init_palm_qpos_ = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # ---------------------------------------------------------------------------------------------
        #      contact parameters
        # ---------------------------------------------------------------------------------------------
        self.mu_object_ = 0.5
        self.n_mj_q_ = self.n_qpos_
        self.n_mj_v_ = self.n_qvel_
        self.max_ncon_ = 20

        # ---------------------------------------------------------------------------------------------
        #      models parameters
        # ---------------------------------------------------------------------------------------------
        self.obj_inertia_ = np.identity(6)
        self.obj_inertia_[0:3, 0:3] = 50 * np.eye(3)
        self.obj_inertia_[3:, 3:] = 0.1 * np.eye(3)

        # Palm stiffness (6 DOF: linear + angular)
        if self.optimize_palm_:
            # When optimizing palm, use moderate stiffness
            # Higher stiffness for position, moderate for rotation
            self.palm_stiff_ = np.diag([100, 100, 100, 50, 50, 50])
        else:
            # When palm is fixed, use same stiffness as fingers to not disturb dynamics
            # (palm command will be constrained to zero anyway)
            self.palm_stiff_ = np.diag(self.n_palm_qvel_ * [1])
        self.robot_stiff_ = np.diag(self.n_robot_qpos_ * [1])

        Q = np.zeros((self.n_qvel_, self.n_qvel_))
        Q[:6, :6] = self.obj_inertia_
        Q[6:12, 6:12] = self.palm_stiff_
        Q[12:, 12:] = self.robot_stiff_
        self.Q = Q

        self.obj_mass_ = 0.01
        self.gravity_ = np.array([0.00, 0.00, -9.8, 0.0, 0.0, 0.0])

        self.model_params = 0.5

        # ---------------------------------------------------------------------------------------------
        #      planner parameters
        # ---------------------------------------------------------------------------------------------
        self.mpc_horizon_ = 4
        self.ipopt_max_iter_ = 50
        self.mpc_model = 'explicit'

        # Control bounds: palm_delta(6) [pos_delta(3), euler_delta(3)] + finger_delta(16)
        # NOTE: Commands are POSITION DELTAS, not velocities (like fingers)
        if self.optimize_palm_:
            palm_linear_u_bound = 0.001  # ±2mm position delta per step
            palm_angular_u_bound = 0.01  # ±0.002 rad ≈ 0.11 degrees per step
            # After fixing FK (removed double application of t_palm_base) and reducing rotation
            # actuator gains (kp=100 instead of 10000), rotation is much smoother with less bouncing
            palm_u_lb = np.hstack([-palm_linear_u_bound * np.ones(3), -palm_angular_u_bound * np.ones(3)])
            palm_u_ub = np.hstack([palm_linear_u_bound * np.ones(3), palm_angular_u_bound * np.ones(3)])
        else:
            palm_u_lb = np.zeros(6)  # Zero palm command when not optimizing (fixed palm)
            palm_u_ub = np.zeros(6)
        finger_u_bound = 0.2  # ±0.2 rad position delta per step for fingers
        self.mpc_u_lb_ = np.hstack([palm_u_lb, -finger_u_bound * np.ones(16)])
        self.mpc_u_ub_ = np.hstack([palm_u_ub, finger_u_bound * np.ones(16)])

        # State bounds: obj(7) + palm(7) + fingers(16)
        obj_pos_lb = np.array([-0.99, -0.99, 0])
        obj_pos_ub = np.array([0.99, 0.99, 0.99])

        # Palm bounds: position(3) + quaternion(4)
        if self.optimize_palm_:
            # Allow ±1cm from initial position when optimizing (conservative to prevent instability)
            palm_pos_tolerance = 0.01
            palm_pos_lb = self.init_palm_qpos_[0:3] - palm_pos_tolerance
            palm_pos_ub = self.init_palm_qpos_[0:3] + palm_pos_tolerance
            palm_pos_lb[2] = max(palm_pos_lb[2], -0.05)  # keep above ground level
        else:
            # Allow small tolerance (±1mm) to account for physics drift when palm is fixed
            # Even with zero command and position control, MuJoCo allows small drift (~0.46mm) from contacts
            palm_drift_tolerance = 0.001
            palm_pos_lb = self.init_palm_qpos_[0:3] - palm_drift_tolerance
            palm_pos_ub = self.init_palm_qpos_[0:3] + palm_drift_tolerance
            palm_pos_lb[2] = max(palm_pos_lb[2], -0.05)  # keep above ground level

        self.mpc_q_lb_ = np.hstack((obj_pos_lb, -1e7 * np.ones(4), palm_pos_lb, -1e7 * np.ones(4), -1e7 * np.ones(16)))
        self.mpc_q_ub_ = np.hstack((obj_pos_ub, 1e7 * np.ones(4), palm_pos_ub, 1e7 * np.ones(4), 1e7 * np.ones(16)))

        self.sol_guess_ = None

    # ---------------------------------------------------------------------------------------------
    #      cost functions for MPC
    # ---------------------------------------------------------------------------------------------
    def init_cost_fns(self):
        x = cs.SX.sym('x', self.n_qpos_)
        u = cs.SX.sym('u', self.n_cmd_)

        # State order: obj_pos(3), obj_quat(4), palm_pos(3), palm_quat(4), finger_qpos(16)
        obj_pose = x[0:7]
        palm_pose = x[7:14]  # palm_pos(3) + palm_quat(4)
        ff_qpos = x[14:18]
        mf_qpos = x[18:22]
        rf_qpos = x[22:26]
        tm_qpos = x[26:30]

        # forward kinematics to compute the position of fingertip (now includes palm_pose)
        ftp_1_position = allegro_fk.fftp_pos_fd_fn(palm_pose, ff_qpos)
        ftp_2_position = allegro_fk.mftp_pos_fd_fn(palm_pose, mf_qpos)
        ftp_3_position = allegro_fk.rftp_pos_fd_fn(palm_pose, rf_qpos)
        ftp_4_position = allegro_fk.thtp_pos_fd_fn(palm_pose, tm_qpos)

        # target cost
        target_position = cs.SX.sym('target_position', 3)
        target_quaternion = cs.SX.sym('target_quaternion', 4)
        position_cost = cs.sumsqr(obj_pose[0:3] - target_position)
        quaternion_cost = 1 - cs.dot(obj_pose[3:7], target_quaternion) ** 2
        contact_cost = (
                cs.sumsqr(obj_pose[0:3] - ftp_1_position) +
                cs.sumsqr(obj_pose[0:3] - ftp_2_position) +
                cs.sumsqr(obj_pose[0:3] - ftp_3_position) +
                cs.sumsqr(obj_pose[0:3] - ftp_4_position)
        )

        # grasp cost
        obj_v0 = ftp_1_position - x[0:3]
        obj_v1 = ftp_2_position - x[0:3]
        obj_v2 = ftp_3_position - x[0:3]
        obj_v3 = ftp_4_position - x[0:3]
        grasp_closure = cs.sumsqr(obj_v0 / cs.norm_2(obj_v0) + obj_v1 / cs.norm_2(obj_v1)
                                  + obj_v2 / cs.norm_2(obj_v2) + obj_v3 / cs.norm_2(obj_v3))

        # control cost with separate weights for palm and fingers
        palm_linear_cmd = u[0:3]  # palm position delta commands (meters)
        palm_angular_cmd = u[3:6]  # palm Euler angle delta commands (radians)
        finger_cmd = u[6:22]  # finger joint delta commands (radians)

        # Separate control costs
        palm_linear_control_cost = cs.sumsqr(palm_linear_cmd)
        palm_angular_control_cost = cs.sumsqr(palm_angular_cmd)
        finger_control_cost = cs.sumsqr(finger_cmd)

        # cost params
        phi_vec = cs.SX.sym('phi_vec', self.max_ncon_ * 4)
        jac_mat = cs.SX.sym('jac_mat', self.max_ncon_ * 4, self.n_qvel_)
        cost_param = cs.vvcat([target_position, target_quaternion, phi_vec, jac_mat])

        # base cost
        base_cost = 1 * contact_cost
        final_cost = 100 * position_cost + 5.0 * quaternion_cost

        # Separate control cost weights
        # Palm linear: higher weight to reduce translation jittering
        # Palm angular: moderate weight for rotation control
        # Fingers: lower weight to allow movement
        palm_linear_weight = 0.1   # Increase this to reduce palm translation jittering
        palm_angular_weight = 0.05  # Weight for palm rotation control
        finger_control_weight = 0.25    # Keep this low to allow finger movement

        path_cost_fn = cs.Function('path_cost_fn', [x, u, cost_param],
                                   [base_cost + palm_linear_weight * palm_linear_control_cost +
                                    palm_angular_weight * palm_angular_control_cost +
                                    finger_control_weight * finger_control_cost])
        final_cost_fn = cs.Function('final_cost_fn', [x, cost_param], [10 * final_cost])

        return path_cost_fn, final_cost_fn
