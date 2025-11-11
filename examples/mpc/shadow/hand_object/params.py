import casadi as cs
import numpy as np

import envs.shadow_fkin as shadow_fk
from utils import rotations


class HandObjectMPCParams:
    """
    MPC parameters for combined hand pose + object manipulation tracking.
    Optimizes both hand trajectory following and object manipulation.
    """
    def __init__(self, rand_seed=1):
        # ---------------------------------------------------------------------------------------------
        #      simulation parameters
        # ---------------------------------------------------------------------------------------------
        self.model_path_ = 'envs/xmls/env_shadow_hand_object.xml'
        self.object_names_ = ['obj']

        self.h_ = 0.1
        self.frame_skip_ = int(50)

        # system dimensions:
        self.n_palm_qpos_ = 7  # palm translation(3) + rotation quaternion(4)
        self.n_palm_qvel_ = 6  # palm linear velocity(3) + angular velocity(3)
        self.n_robot_qpos_ = 22  # finger joints (Shadow hand: 4+4+4+5+5)
        self.n_qpos_ = 36  # obj(7) + palm(7) + fingers(22)
        self.n_qvel_ = 34  # obj(6) + palm(6) + fingers(22)
        self.n_cmd_ = 28  # palm(6) + fingers(22)

        # ---------------------------------------------------------------------------------------------
        #      initial state and target state
        # ---------------------------------------------------------------------------------------------
        # Joint limits order: FF(4), MF(4), RF(4), LF(5), TH(5)
        joint_limits_lower = np.array([
            # First Finger (FF)
            -0.349, -0.262, 0.0, 0.0,
            # Middle Finger (MF)
            -0.349, -0.262, 0.0, 0.0,
            # Ring Finger (RF)
            -0.349, -0.262, 0.0, 0.0,
            # Little Finger (LF)
            0.0, -0.349, -0.262, 0.0, 0.0,
            # Thumb (TH)
            -1.047, 0.0, -0.209, -0.698, -0.262
        ])

        joint_limits_upper = np.array([
            # First Finger (FF)
            0.349, 1.571, 1.571, 1.571,
            # Middle Finger (MF)
            0.349, 1.571, 1.571, 1.571,
            # Ring Finger (RF)
            0.349, 1.571, 1.571, 1.571,
            # Little Finger (LF)
            0.785, 0.349, 1.571, 1.571, 1.571,
            # Thumb (TH)
            1.047, 1.222, 0.209, 0.698, 1.571
        ])

        # Initial finger joint positions (will be overridden by trajectory)
        self.init_robot_qpos_ = (joint_limits_lower + joint_limits_upper) / 2.0

        # Palm optimization flag - must be True for hand-object tracking
        self.optimize_palm_ = True

        # Initial object pose (will be overridden by trajectory)
        init_obj_pos = np.array([0.0, 0.0, 0.05])
        init_obj_quat = rotations.rpy_to_quaternion(np.array([0.0, 0.0, 0.0]))
        self.init_obj_qpos_ = np.hstack((init_obj_pos, init_obj_quat))

        # Target poses (will be updated from trajectory)
        self.target_p_ = init_obj_pos.copy()
        self.target_q_ = init_obj_quat.copy()

        # Initial palm pose at world origin with zero rotation
        self.init_palm_qpos_ = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Target hand pose (will be updated from trajectory)
        self.target_palm_qpos_ = np.hstack([
            np.array([0.0, 0.0, 0.0]),  # position
            np.array([1.0, 0.0, 0.0, 0.0])  # quaternion
        ])
        self.target_robot_qpos_ = self.init_robot_qpos_.copy()

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
        self.palm_stiff_ = np.diag([100, 100, 100, 50, 50, 50])
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

        # Control bounds: palm_delta(6) [pos_delta(3), euler_delta(3)] + finger_delta(22)
        palm_linear_u_bound = 0.002  # ±2mm position delta per step
        palm_angular_u_bound = 0.005  # ±0.005 rad per step
        palm_u_lb = np.hstack([-palm_linear_u_bound * np.ones(3), -palm_angular_u_bound * np.ones(3)])
        palm_u_ub = np.hstack([palm_linear_u_bound * np.ones(3), palm_angular_u_bound * np.ones(3)])

        finger_u_bound = 0.05  # ±0.05 rad position delta per step for fingers
        self.mpc_u_lb_ = np.hstack([palm_u_lb, -finger_u_bound * np.ones(22)])
        self.mpc_u_ub_ = np.hstack([palm_u_ub, finger_u_bound * np.ones(22)])

        # State bounds: obj(7) + palm(7) + fingers(22)
        obj_pos_lb = np.array([-1e7, -1e7, -1e7])
        obj_pos_ub = np.array([1e7, 1e7, 1e7])

        # Palm bounds: use relaxed bounds to allow tracking
        palm_pos_tolerance = 0.5  # ±50cm from initial position
        palm_pos_lb = self.init_palm_qpos_[0:3] - palm_pos_tolerance
        palm_pos_ub = self.init_palm_qpos_[0:3] + palm_pos_tolerance

        self.mpc_q_lb_ = np.hstack((obj_pos_lb, -1e7 * np.ones(4), palm_pos_lb, -1e7 * np.ones(4), -1e7 * np.ones(22)))
        self.mpc_q_ub_ = np.hstack((obj_pos_ub, 1e7 * np.ones(4), palm_pos_ub, 1e7 * np.ones(4), 1e7 * np.ones(22)))

        self.sol_guess_ = None

    # ---------------------------------------------------------------------------------------------
    #      cost functions for MPC - HYBRID HAND + OBJECT TRACKING
    # ---------------------------------------------------------------------------------------------
    def init_cost_fns(self):
        x = cs.SX.sym('x', self.n_qpos_)
        u = cs.SX.sym('u', self.n_cmd_)

        # State order: obj_pos(3), obj_quat(4), palm_pos(3), palm_quat(4), finger_qpos(22)
        obj_pos = x[0:3]
        obj_quat = x[3:7]
        palm_pos = x[7:10]
        palm_quat = x[10:14]
        finger_qpos = x[14:36]
        palm_pose = x[7:14]  # palm_pos(3) + palm_quat(4) for FK

        # Extract finger groups for FK
        ff_qpos = finger_qpos[0:4]   # FF: 0-3
        mf_qpos = finger_qpos[4:8]   # MF: 4-7
        rf_qpos = finger_qpos[8:12]  # RF: 8-11
        lf_qpos = finger_qpos[12:17] # LF: 12-16
        th_qpos = finger_qpos[17:22] # TH: 17-21

        # Forward kinematics for fingertips
        ftp_1_position = shadow_fk.fftp_pos_fd_fn(palm_pose, ff_qpos)
        ftp_2_position = shadow_fk.mftp_pos_fd_fn(palm_pose, mf_qpos)
        ftp_3_position = shadow_fk.rftp_pos_fd_fn(palm_pose, rf_qpos)
        ftp_4_position = shadow_fk.lftp_pos_fd_fn(palm_pose, lf_qpos)
        ftp_5_position = shadow_fk.thtp_pos_fd_fn(palm_pose, th_qpos)

        # Target parameters: object pose + hand pose
        target_obj_pos = cs.SX.sym('target_obj_pos', 3)
        target_obj_quat = cs.SX.sym('target_obj_quat', 4)
        target_palm_pos = cs.SX.sym('target_palm_pos', 3)
        target_palm_quat = cs.SX.sym('target_palm_quat', 4)
        target_finger_qpos = cs.SX.sym('target_finger_qpos', 22)

        # -----------------------------------------------------------------------------------------
        #      Object manipulation costs
        # -----------------------------------------------------------------------------------------
        # Object position and orientation tracking
        obj_position_cost = cs.sumsqr(obj_pos - target_obj_pos)
        obj_quaternion_cost = 1 - cs.dot(obj_quat, target_obj_quat) ** 2

        # Contact maintenance cost (keep fingertips close to object)
        contact_cost = (
            cs.sumsqr(obj_pos - ftp_1_position) +
            cs.sumsqr(obj_pos - ftp_2_position) +
            cs.sumsqr(obj_pos - ftp_3_position) +
            cs.sumsqr(obj_pos - ftp_4_position) +
            cs.sumsqr(obj_pos - ftp_5_position)
        )

        # Grasp closure cost (force closure metric)
        obj_v1 = ftp_1_position - obj_pos
        obj_v2 = ftp_2_position - obj_pos
        obj_v3 = ftp_3_position - obj_pos
        obj_v4 = ftp_4_position - obj_pos
        obj_v5 = ftp_5_position - obj_pos
        grasp_closure = cs.sumsqr(
            obj_v1 / cs.norm_2(obj_v1) +
            obj_v2 / cs.norm_2(obj_v2) +
            obj_v3 / cs.norm_2(obj_v3) +
            obj_v4 / cs.norm_2(obj_v4) +
            obj_v5 / cs.norm_2(obj_v5)
        )

        # -----------------------------------------------------------------------------------------
        #      Hand pose tracking costs
        # -----------------------------------------------------------------------------------------
        # Palm position and orientation tracking
        palm_position_cost = cs.sumsqr(palm_pos - target_palm_pos)
        palm_quaternion_cost = 1 - cs.dot(palm_quat, target_palm_quat) ** 2

        # Finger joint position tracking
        finger_position_cost = cs.sumsqr(finger_qpos - target_finger_qpos)

        # -----------------------------------------------------------------------------------------
        #      Control costs
        # -----------------------------------------------------------------------------------------
        palm_linear_cmd = u[0:3]  # palm position delta commands (meters)
        palm_angular_cmd = u[3:6]  # palm Euler angle delta commands (radians)
        finger_cmd = u[6:28]  # finger joint delta commands (radians) - 22 DOFs

        # Separate control costs
        palm_linear_control_cost = cs.sumsqr(palm_linear_cmd)
        palm_angular_control_cost = cs.sumsqr(palm_angular_cmd)
        finger_control_cost = cs.sumsqr(finger_cmd)

        # -----------------------------------------------------------------------------------------
        #      Cost parameters
        # -----------------------------------------------------------------------------------------
        phi_vec = cs.SX.sym('phi_vec', self.max_ncon_ * 4)
        jac_mat = cs.SX.sym('jac_mat', self.max_ncon_ * 4, self.n_qvel_)
        cost_param = cs.vvcat([target_obj_pos, target_obj_quat,
                               target_palm_pos, target_palm_quat, target_finger_qpos,
                               phi_vec, jac_mat])

        # -----------------------------------------------------------------------------------------
        #      Total cost weights
        # -----------------------------------------------------------------------------------------
        # Path cost: at every step along the horizon
        # Balance between tracking trajectory and maintaining control smoothness
        path_obj_pos_weight = 1.0       # Object position tracking
        path_obj_quat_weight = 0.5      # Object orientation tracking
        path_contact_weight = 1.0       # Maintain contact
        path_palm_pos_weight = 0.5      # Palm position tracking
        path_palm_quat_weight = 0.2     # Palm orientation tracking
        path_finger_pos_weight = 1.0    # Finger joint tracking

        palm_linear_weight = 0.05       # Control regularization
        palm_angular_weight = 0.05
        finger_control_weight = 0.1

        path_cost_fn = cs.Function('path_cost_fn', [x, u, cost_param],
                                   [path_obj_pos_weight * obj_position_cost +
                                    path_obj_quat_weight * obj_quaternion_cost +
                                    path_contact_weight * contact_cost +
                                    path_palm_pos_weight * palm_position_cost +
                                    path_palm_quat_weight * palm_quaternion_cost +
                                    path_finger_pos_weight * finger_position_cost +
                                    palm_linear_weight * palm_linear_control_cost +
                                    palm_angular_weight * palm_angular_control_cost +
                                    finger_control_weight * finger_control_cost])

        # Final cost: strong penalty at end of horizon to ensure targets are reached
        # Object tracking has highest priority, hand tracking is secondary
        final_obj_pos_weight = 100.0     # Strong object position matching
        final_obj_quat_weight = 10.0     # Strong object orientation matching
        final_palm_pos_weight = 20.0     # Moderate palm position matching
        final_palm_quat_weight = 5.0     # Moderate palm orientation matching
        final_finger_pos_weight = 10.0   # Moderate finger joint matching
        final_grasp_weight = 2.0         # Maintain grasp closure

        final_cost = (
            final_obj_pos_weight * obj_position_cost +
            final_obj_quat_weight * obj_quaternion_cost +
            final_palm_pos_weight * palm_position_cost +
            final_palm_quat_weight * palm_quaternion_cost +
            final_finger_pos_weight * finger_position_cost +
            final_grasp_weight * grasp_closure
        )

        final_cost_fn = cs.Function('final_cost_fn', [x, cost_param], [final_cost])

        return path_cost_fn, final_cost_fn
