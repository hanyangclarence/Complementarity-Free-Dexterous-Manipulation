import casadi as cs
import numpy as np

import envs.allegro_fkin as allegro_fk
from utils import rotations


class HandPoseMPCParams:
    def __init__(self, rand_seed=1):
        # ---------------------------------------------------------------------------------------------
        #      simulation parameters
        # ---------------------------------------------------------------------------------------------
        self.model_path_ = 'envs/xmls/env_allegro_hand_pose.xml'
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

        # Initial finger joint positions
        self.init_robot_qpos_ = np.array([
            0.125, 1.13, 1.45, 1.24,
            -0.02, 0.445, 1.17, 1.5,
            -0.459, 1.54, 1.11, 1.23,
            0.638, 1.85, 1.5, 1.26
        ])

        # Palm optimization flag - set to True for hand pose matching
        self.optimize_palm_ = True

        # Initial object pose (cube) - move it far away to avoid contact interference
        # For hand pose matching, we don't want cube contacts affecting palm dynamics
        init_obj_pos = np.array([0.5, 0.5, 0.5])  # 50cm away from hand workspace
        init_yaw_angle = 0.0
        init_obj_quat = rotations.rpy_to_quaternion(np.array([init_yaw_angle, 0, 0]))
        self.init_obj_qpos_ = np.hstack((init_obj_pos, init_obj_quat))

        # Target object pose - same as initial, we're not moving the cube
        # These are kept for compatibility with env.set_goal() (cube visualization)
        self.target_p_ = init_obj_pos.copy()
        self.target_q_ = init_obj_quat.copy()

        # Initial palm pose at world origin with zero rotation
        self.init_palm_qpos_ = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # ---------------------------------------------------------------------------------------------
        #      Target hand pose (palm + fingers) - RANDOMIZED
        # ---------------------------------------------------------------------------------------------
        # Target palm pose: randomize position and orientation
        # Position: allow ±3cm from initial in x,y and ±2cm in z
        target_palm_pos_offset = np.array([
            0.05 * np.random.rand() + 0.05,  # x: ±10cm
            0.1 * (2 * np.random.rand() - 1),  # y: ±10cm
            0.1 * (2 * np.random.rand() - 1)   # z: ±10cm
        ])
        target_palm_pos = self.init_palm_qpos_[0:3] + target_palm_pos_offset

        # Orientation: randomize roll, pitch, yaw within reasonable ranges
        target_palm_rpy = np.array([
            1.0 * (2 * np.random.rand() - 1),   # roll: ±0.2 rad (±11.5 deg)
            1.0 * (2 * np.random.rand() - 1),   # pitch: ±0.2 rad
            1.0 * (2 * np.random.rand() - 1)    # yaw: ±0.3 rad (±17 deg)
        ])
        target_palm_quat = rotations.rpy_to_quaternion(target_palm_rpy)

        # Combine target palm pose: position(3) + quaternion(4)
        self.target_palm_qpos_ = np.hstack([target_palm_pos, target_palm_quat])

        # Target finger joint positions: randomize around initial configuration
        # Allow ±0.5 rad variation for each joint
        finger_variation = (2 * np.random.rand(16) - 1) - 0.5
        self.target_robot_qpos_ = self.init_robot_qpos_ + finger_variation

        # Clip to reasonable joint limits to avoid impossible configurations
        # Allegro hand typical joint limits are approximately [-0.5, 1.8] rad
        self.target_robot_qpos_ = np.clip(self.target_robot_qpos_, -0.3, 1.8)

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

        # Control bounds: palm_delta(6) [pos_delta(3), euler_delta(3)] + finger_delta(16)
        palm_linear_u_bound = 0.002  # ±2mm position delta per step
        palm_angular_u_bound = 0.02  # ±0.02 rad per step
        palm_u_lb = np.hstack([-palm_linear_u_bound * np.ones(3), -palm_angular_u_bound * np.ones(3)])
        palm_u_ub = np.hstack([palm_linear_u_bound * np.ones(3), palm_angular_u_bound * np.ones(3)])

        finger_u_bound = 0.2  # ±0.2 rad position delta per step for fingers
        self.mpc_u_lb_ = np.hstack([palm_u_lb, -finger_u_bound * np.ones(16)])
        self.mpc_u_ub_ = np.hstack([palm_u_ub, finger_u_bound * np.ones(16)])

        # State bounds: obj(7) + palm(7) + fingers(16)
        obj_pos_lb = np.array([-1e7, -1e7, -1e7])
        obj_pos_ub = np.array([1e7, 1e7, 1e7])

        # Palm bounds: use very relaxed bounds for hand pose matching
        # Large bounds (±0.5m) prevent infeasibility while cost function guides optimization
        palm_pos_tolerance = 0.5  # ±50cm from initial position
        palm_pos_lb = self.init_palm_qpos_[0:3] - palm_pos_tolerance
        palm_pos_ub = self.init_palm_qpos_[0:3] + palm_pos_tolerance
        # palm_pos_lb[2] = max(palm_pos_lb[2], -0.05)  # keep above ground level

        self.mpc_q_lb_ = np.hstack((obj_pos_lb, -1e7 * np.ones(4), palm_pos_lb, -1e7 * np.ones(4), -1e7 * np.ones(16)))
        self.mpc_q_ub_ = np.hstack((obj_pos_ub, 1e7 * np.ones(4), palm_pos_ub, 1e7 * np.ones(4), 1e7 * np.ones(16)))

        self.sol_guess_ = None

    # ---------------------------------------------------------------------------------------------
    #      cost functions for MPC - HAND POSE MATCHING
    # ---------------------------------------------------------------------------------------------
    def init_cost_fns(self):
        x = cs.SX.sym('x', self.n_qpos_)
        u = cs.SX.sym('u', self.n_cmd_)

        # State order: obj_pos(3), obj_quat(4), palm_pos(3), palm_quat(4), finger_qpos(16)
        obj_pose = x[0:7]
        palm_pos = x[7:10]       # palm position(3)
        palm_quat = x[10:14]     # palm quaternion(4)
        finger_qpos = x[14:30]   # all 16 finger joints

        # Target hand pose (passed as parameters)
        target_palm_pos = cs.SX.sym('target_palm_pos', 3)
        target_palm_quat = cs.SX.sym('target_palm_quat', 4)
        target_finger_qpos = cs.SX.sym('target_finger_qpos', 16)

        # -----------------------------------------------------------------------------------------
        #      Hand pose matching costs
        # -----------------------------------------------------------------------------------------
        # Palm position cost
        palm_position_cost = cs.sumsqr(palm_pos - target_palm_pos)

        # Palm orientation cost (quaternion distance)
        palm_quaternion_cost = 1 - cs.dot(palm_quat, target_palm_quat) ** 2

        # Finger joint position cost
        finger_position_cost = cs.sumsqr(finger_qpos - target_finger_qpos)

        # -----------------------------------------------------------------------------------------
        #      Control costs
        # -----------------------------------------------------------------------------------------
        palm_linear_cmd = u[0:3]  # palm position delta commands (meters)
        palm_angular_cmd = u[3:6]  # palm Euler angle delta commands (radians)
        finger_cmd = u[6:22]  # finger joint delta commands (radians)

        # Separate control costs
        palm_linear_control_cost = cs.sumsqr(palm_linear_cmd)
        palm_angular_control_cost = cs.sumsqr(palm_angular_cmd)
        finger_control_cost = cs.sumsqr(finger_cmd)

        # -----------------------------------------------------------------------------------------
        #      Cost parameters (keeping contact info for future use with cube)
        # -----------------------------------------------------------------------------------------
        phi_vec = cs.SX.sym('phi_vec', self.max_ncon_ * 4)
        jac_mat = cs.SX.sym('jac_mat', self.max_ncon_ * 4, self.n_qvel_)
        cost_param = cs.vvcat([target_palm_pos, target_palm_quat, target_finger_qpos,
                               phi_vec, jac_mat])

        # -----------------------------------------------------------------------------------------
        #      Total cost weights
        # -----------------------------------------------------------------------------------------
        # Path cost: tracking at EVERY step + control regularization
        # These weights encourage the hand to move toward target throughout the trajectory
        path_palm_pos_weight = 0.0      # Track palm position along path
        path_palm_quat_weight = 0.0     # Track palm orientation along path
        path_finger_pos_weight = 0.0     # Track finger joints along path

        palm_linear_weight = 0.05        # Control regularization
        palm_angular_weight = 0.05
        finger_control_weight = 0.1

        path_cost_fn = cs.Function('path_cost_fn', [x, u, cost_param],
                                   [path_palm_pos_weight * palm_position_cost +
                                    path_palm_quat_weight * palm_quaternion_cost +
                                    path_finger_pos_weight * finger_position_cost +
                                    palm_linear_weight * palm_linear_control_cost +
                                    palm_angular_weight * palm_angular_control_cost +
                                    finger_control_weight * finger_control_cost])

        # Final cost: extra penalty at end of horizon to ensure target is reached precisely
        final_palm_pos_weight = 100.0     # Strong final palm position matching
        final_palm_quat_weight = 10.0    # Strong final palm orientation matching
        final_finger_pos_weight = 10.0    # Strong final finger joint matching

        final_cost = (final_palm_pos_weight * palm_position_cost +
                     final_palm_quat_weight * palm_quaternion_cost +
                     final_finger_pos_weight * finger_position_cost)

        final_cost_fn = cs.Function('final_cost_fn', [x, cost_param], [final_cost])

        return path_cost_fn, final_cost_fn
