import mujoco
import mujoco.viewer
import pathlib
import numpy as np
import time
import casadi as cs

import utils.rotations as rot


class MjSimulator():
    def __init__(self, param):

        self.param_ = param

        # init model data
        self.model_ = mujoco.MjModel.from_xml_path(self.param_.model_path_)
        self.data_ = mujoco.MjData(self.model_)

        self.fingertip_names_ = ['ftp_0', 'ftp_1', 'ftp_2', 'ftp_3', 'ftp_4']

        self.test_ft1_cmd = np.zeros(3)
        self.keyboard_sensitivity = 0.1
        self.break_out_signal_ = False
        self.dyn_paused_ = False

        self.set_goal(self.param_.target_p_, self.param_.target_q_)
        self.reset_env()

        self.viewer_ = mujoco.viewer.launch_passive(self.model_, self.data_, key_callback=self.keyboardCallback)

        self.shadow_fd_fn()

    def keyboardCallback(self, keycode):
        if chr(keycode) == ' ':
            self.dyn_paused_ = not self.dyn_paused_
            if self.dyn_paused_:
                print('simulation paused!')
            elif chr(keycode) == 'ĉ':
                self.test_ft1_cmd[1] += 0.001 * self.keyboard_sensitivity
            elif chr(keycode) == 'Ĉ':
                self.test_ft1_cmd[1] -= 0.001 * self.keyboard_sensitivity
            elif chr(keycode) == 'ć':
                self.test_ft1_cmd[0] -= 0.001 * self.keyboard_sensitivity
            elif chr(keycode) == 'Ć':
                self.test_ft1_cmd[0] += 0.001 * self.keyboard_sensitivity
            elif chr(keycode) == 'O':
                self.test_ft1_cmd[2] += 0.0005 * self.keyboard_sensitivity
            elif chr(keycode) == 'P':
                self.test_ft1_cmd[2] -= 0.0005 * self.keyboard_sensitivity
            elif chr(keycode) == 'R':
                self.test_ft1_cmd = np.array([0.0, 0.0, 0.0])
            else:
                print('simulation resumed!')
        elif chr(keycode) == 'Ā':
            self.break_out_signal_ = True

    def reset_env(self):
        # MuJoCo qpos order: palm_pos(3), palm_euler(3), fingers(22), object(7)
        self.data_.qpos[:] = np.hstack((self.param_.init_palm_qpos_, self.param_.init_robot_qpos_, self.param_.init_obj_qpos_))
        self.data_.qvel[:] = np.zeros(self.param_.n_qvel_)

        mujoco.mj_forward(self.model_, self.data_)

    def step(self, jpos_cmd):
        # Command is [palm_cmd(6), finger_pos_cmd(22)]
        # palm_cmd = [pos_delta(3), euler_delta(3)] - position deltas like fingers
        palm_cmd = jpos_cmd[0:6]
        finger_pos_cmd = jpos_cmd[6:]

        # Compute target palm pose
        if self.param_.optimize_palm_:
            # Get current palm pose (MuJoCo uses Euler angles)
            curr_palm_pos = self.data_.qpos[0:3].copy()
            curr_palm_euler = self.data_.qpos[3:6].copy()  # [roll, pitch, yaw]

            # Add position deltas (same as fingers)
            target_palm_pos = curr_palm_pos + palm_cmd[0:3]
            target_palm_euler = curr_palm_euler + palm_cmd[3:6]
        else:
            target_palm_pos = self.param_.init_palm_qpos_[0:3]
            target_palm_euler = self.param_.init_palm_qpos_[3:6]

        curr_finger_pos = self.get_jpos()
        target_finger_pos = curr_finger_pos + finger_pos_cmd

        # Combine into full control vector
        # MuJoCo control order: palm_pos(3), palm_euler(3), fingers(22)
        target_ctrl = np.hstack((target_palm_pos, target_palm_euler, target_finger_pos))

        for i in range(self.param_.frame_skip_):
            self.data_.ctrl = target_ctrl
            mujoco.mj_step(self.model_, self.data_)
            self.viewer_.sync()

    def reset_fingers_qpos(self):
        for iter in range(self.param_.frame_skip_):
            self.data_.ctrl = self.param_.init_robot_qpos_
            mujoco.mj_step(self.model_, self.data_)
            time.sleep(0.001)
            self.viewer_.sync()

    def get_state(self):
        # MuJoCo qpos order: palm_pos(3), palm_euler(3), fingers(22), object(7)
        # Return order for MPC: object(7), palm_pose(7) [pos(3)+quat(4)], fingers(22)
        palm_pos = self.data_.qpos.flatten().copy()[0:3]  # Joint space position
        finger_pos = self.data_.qpos.flatten().copy()[6:28]
        obj_pose = self.data_.qpos.flatten().copy()[28:35]

        # Get palm orientation from body quaternion (world frame), not from Euler angles
        # qpos[3:6] contains EXTRINSIC XYZ Euler angles (palm_rx, palm_ry, palm_rz are HINGE joints)
        # These compose as fixed-axis rotations in world frame, but we need the final quaternion
        palm_body_id = mujoco.mj_name2id(self.model_, mujoco.mjtObj.mjOBJ_BODY, 'palm')
        palm_quat = self.data_.xquat[palm_body_id].copy()  # [w, x, y, z] in world frame
        palm_pose = np.concatenate((palm_pos, palm_quat))

        return np.concatenate((obj_pose, palm_pose, finger_pos))

    def get_jpos(self):
        # Return only finger positions (skip palm pose)
        return self.data_.qpos.flatten().copy()[6:28]

    def get_fingertips_position(self):
        fts_pos = []
        for ft_name in self.fingertip_names_:
            fts_pos.append(self.data_.site(ft_name).xpos)
        return np.concatenate(fts_pos).flatten().copy()

    def set_goal(self, goal_pos=None, goal_quat=None):
        if goal_pos is not None:
            self.model_.body('goal').pos = goal_pos
        if goal_quat is not None:
            self.model_.body('goal').quat = goal_quat
        mujoco.mj_forward(self.model_, self.data_)
        pass

    def set_target_hand_pose(self, target_palm_pos=None, target_palm_quat=None, target_finger_qpos=None):
        """
        Set the target hand pose visualization in the viewer.

        Args:
            target_palm_pos: target palm position in world coordinates (3,)
            target_palm_quat: target palm quaternion (4,) [w, x, y, z] in world frame
            target_finger_qpos: target finger joint positions (22,)
        """
        if target_palm_pos is not None and target_palm_quat is not None:
            try:
                # For Shadow hand, qpos is already in world coordinates
                # No transformation needed (verified: qpos ≈ xpos)
                self.model_.body('target_palm').pos = target_palm_pos
                self.model_.body('target_palm').quat = target_palm_quat
            except KeyError:
                # target_palm body doesn't exist in this XML, skip
                pass

        # Compute and set target fingertip positions using forward kinematics
        if target_finger_qpos is not None and target_palm_pos is not None and target_palm_quat is not None:
            # FK expects palm_pose in world coordinates
            target_palm_pose = np.hstack([target_palm_pos, target_palm_quat])

            # Compute target fingertip positions
            try:
                ff_qpos = target_finger_qpos[0:4]
                mf_qpos = target_finger_qpos[4:8]
                rf_qpos = target_finger_qpos[8:12]
                lf_qpos = target_finger_qpos[12:17]
                th_qpos = target_finger_qpos[17:22]

                target_fftip_pos = np.array(self.fftp_pos_fd_fn(target_palm_pose, ff_qpos)).flatten()
                target_mftip_pos = np.array(self.mftp_pos_fd_fn(target_palm_pose, mf_qpos)).flatten()
                target_rftip_pos = np.array(self.rftp_pos_fd_fn(target_palm_pose, rf_qpos)).flatten()
                target_lftip_pos = np.array(self.lftp_pos_fd_fn(target_palm_pose, lf_qpos)).flatten()
                target_thtip_pos = np.array(self.thtp_pos_fd_fn(target_palm_pose, th_qpos)).flatten()

                # Set fingertip visualization positions
                self.model_.body('target_fftip').pos = target_fftip_pos
                self.model_.body('target_mftip').pos = target_mftip_pos
                self.model_.body('target_rftip').pos = target_rftip_pos
                self.model_.body('target_lftip').pos = target_lftip_pos
                self.model_.body('target_thtip').pos = target_thtip_pos
            except KeyError:
                # target fingertip bodies don't exist in this XML, skip
                pass

        mujoco.mj_forward(self.model_, self.data_)
        pass

    # forward kinematics of shadow hand
    def shadow_fd_fn(self):

        # palm pose: [pos(3), quat(4)]
        palm_pose = cs.SX.sym('palm_pose', 7)
        palm_pos = palm_pose[0:3]
        palm_quat = palm_pose[3:7]

        # First finger (4 DOFs)
        ff_qpos = cs.SX.sym('ff_qpos', 4)
        ff_t_base = rot.ttmat_fn(palm_pos) @ rot.quattmat_fn(palm_quat) @ rot.ttmat_fn([0.033, 0, 0.095])
        ff_t_knuckle = ff_t_base @ rot.rytmat_fn(-ff_qpos[0])
        ff_t_proximal = ff_t_knuckle @ rot.rxtmat_fn(ff_qpos[1])
        ff_t_middle = ff_t_proximal @ rot.ttmat_fn([0, 0, 0.045]) @ rot.rxtmat_fn(ff_qpos[2])
        ff_t_distal = ff_t_middle @ rot.ttmat_fn([0, 0, 0.025]) @ rot.rxtmat_fn(ff_qpos[3])
        ff_t_tip = ff_t_distal @ rot.ttmat_fn([0, 0, 0.026])
        self.fftp_pos_fd_fn = cs.Function('fftp_pos_fd_fn', [palm_pose, ff_qpos], [ff_t_tip[0:3, -1]])

        # Middle finger (4 DOFs)
        mf_qpos = cs.SX.sym('mf_qpos', 4)
        mf_t_base = rot.ttmat_fn(palm_pos) @ rot.quattmat_fn(palm_quat) @ rot.ttmat_fn([0.011, 0, 0.099])
        mf_t_knuckle = mf_t_base @ rot.rytmat_fn(-mf_qpos[0])
        mf_t_proximal = mf_t_knuckle @ rot.rxtmat_fn(mf_qpos[1])
        mf_t_middle = mf_t_proximal @ rot.ttmat_fn([0, 0, 0.045]) @ rot.rxtmat_fn(mf_qpos[2])
        mf_t_distal = mf_t_middle @ rot.ttmat_fn([0, 0, 0.025]) @ rot.rxtmat_fn(mf_qpos[3])
        mf_t_tip = mf_t_distal @ rot.ttmat_fn([0, 0, 0.026])
        self.mftp_pos_fd_fn = cs.Function('mftp_pos_fd_fn', [palm_pose, mf_qpos], [mf_t_tip[0:3, -1]])

        # Ring finger (4 DOFs)
        rf_qpos = cs.SX.sym('rf_qpos', 4)
        rf_t_base = rot.ttmat_fn(palm_pos) @ rot.quattmat_fn(palm_quat) @ rot.ttmat_fn([-0.011, 0, 0.095])
        rf_t_knuckle = rf_t_base @ rot.rytmat_fn(rf_qpos[0])
        rf_t_proximal = rf_t_knuckle @ rot.rxtmat_fn(rf_qpos[1])
        rf_t_middle = rf_t_proximal @ rot.ttmat_fn([0, 0, 0.045]) @ rot.rxtmat_fn(rf_qpos[2])
        rf_t_distal = rf_t_middle @ rot.ttmat_fn([0, 0, 0.025]) @ rot.rxtmat_fn(rf_qpos[3])
        rf_t_tip = rf_t_distal @ rot.ttmat_fn([0, 0, 0.026])
        self.rftp_pos_fd_fn = cs.Function('rftp_pos_fd_fn', [palm_pose, rf_qpos], [rf_t_tip[0:3, -1]])

        # Little finger (5 DOFs)
        lf_qpos = cs.SX.sym('lf_qpos', 5)
        lf_t_metacarpal_base = rot.ttmat_fn(palm_pos) @ rot.quattmat_fn(palm_quat) @ rot.ttmat_fn([-0.033, 0, 0.02071]) @ rot.quattmat_fn([0.88701801, 0, -0.46173483, 0])
        lf_t_metacarpal = lf_t_metacarpal_base @ rot.rxtmat_fn(lf_qpos[0])
        lf_t_knuckle = lf_t_metacarpal @ rot.ttmat_fn([0.05397275, 0, 0.03779463]) @ rot.quattmat_fn([0.88701801, 0, 0.46173483, 0]) @ rot.rytmat_fn(lf_qpos[1])
        lf_t_proximal = lf_t_knuckle @ rot.rxtmat_fn(lf_qpos[2])
        lf_t_middle = lf_t_proximal @ rot.ttmat_fn([0, 0, 0.045]) @ rot.rxtmat_fn(lf_qpos[3])
        lf_t_distal = lf_t_middle @ rot.ttmat_fn([0, 0, 0.025]) @ rot.rxtmat_fn(lf_qpos[4])
        lf_t_tip = lf_t_distal @ rot.ttmat_fn([0, 0, 0.026])
        self.lftp_pos_fd_fn = cs.Function('lftp_pos_fd_fn', [palm_pose, lf_qpos], [lf_t_tip[0:3, -1]])

        # Thumb (5 DOFs)
        th_qpos = cs.SX.sym('th_qpos', 5)
        th_t_base = rot.ttmat_fn(palm_pos) @ rot.quattmat_fn(palm_quat) @ rot.ttmat_fn([0.034, -0.00858, 0.029]) @ rot.quattmat_fn([0.65328148, -0.27059805, 0.27059805, -0.65328148])
        th_t_base1 = th_t_base @ rot.rztmat_fn(-th_qpos[0])
        th_t_proximal = th_t_base1 @ rot.rytmat_fn(th_qpos[1])
        th_t_hub = th_t_proximal @ rot.ttmat_fn([0, 0, 0.038]) @ rot.rytmat_fn(th_qpos[2])
        th_t_middle = th_t_hub @ rot.rxtmat_fn(th_qpos[3])
        th_t_distal = th_t_middle @ rot.ttmat_fn([0, 0, 0.032]) @ rot.rxtmat_fn(th_qpos[4])
        th_t_tip = th_t_distal @ rot.ttmat_fn([0, 0, 0.0275])
        self.thtp_pos_fd_fn = cs.Function('thtp_pos_fd_fn', [palm_pose, th_qpos], [th_t_tip[0:3, -1]])

        return 0
