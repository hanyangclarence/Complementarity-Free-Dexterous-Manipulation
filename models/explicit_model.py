import mujoco
import mujoco.viewer

import numpy as np
import casadi as cs

np.set_printoptions(suppress=True)


class ExplicitModel:
    def __init__(self, param):
        self.param_ = param

        self.init_utils()
        self.init_model()

    def init_utils(self):
        # -------------------------------
        #    quaternion integration fn
        # -------------------------------
        quat = cs.SX.sym('quat', 4)
        H_q_body = cs.vertcat(cs.horzcat(-quat[1], quat[0], quat[3], -quat[2]),
                              cs.horzcat(-quat[2], -quat[3], quat[0], quat[1]),
                              cs.horzcat(-quat[3], quat[2], -quat[1], quat[0]))
        self.cs_qmat_body_fn_ = cs.Function('cs_qmat_body_fn', [quat], [H_q_body.T])

        # -------------------------------
        #    state integration fn
        # -------------------------------
        # State order: obj_pos(3), obj_quat(4), palm_pos(3), palm_quat(4), finger_qpos(16)
        # Velocity order: obj_vel(6), palm_vel(6), finger_qvel(16)
        qvel = cs.SX.sym('qvel', self.param_.n_qvel_)
        qpos = cs.SX.sym('qpos', self.param_.n_qpos_)

        next_obj_pos = qpos[0:3] + self.param_.h_ * qvel[0:3]
        next_obj_quat = (qpos[3:7] + 0.5 * self.param_.h_ * self.cs_qmat_body_fn_(qpos[3:7]) @ qvel[3:6])
        next_palm_pos = qpos[7:10] + self.param_.h_ * qvel[6:9]
        next_palm_quat = (qpos[10:14] + 0.5 * self.param_.h_ * self.cs_qmat_body_fn_(qpos[10:14]) @ qvel[9:12])
        next_robot_qpos = qpos[14:] + self.param_.h_ * qvel[12:]

        next_qpos = cs.vertcat(next_obj_pos, next_obj_quat, next_palm_pos, next_palm_quat, next_robot_qpos)
        self.cs_qposInteg_ = cs.Function('cs_qposInte', [qpos, qvel], [next_qpos])

    def init_model(self):
        curr_q = cs.SX.sym('qvel', self.param_.n_qpos_)
        cmd = cs.SX.sym('cmd', self.param_.n_cmd_)
        phi_vec = cs.SX.sym('phi_vec', self.param_.max_ncon_ * 4)
        jac_mat = cs.SX.sym('jac_mat', self.param_.max_ncon_ * 4, self.param_.n_qvel_)

        # b vector in the QP formulation
        # Command order: palm_delta(6) [pos_delta(3) + euler_delta(3)], finger_delta(16)
        # NOTE: Commands are POSITION DELTAS, treated same as finger joint deltas
        palm_cmd = cmd[0:6]  # palm position and Euler angle deltas
        finger_cmd = cmd[6:]  # finger joint angle deltas

        b_o = cs.DM(self.param_.obj_mass_ * self.param_.gravity_)
        b_palm = self.param_.palm_stiff_ @ palm_cmd
        b_r = self.param_.robot_stiff_ @ finger_cmd
        b = cs.vertcat(b_o, b_palm, b_r)

        # Q matrix in the QP formulation
        Q = self.param_.Q
        Q_inv = np.linalg.inv(Q)

        # K matrix in the explicit model
        model_params = cs.SX.sym('sigma', 1)
        # K = sigma * cs.DM.eye(self.param_.max_ncon_ * 4)

        # time step h
        h = self.param_.h_

        # calculate the non-contact term
        v_non_contact = Q_inv @ b / h

        # calculate the contact term
        # contact_force = cs.fmax(-K @ (jac_mat @ Q_inv @ b + phi_vec), 0)
        contact_force = -model_params @ (jac_mat @ Q_inv @ b + phi_vec) - 0.1 * model_params @ jac_mat @ Q_inv @ b / h
        beta = 100.0
        contact_force = cs.log(1 + cs.exp(beta * contact_force)) / beta
        v_contact = Q_inv @ jac_mat.T @ contact_force / h

        # combine the velocity
        v = v_non_contact + v_contact

        # If palm is not being optimized, zero out palm velocity to enforce fixed palm
        if not self.param_.optimize_palm_:
            # Palm velocity (linear + angular) is at indices 6:12 in qvel
            v = cs.vertcat(v[0:6], cs.DM.zeros(6), v[12:])

        # time integration
        next_qpos = self.cs_qposInteg_(curr_q, v)

        # assemble the casadi function
        self.step_once_fn = cs.Function('step_once', [curr_q, cmd, phi_vec, jac_mat, model_params], [next_qpos])

    def step(self, curr_q, cmd, phi_vec, jac_mat, sigma):
        return self.step_once_fn(curr_q, cmd, phi_vec, jac_mat, sigma)
