import casadi as cs
import numpy as np
import time

from models.explicit_model import ExplicitModel


class MPCHandObject:
    """
    MPC for combined hand pose + object manipulation tracking.
    Optimizes both hand trajectory following and object manipulation simultaneously.
    """
    def __init__(self, param):
        self.param_ = param

        # cost function
        self.path_cost_fn, self.final_cost_fn = self.param_.init_cost_fns()

        # parse mpc model
        self.mpc_model = self.param_.mpc_model

        if self.mpc_model == 'explicit':
            self.model = ExplicitModel(param)
            self.init_MPC()
        else:
            raise ValueError('Invalid model type')

    def plan_once(self, target_obj_pos, target_obj_quat,
                  target_palm_pos, target_palm_quat, target_finger_qpos,
                  curr_x, phi_vec, jac_mat, sol_guess=None):
        """
        Plan one step for combined hand-object tracking.

        Args:
            target_obj_pos: target object position (3,)
            target_obj_quat: target object quaternion (4,)
            target_palm_pos: target palm position (3,)
            target_palm_quat: target palm quaternion (4,)
            target_finger_qpos: target finger joint positions (22,)
            curr_x: current state (36,)
            phi_vec: contact signed distances
            jac_mat: contact jacobian
            sol_guess: previous solution for warm start
        """
        if sol_guess is None:
            sol_guess = dict(x0=self.nlp_w0_, lam_x0=self.nlp_lam_x0_, lam_g0=self.nlp_lam_g0_)

        # Pack cost parameters: object target + hand target + contact info
        cost_params = cs.vvcat([target_obj_pos, target_obj_quat,
                               target_palm_pos, target_palm_quat, target_finger_qpos,
                               phi_vec, jac_mat])

        nlp_param = self.nlp_params_fn_(curr_x, phi_vec, jac_mat, cost_params, self.param_.model_params)

        nlp_lbw, nlp_ubw = self.nlp_bounds_fn_(self.param_.mpc_u_lb_, self.param_.mpc_u_ub_,
                                               self.param_.mpc_q_lb_,
                                               self.param_.mpc_q_ub_)

        st = time.time()
        raw_sol = self.ipopt_solver(x0=sol_guess['x0'],
                                    lam_x0=sol_guess['lam_x0'],
                                    lam_g0=sol_guess['lam_g0'],
                                    lbx=nlp_lbw, ubx=nlp_ubw,
                                    lbg=0.0, ubg=0.0,
                                    p=nlp_param)
        # print("mpc solve time:", time.time() - st)
        # print('mpc solve status = ', self.ipopt_solver.stats()['return_status'])

        w_opt = raw_sol['x'].full().flatten()
        cost_opt = raw_sol['f'].full().flatten()

        # extract the solution from the raw solution
        sol_traj = np.reshape(w_opt, (self.param_.mpc_horizon_, -1))
        opt_u_traj = sol_traj[:, 0:self.param_.n_cmd_]

        return dict(action=opt_u_traj[0, :],
                    sol_guess=dict(x0=w_opt,
                                   lam_x0=raw_sol['lam_x'],
                                   lam_g0=raw_sol['lam_g'],
                                   opt_cost=raw_sol['f'].full().item()),
                    cost_opt=cost_opt,
                    solve_status=self.ipopt_solver.stats()['return_status'])

    def init_MPC(self):
        model_params = cs.SX.sym('model_param', 1)

        phi_vec = cs.SX.sym('phi_vec', self.param_.max_ncon_ * 4)
        jac_mat = cs.SX.sym('jac_mat', self.param_.max_ncon_ * 4, self.param_.n_qvel_)

        cost_params = cs.SX.sym('cost_params', self.path_cost_fn.size_in(2))

        lbu = cs.SX.sym('lbu', self.param_.n_cmd_)
        ubu = cs.SX.sym('ubu', self.param_.n_cmd_)

        lbq = cs.SX.sym('lbq', self.param_.n_qpos_)
        ubq = cs.SX.sym('ubq', self.param_.n_qpos_)

        # start with empty NLP
        w, w0, lbw, ubw, g = [], [], [], [], []
        J = 0.0
        q0 = cs.SX.sym('q', self.param_.n_qpos_)
        qk = q0
        for k in range(self.param_.mpc_horizon_):
            # control at time k
            uk = cs.SX.sym('u' + str(k), self.param_.n_cmd_)
            w += [uk]
            lbw += [lbu]
            ubw += [ubu]
            w0 += [cs.DM.zeros(self.param_.n_cmd_)]

            # lse dyn function
            pred_q = self.model.step_once_fn(qk, uk, phi_vec, jac_mat, model_params)

            # compute the cost function
            J += self.path_cost_fn(qk, uk, cost_params)

            # q at time k+1 .... q_new
            qk = cs.SX.sym('q' + str(k + 1), self.param_.n_qpos_)
            w += [qk]
            w0 += [cs.DM.zeros(self.param_.n_qpos_)]
            lbw += [lbq]
            ubw += [ubq]

            # add the concatenation constraint
            g += [pred_q - qk]

        # compute the final cost
        J += self.final_cost_fn(qk, cost_params)

        # create an NLP solver
        nlp_params = cs.vvcat([q0, phi_vec, jac_mat, cost_params, model_params])
        nlp_prog = {'f': J, 'x': cs.vcat(w), 'g': cs.vcat(g), 'p': nlp_params}
        nlp_opts = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0,
                    'ipopt.max_iter': self.param_.ipopt_max_iter_}
        self.ipopt_solver = cs.nlpsol('solver', 'ipopt', nlp_prog, nlp_opts)

        # useful mappings
        self.nlp_w0_ = cs.vcat(w0)
        self.nlp_lam_x0_ = cs.DM.zeros(self.nlp_w0_.shape)
        self.nlp_lam_g0_ = cs.DM.zeros(cs.vcat(g).shape)
        self.nlp_bounds_fn_ = cs.Function('nlp_bounds_fn', [lbu, ubu, lbq, ubq], [cs.vcat(lbw), cs.vvcat(ubw)])
        self.nlp_params_fn_ = cs.Function('nlp_params_fn',
                                          [q0, phi_vec, jac_mat, cost_params, model_params], [nlp_params])
