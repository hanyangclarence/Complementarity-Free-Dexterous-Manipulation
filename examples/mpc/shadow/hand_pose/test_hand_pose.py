import time

import numpy as np

from examples.mpc.shadow.hand_pose.params import HandPoseMPCParams
from planning.mpc_hand_pose import MPCHandPose
from envs.shadow_env import MjSimulator
from contact.shadow_collision_detection import Contact

from utils import metrics

# -------------------------------
#       loop trials
# -------------------------------
save_flag = False
if save_flag:
    save_dir = './examples/mpc/shadow/hand_pose/save/'
    prefix_data_name = 'hand_pose_'
    save_data = dict()

trial_num = 20
# Success thresholds for hand pose matching
success_palm_pos_threshold = 0.01  # 1cm for palm position
success_palm_quat_threshold = 0.05  # quaternion error threshold
success_finger_threshold = 0.1  # 0.1 rad for finger joints
consecutive_success_time_threshold = 20
max_rollout_length = 200  # Longer horizon for hand pose matching

trial_count = 0
while trial_count < trial_num:

    # -------------------------------
    #        init parameters
    # -------------------------------
    param = HandPoseMPCParams(rand_seed=trial_count)

    print(f"\n{'='*60}")
    print(f"Trial {trial_count + 1}/{trial_num}")
    print(f"{'='*60}")
    print(f"Target palm position: {param.target_palm_qpos_[0:3]}")
    print(f"Target palm quaternion: {param.target_palm_qpos_[3:7]}")
    print(f"Target finger joints (first 4): {param.target_robot_qpos_[0:4]}")
    print(f"Initial palm position: {param.init_palm_qpos_[0:3]}")
    print(f"Initial finger joints (first 4): {param.init_robot_qpos_[0:4]}")
    print(f"{'='*60}\n")

    # -------------------------------
    #        init contact
    # -------------------------------
    contact = Contact(param)

    # -------------------------------
    #        init envs
    # -------------------------------
    env = MjSimulator(param)

    # -------------------------------
    #        set target hand pose visualization
    # -------------------------------
    print("DEBUG: Setting visualization target:")
    print(f"  Target palm pos (world): {param.target_palm_qpos_[0:3]}")
    print(f"  Target palm quat: {param.target_palm_qpos_[3:7]}")

    env.set_target_hand_pose(
        target_palm_pos=param.target_palm_qpos_[0:3],
        target_palm_quat=param.target_palm_qpos_[3:7],
        target_finger_qpos=param.target_robot_qpos_
    )

    # Verify what was actually set in the model
    try:
        actual_viz_pos = env.model_.body('target_palm').pos
        actual_viz_quat = env.model_.body('target_palm').quat
        print(f"DEBUG: Actual visualization body after setting:")
        print(f"  target_palm body pos: {actual_viz_pos}")
        print(f"  target_palm body quat: {actual_viz_quat}")
    except:
        print("  Could not read target_palm body")

    # Check actual palm body for comparison
    try:
        palm_body_pos = env.model_.body('palm').pos
        palm_body_quat = env.model_.body('palm').quat
        print(f"DEBUG: Actual palm body (base values):")
        print(f"  palm body pos: {palm_body_pos}")
        print(f"  palm body quat: {palm_body_quat}")
    except:
        print("  Could not read palm body")

    # -------------------------------
    #        init planner
    # -------------------------------
    mpc = MPCHandPose(param)

    # -------------------------------
    #        MPC rollout
    # -------------------------------
    rollout_step = 0
    consecutive_success_time = 0

    rollout_q_traj = []
    while rollout_step < max_rollout_length:
        if not env.dyn_paused_:
            # get state
            curr_q = env.get_state()
            rollout_q_traj.append(curr_q)

            # -----------------------
            #     contact detect
            # -----------------------
            phi_vec, jac_mat = contact.detect_once(env)

            # Debug: check contact status
            # Real contacts have negative phi (penetration)
            n_actual_contacts = np.sum(phi_vec < 0)
            n_mujoco_contacts = env.data_.ncon

            if rollout_step % 10 == 0:
                print(f"DEBUG: Contact info:")
                print(f"  MuJoCo detected contacts: {n_mujoco_contacts}")
                print(f"  Negative phi count: {n_actual_contacts}")
                print(f"  Phi values (first 8): {phi_vec[0:8]}")
                print(f"  Contact jac magnitude: {np.linalg.norm(jac_mat):.4f}")

                # Check Jacobian structure - which DOFs does it affect?
                if n_actual_contacts > 0:
                    # jac_mat is (max_ncon*4, n_qvel) = (80, 34)
                    # Columns: obj_vel(0:6), palm_vel(6:12), finger_vel(12:34)
                    obj_jac_norm = np.linalg.norm(jac_mat[:, 0:6])
                    palm_jac_norm = np.linalg.norm(jac_mat[:, 6:12])
                    finger_jac_norm = np.linalg.norm(jac_mat[:, 12:34])
                    print(f"  Jac norm by DOF - Obj: {obj_jac_norm:.4f}, Palm: {palm_jac_norm:.4f}, Fingers: {finger_jac_norm:.4f}")

                # Check cube position
                curr_cube_pos = curr_q[0:3]
                print(f"  Cube position: {curr_cube_pos}")

            # -----------------------
            #        planning
            # -----------------------
            # Pass target hand pose instead of object pose
            sol = mpc.plan_once(
                param.target_palm_qpos_[0:3],  # target palm position
                param.target_palm_qpos_[3:7],  # target palm quaternion
                param.target_robot_qpos_,       # target finger joints
                curr_q,
                phi_vec,
                jac_mat,
                sol_guess=param.sol_guess_)
            param.sol_guess_ = sol['sol_guess']
            action = sol['action']

            # -----------------------
            #        simulate
            # -----------------------
            env.step(action)
            rollout_step = rollout_step + 1

            # -----------------------
            #        success check
            # -----------------------
            curr_q = env.get_state()
            # Extract current hand pose
            curr_palm_pos = curr_q[7:10]
            curr_palm_quat = curr_q[10:14]
            curr_finger_qpos = curr_q[14:36]

            # Compute errors
            palm_pos_error = np.linalg.norm(curr_palm_pos - param.target_palm_qpos_[0:3])
            palm_quat_error = metrics.comp_quat_error(curr_palm_quat, param.target_palm_qpos_[3:7])
            finger_error = np.linalg.norm(curr_finger_qpos - param.target_robot_qpos_)

            # Check success criteria
            if (palm_quat_error < success_palm_quat_threshold and
                palm_pos_error < success_palm_pos_threshold and
                finger_error < success_finger_threshold):
                consecutive_success_time = consecutive_success_time + 1
            else:
                consecutive_success_time = 0

            # Print detailed progress every 10 steps
            if rollout_step % 10 == 0:
                print(f"\n{'='*70}")
                print(f"Step {rollout_step}:")
                print(f"  Palm Position Error: {palm_pos_error:.6f} m")
                print(f"    Current: {curr_palm_pos}")
                print(f"    Target:  {param.target_palm_qpos_[0:3]}")
                print(f"  Palm Quaternion Error: {palm_quat_error:.6f}")
                print(f"    Current: {curr_palm_quat}")
                print(f"    Target:  {param.target_palm_qpos_[3:7]}")
                print(f"  Finger Joint Error: {finger_error:.6f} rad")
                print(f"    Current: {curr_finger_qpos}")
                print(f"    Target : {param.target_robot_qpos_}")
                print(f"  Control action (palm): {action[0:6]}")
                print(f"  Control action (fingers, first 4): {action[6:10]}")
                print(f"{'='*70}")

            # -----------------------
            #       early termination
            # -----------------------
            if consecutive_success_time > consecutive_success_time_threshold:
                print(f"\nSuccess! Reached target hand pose at step {rollout_step}")
                break

    # -------------------------------
    #        close viewer
    # -------------------------------
    env.viewer_.close()
    time.sleep(0.5)

    # Print final result
    if rollout_step < max_rollout_length:
        print(f"Trial {trial_count + 1} SUCCEEDED in {rollout_step} steps")
    else:
        print(f"Trial {trial_count + 1} FAILED - max steps reached")

    # -------------------------------
    #        save data
    # -------------------------------
    if save_flag:
        # save
        save_data.update(target_palm_qpos=param.target_palm_qpos_)
        save_data.update(target_finger_qpos=param.target_robot_qpos_)
        save_data.update(rollout_traj=np.array(rollout_q_traj))
        # success index
        if rollout_step < max_rollout_length:
            save_data.update(success=True)
        else:
            save_data.update(success=False)
        # save to file
        metrics.save_data(save_data, data_name=prefix_data_name + 'trial_' + str(trial_count) + '_rollout',
                          save_dir=save_dir)

    trial_count = trial_count + 1
