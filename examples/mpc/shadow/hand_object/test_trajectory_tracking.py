import time
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco

from examples.mpc.shadow.hand_object.params import HandObjectMPCParams
from planning.mpc_hand_object import MPCHandObject
from envs.shadow_env import MjSimulator
from contact.shadow_collision_detection import Contact

from utils import metrics

# -------------------------------
#   Load trajectory data
# -------------------------------
trajectory_path = Path('/home/hanyang/code/humanoid/Genesis/output_trajectories_mujoco/shadow_hand_trajectory_g0_mujoco.pkl')
if not trajectory_path.exists():
    raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")

print(f"\n{'='*60}")
print(f"Loading trajectory from: {trajectory_path}")
print(f"{'='*60}")

with open(trajectory_path, 'rb') as f:
    data = pickle.load(f)

# Extract shadow hand trajectory
shadow_traj = data['shadow_hand_trajectory']
wrist_positions = shadow_traj['wrist_positions']  # (T, 3)
wrist_rotations_aa = shadow_traj['wrist_rotations_aa']  # (T, 3) axis-angle
dof_positions = shadow_traj['dof_positions']  # (T, 22)

# Extract object trajectory
object_traj = data['object_trajectory']
object_positions = object_traj['positions']  # (T, 3)
object_pose_matrices = object_traj['pose_matrices']  # (T, 4, 4)

T = len(wrist_positions)
print(f"Trajectory length: {T} timesteps")
print(f"DOF positions shape: {dof_positions.shape}")
print(f"Object positions shape: {object_positions.shape}")

# IMPORTANT: Reorder DOF positions from trajectory order to MuJoCo order
# Trajectory order: FF(4), LF(5), MF(4), RF(4), TH(5)
# MuJoCo order:     FF(4), MF(4), RF(4), LF(5), TH(5)
# We need to swap LF and MF positions
print(f"\nReordering finger DOFs from trajectory to MuJoCo order...")
print(f"  Trajectory: FF(4), LF(5), MF(4), RF(4), TH(5)")
print(f"  MuJoCo:     FF(4), MF(4), RF(4), LF(5), TH(5)")

dof_positions_reordered = np.zeros_like(dof_positions)
for t in range(T):
    traj_dofs = dof_positions[t]
    # Extract from trajectory order
    ff_traj = traj_dofs[0:4]    # FF: indices 0-3
    lf_traj = traj_dofs[4:9]    # LF: indices 4-8
    mf_traj = traj_dofs[9:13]   # MF: indices 9-12
    rf_traj = traj_dofs[13:17]  # RF: indices 13-16
    th_traj = traj_dofs[17:22]  # TH: indices 17-21

    # Reorder to MuJoCo order: FF, MF, RF, LF, TH
    dof_positions_reordered[t] = np.concatenate([
        ff_traj,  # FF: 0-3
        mf_traj,  # MF: 4-7
        rf_traj,  # RF: 8-11
        lf_traj,  # LF: 12-16
        th_traj   # TH: 17-21
    ])

# Replace with reordered version
dof_positions = dof_positions_reordered

print(f"{'='*60}\n")

# Convert axis-angle rotations to quaternions for wrist
wrist_quaternions = np.zeros((T, 4))
for t in range(T):
    axis_angle = wrist_rotations_aa[t]
    angle = np.linalg.norm(axis_angle)
    if angle > 1e-6:
        axis = axis_angle / angle
        rot = R.from_rotvec(axis_angle)
        quat = rot.as_quat()  # [x, y, z, w] format
        # Convert to [w, x, y, z] format for MuJoCo
        wrist_quaternions[t] = np.array([quat[3], quat[0], quat[1], quat[2]])
    else:
        wrist_quaternions[t] = np.array([1.0, 0.0, 0.0, 0.0])

# Convert object pose matrices to position + quaternion
object_quaternions = np.zeros((T, 4))
for t in range(T):
    pose_matrix = object_pose_matrices[t]
    rot_matrix = pose_matrix[:3, :3]
    rot = R.from_matrix(rot_matrix)
    quat = rot.as_quat()  # [x, y, z, w] format
    # Convert to [w, x, y, z] format for MuJoCo
    object_quaternions[t] = np.array([quat[3], quat[0], quat[1], quat[2]])

print(f"Sample trajectory data (timestep 0):")
print(f"  Object position: {object_positions[0]}")
print(f"  Object quaternion [w,x,y,z]: {object_quaternions[0]}")
print(f"  Wrist position: {wrist_positions[0]}")
print(f"  Wrist quaternion [w,x,y,z]: {wrist_quaternions[0]}")
print(f"  Finger DOFs (first 5): {dof_positions[0, :5]}")
print()

# -------------------------------
#   Initialize parameters
# -------------------------------
# Use first timestep as initial pose
param = HandObjectMPCParams(rand_seed=0)

# Override initial poses with trajectory start
# Object
param.init_obj_qpos_[0:3] = object_positions[0]
param.init_obj_qpos_[3:7] = object_quaternions[0]

# Palm - use extrinsic XYZ Euler angles
rot_palm_0 = R.from_quat([wrist_quaternions[0][1], wrist_quaternions[0][2],
                           wrist_quaternions[0][3], wrist_quaternions[0][0]])
palm_euler_0 = rot_palm_0.as_euler('XYZ')  # Extrinsic
param.init_palm_qpos_[0:3] = wrist_positions[0]
param.init_palm_qpos_[3:6] = palm_euler_0

# Fingers
param.init_robot_qpos_ = dof_positions[0]

print(f"Initial state:")
print(f"  Object position: {param.init_obj_qpos_[0:3]}")
print(f"  Object quaternion: {param.init_obj_qpos_[3:7]}")
print(f"  Palm position: {param.init_palm_qpos_[0:3]}")
print(f"  Palm euler (extrinsic XYZ): {param.init_palm_qpos_[3:6]}")
print(f"  Initial finger DOFs (first 5): {param.init_robot_qpos_[:5]}")
print()

# -------------------------------
#   Initialize contact
# -------------------------------
contact = Contact(param)

# -------------------------------
#   Initialize environment
# -------------------------------
env = MjSimulator(param)

# Directly set the initial state to match trajectory
# Object
env.data_.qpos[28:31] = object_positions[0]  # Object position
env.data_.qpos[31:35] = object_quaternions[0]  # Object quaternion

# Palm
env.data_.qpos[0:3] = wrist_positions[0]  # Palm position
# For orientation: MuJoCo palm uses 3 HINGE joints (rx, ry, rz) that compose as EXTRINSIC rotations
env.data_.qpos[3:6] = palm_euler_0  # Palm extrinsic Euler angles

# Fingers
env.data_.qpos[6:28] = dof_positions[0]

# Call forward kinematics to update everything
mujoco.mj_forward(env.model_, env.data_)

# Verify initialization
curr_q_init = env.get_state()
obj_pos_init = curr_q_init[0:3]
obj_quat_init = curr_q_init[3:7]
palm_pos_init = curr_q_init[7:10]
palm_quat_init = curr_q_init[10:14]
finger_init = curr_q_init[14:36]

obj_pos_error = np.linalg.norm(obj_pos_init - object_positions[0])
obj_quat_error = metrics.comp_quat_error(obj_quat_init, object_quaternions[0])
palm_pos_error = np.linalg.norm(palm_pos_init - wrist_positions[0])
palm_quat_error = metrics.comp_quat_error(palm_quat_init, wrist_quaternions[0])
finger_error = np.linalg.norm(finger_init - dof_positions[0])

print(f"Initial pose verification:")
print(f"  Object position error: {obj_pos_error:.6f} m")
print(f"  Object quaternion error: {obj_quat_error:.6f}")
print(f"  Palm position error: {palm_pos_error:.6f} m")
print(f"  Palm quaternion error: {palm_quat_error:.6f}")
print(f"  Finger error: {finger_error:.6f} rad")
print()

# -------------------------------
#   Initialize planner
# -------------------------------
mpc = MPCHandObject(param)

# -------------------------------
#   Tracking parameters
# -------------------------------
trajectory_step_time = 5  # MPC steps per trajectory timestep (adjustable)

# -------------------------------
#   MPC rollout with trajectory tracking
# -------------------------------
rollout_step = 0
trajectory_idx = 0

rollout_q_traj = []
tracking_errors = {
    'obj_pos': [],
    'obj_quat': [],
    'palm_pos': [],
    'palm_quat': [],
    'finger': [],
    'timestep': []
}

print(f"{'='*60}")
print(f"Hand-Object Trajectory Tracking ({T} frames)")
print(f"{'='*60}\n")

while trajectory_idx < T:
    if not env.dyn_paused_:
        # Get current state
        curr_q = env.get_state()
        rollout_q_traj.append(curr_q)

        # Get targets for current trajectory frame
        target_obj_pos = object_positions[trajectory_idx]
        target_obj_quat = object_quaternions[trajectory_idx]
        target_palm_pos = wrist_positions[trajectory_idx]
        target_palm_quat = wrist_quaternions[trajectory_idx]
        target_finger_qpos = dof_positions[trajectory_idx]

        # Update visualization
        if rollout_step % trajectory_step_time == 0:
            # Update object goal visualization
            env.set_goal(target_obj_pos, target_obj_quat)

            # Update hand target visualization
            env.set_target_hand_pose(
                target_palm_pos=target_palm_pos,
                target_palm_quat=target_palm_quat,
                target_finger_qpos=target_finger_qpos
            )

        # -----------------------
        #     contact detect
        # -----------------------
        phi_vec, jac_mat = contact.detect_once(env)

        # -----------------------
        #        planning
        # -----------------------
        sol = mpc.plan_once(
            target_obj_pos,
            target_obj_quat,
            target_palm_pos,
            target_palm_quat,
            target_finger_qpos,
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
        #   tracking error check
        # -----------------------
        curr_q = env.get_state()
        curr_obj_pos = curr_q[0:3]
        curr_obj_quat = curr_q[3:7]
        curr_palm_pos = curr_q[7:10]
        curr_palm_quat = curr_q[10:14]
        curr_finger_qpos = curr_q[14:36]

        # Compute errors
        obj_pos_error = np.linalg.norm(curr_obj_pos - target_obj_pos)
        obj_quat_error = metrics.comp_quat_error(curr_obj_quat, target_obj_quat)
        palm_pos_error = np.linalg.norm(curr_palm_pos - target_palm_pos)
        palm_quat_error = metrics.comp_quat_error(curr_palm_quat, target_palm_quat)
        finger_error = np.linalg.norm(curr_finger_qpos - target_finger_qpos)

        # Record errors
        tracking_errors['obj_pos'].append(obj_pos_error)
        tracking_errors['obj_quat'].append(obj_quat_error)
        tracking_errors['palm_pos'].append(palm_pos_error)
        tracking_errors['palm_quat'].append(palm_quat_error)
        tracking_errors['finger'].append(finger_error)
        tracking_errors['timestep'].append(trajectory_idx)

        # Print progress
        if rollout_step % 10 == 0:
            print(f"Tracking step {rollout_step}, Trajectory frame {trajectory_idx}/{T}:")
            print(f"  Object pos error: {obj_pos_error:.4f} m")
            print(f"  Object quat error: {obj_quat_error:.4f}")
            print(f"  Palm pos error: {palm_pos_error:.4f} m")
            print(f"  Palm quat error: {palm_quat_error:.4f}")
            print(f"  Finger error: {finger_error:.4f} rad")

        # -----------------------
        #   advance trajectory
        # -----------------------
        if rollout_step % trajectory_step_time == 0:
            trajectory_idx = min(trajectory_idx + 1, T - 1)

        # Early termination if trajectory completed
        if trajectory_idx >= T - 1:
            print(f"\nTrajectory completed at step {rollout_step}")
            break

# -------------------------------
#   Close viewer
# -------------------------------
env.viewer_.close()
time.sleep(0.5)

# -------------------------------
#   Print statistics
# -------------------------------
print(f"\n{'='*60}")
print(f"Tracking Statistics")
print(f"{'='*60}")
print(f"Total rollout steps: {rollout_step}")
print(f"Trajectory timesteps covered: {trajectory_idx + 1}/{T}")
print(f"\nAverage tracking errors:")
print(f"  Object position: {np.mean(tracking_errors['obj_pos']):.4f} m")
print(f"  Object quaternion: {np.mean(tracking_errors['obj_quat']):.4f}")
print(f"  Palm position: {np.mean(tracking_errors['palm_pos']):.4f} m")
print(f"  Palm quaternion: {np.mean(tracking_errors['palm_quat']):.4f}")
print(f"  Finger joints: {np.mean(tracking_errors['finger']):.4f} rad")
print(f"\nMax tracking errors:")
print(f"  Object position: {np.max(tracking_errors['obj_pos']):.4f} m")
print(f"  Object quaternion: {np.max(tracking_errors['obj_quat']):.4f}")
print(f"  Palm position: {np.max(tracking_errors['palm_pos']):.4f} m")
print(f"  Palm quaternion: {np.max(tracking_errors['palm_quat']):.4f}")
print(f"  Finger joints: {np.max(tracking_errors['finger']):.4f} rad")
print(f"{'='*60}\n")

# -------------------------------
#   Optional: Save tracking data
# -------------------------------
save_flag = False
if save_flag:
    save_dir = './examples/mpc/shadow/hand_object/save/'
    save_data = {
        'rollout_traj': np.array(rollout_q_traj),
        'target_trajectory': {
            'object_positions': object_positions,
            'object_quaternions': object_quaternions,
            'wrist_positions': wrist_positions,
            'wrist_quaternions': wrist_quaternions,
            'dof_positions': dof_positions
        },
        'tracking_errors': tracking_errors
    }
    metrics.save_data(save_data,
                      data_name='hand_object_tracking_rollout',
                      save_dir=save_dir)
    print(f"Tracking data saved to {save_dir}")
