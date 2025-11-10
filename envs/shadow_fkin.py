import casadi as cs
import utils.rotations as rot


# Forward kinematics for Shadow Hand (5 fingers with varying DOFs)
# Based on shadow_hand_right_woarm.urdf kinematics

# palm_pose: [pos(3), quat(4)]
palm_pose = cs.SX.sym('palm_pose', 7)
palm_pos = palm_pose[0:3]
palm_quat = palm_pose[3:7]

# ============================================================================
# First Finger (Index) - 4 DOFs: FFJ3, FFJ2, FFJ1, FFJ0
# ============================================================================
ff_qpos = cs.SX.sym('ff_qpos', 4)
# Base at palm: xyz="0.033 0 0.095", axis="0 -1 0"
ff_t_base = rot.ttmat_fn(palm_pos) @ rot.quattmat_fn(palm_quat) @ rot.ttmat_fn([0.033, 0, 0.095])
ff_t_knuckle = ff_t_base @ rot.rytmat_fn(-ff_qpos[0])  # FFJ3 (knuckle, axis 0 -1 0)
# Proximal: xyz="0 0 0", axis="1 0 0"
ff_t_proximal = ff_t_knuckle @ rot.rxtmat_fn(ff_qpos[1])  # FFJ2
# Middle: xyz="0 0 0.045", axis="1 0 0"
ff_t_middle = ff_t_proximal @ rot.ttmat_fn([0, 0, 0.045]) @ rot.rxtmat_fn(ff_qpos[2])  # FFJ1
# Distal: xyz="0 0 0.025", axis="1 0 0"
ff_t_distal = ff_t_middle @ rot.ttmat_fn([0, 0, 0.025]) @ rot.rxtmat_fn(ff_qpos[3])  # FFJ0
# Tip: xyz="0 0 0.026"
ff_t_tip = ff_t_distal @ rot.ttmat_fn([0, 0, 0.026])
fftp_pos_fd_fn = cs.Function('fftp_pos_fd_fn', [palm_pose, ff_qpos], [ff_t_tip[0:3, -1]])

# ============================================================================
# Middle Finger - 4 DOFs: MFJ3, MFJ2, MFJ1, MFJ0
# ============================================================================
mf_qpos = cs.SX.sym('mf_qpos', 4)
# Base at palm: xyz="0.011 0 0.099", axis="0 -1 0"
mf_t_base = rot.ttmat_fn(palm_pos) @ rot.quattmat_fn(palm_quat) @ rot.ttmat_fn([0.011, 0, 0.099])
mf_t_knuckle = mf_t_base @ rot.rytmat_fn(-mf_qpos[0])  # MFJ3
mf_t_proximal = mf_t_knuckle @ rot.rxtmat_fn(mf_qpos[1])  # MFJ2
mf_t_middle = mf_t_proximal @ rot.ttmat_fn([0, 0, 0.045]) @ rot.rxtmat_fn(mf_qpos[2])  # MFJ1
mf_t_distal = mf_t_middle @ rot.ttmat_fn([0, 0, 0.025]) @ rot.rxtmat_fn(mf_qpos[3])  # MFJ0
mf_t_tip = mf_t_distal @ rot.ttmat_fn([0, 0, 0.026])
mftp_pos_fd_fn = cs.Function('mftp_pos_fd_fn', [palm_pose, mf_qpos], [mf_t_tip[0:3, -1]])

# ============================================================================
# Ring Finger - 4 DOFs: RFJ3, RFJ2, RFJ1, RFJ0
# ============================================================================
rf_qpos = cs.SX.sym('rf_qpos', 4)
# Base at palm: xyz="-0.011 0 0.095", axis="0 1 0"
rf_t_base = rot.ttmat_fn(palm_pos) @ rot.quattmat_fn(palm_quat) @ rot.ttmat_fn([-0.011, 0, 0.095])
rf_t_knuckle = rf_t_base @ rot.rytmat_fn(rf_qpos[0])  # RFJ3 (axis 0 1 0)
rf_t_proximal = rf_t_knuckle @ rot.rxtmat_fn(rf_qpos[1])  # RFJ2
rf_t_middle = rf_t_proximal @ rot.ttmat_fn([0, 0, 0.045]) @ rot.rxtmat_fn(rf_qpos[2])  # RFJ1
rf_t_distal = rf_t_middle @ rot.ttmat_fn([0, 0, 0.025]) @ rot.rxtmat_fn(rf_qpos[3])  # RFJ0
rf_t_tip = rf_t_distal @ rot.ttmat_fn([0, 0, 0.026])
rftp_pos_fd_fn = cs.Function('rftp_pos_fd_fn', [palm_pose, rf_qpos], [rf_t_tip[0:3, -1]])

# ============================================================================
# Little Finger - 5 DOFs: LFJ4, LFJ3, LFJ2, LFJ1, LFJ0
# ============================================================================
lf_qpos = cs.SX.sym('lf_qpos', 5)
# Metacarpal at palm: xyz="-0.033 0 0.02071", rpy="0 -0.9599 0", axis="1 0 0"
# quat = [w, x, y, z] = [0.88701801, 0, -0.46173483, 0]
lf_t_metacarpal_base = rot.ttmat_fn(palm_pos) @ rot.quattmat_fn(palm_quat) @ rot.ttmat_fn([-0.033, 0, 0.02071]) @ rot.quattmat_fn([0.88701801, 0, -0.46173483, 0])
lf_t_metacarpal = lf_t_metacarpal_base @ rot.rxtmat_fn(lf_qpos[0])  # LFJ4
# Knuckle: xyz="0.05397275 0 0.03779463" (in metacarpal's rotated frame), rpy="0 0.9599 0", axis="0 1 0"
# quat = [0.88701801, 0, 0.46173483, 0]
lf_t_knuckle = lf_t_metacarpal @ rot.ttmat_fn([0.05397275, 0, 0.03779463]) @ rot.quattmat_fn([0.88701801, 0, 0.46173483, 0]) @ rot.rytmat_fn(lf_qpos[1])  # LFJ3
lf_t_proximal = lf_t_knuckle @ rot.rxtmat_fn(lf_qpos[2])  # LFJ2
lf_t_middle = lf_t_proximal @ rot.ttmat_fn([0, 0, 0.045]) @ rot.rxtmat_fn(lf_qpos[3])  # LFJ1
lf_t_distal = lf_t_middle @ rot.ttmat_fn([0, 0, 0.025]) @ rot.rxtmat_fn(lf_qpos[4])  # LFJ0
lf_t_tip = lf_t_distal @ rot.ttmat_fn([0, 0, 0.026])
lftp_pos_fd_fn = cs.Function('lftp_pos_fd_fn', [palm_pose, lf_qpos], [lf_t_tip[0:3, -1]])

# ============================================================================
# Thumb - 5 DOFs: THJ4, THJ3, THJ2, THJ1, THJ0
# ============================================================================
th_qpos = cs.SX.sym('th_qpos', 5)
# Base at palm: xyz="0.034 -0.00858 0.029", rpy="-0.785398163397 0 -1.57079632679", axis="0 0 -1"
# rpy = [-45deg, 0deg, -90deg], quat = [0.65328148, -0.27059805, 0.27059805, -0.65328148]
th_t_base = rot.ttmat_fn(palm_pos) @ rot.quattmat_fn(palm_quat) @ rot.ttmat_fn([0.034, -0.00858, 0.029]) @ rot.quattmat_fn([0.65328148, -0.27059805, 0.27059805, -0.65328148])
th_t_base1 = th_t_base @ rot.rztmat_fn(-th_qpos[0])  # THJ4 (axis 0 0 -1)
# Proximal: xyz="0 0 0", axis="0 1 0"
th_t_proximal = th_t_base1 @ rot.rytmat_fn(th_qpos[1])  # THJ3
# Hub: xyz="0 0 0.038", axis="0 1 0"
th_t_hub = th_t_proximal @ rot.ttmat_fn([0, 0, 0.038]) @ rot.rytmat_fn(th_qpos[2])  # THJ2
# Middle: xyz="0 0 0", axis="1 0 0"
th_t_middle = th_t_hub @ rot.rxtmat_fn(th_qpos[3])  # THJ1
# Distal: xyz="0 0 0.032", axis="1 0 0"
th_t_distal = th_t_middle @ rot.ttmat_fn([0, 0, 0.032]) @ rot.rxtmat_fn(th_qpos[4])  # THJ0
# Tip: xyz="0 0 0.0275"
th_t_tip = th_t_distal @ rot.ttmat_fn([0, 0, 0.0275])
thtp_pos_fd_fn = cs.Function('thtp_pos_fd_fn', [palm_pose, th_qpos], [th_t_tip[0:3, -1]])
