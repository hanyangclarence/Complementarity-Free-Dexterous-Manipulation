# Target Hand Pose Visualization Implementation

This document explains how the target hand pose visualization works, following the same pattern as the cube target visualization in the original example.

## Overview

The visualization shows the target hand configuration that the MPC controller is trying to achieve:
- **Target palm pose**: Position + orientation (6 DOF)
- **Target fingertip positions**: Computed from target finger joints (4 fingertips)

## Implementation Pattern (Same as Cube Example)

### 1. XML Scene Definition (`env_allegro_hand_pose.xml`)

Visual markers are defined as static MuJoCo bodies:

```xml
<!-- Target palm visualization with coordinate frame -->
<body name="target_palm" pos="0.0 0.0 0.0">
    <geom name="target_palm_box" type="box" size="0.03 0.06 0.02"
          conaffinity="0" contype="0" rgba="0 1 0 0.3"/>
    <geom name="target_palm_x" type="cylinder" ... rgba="1 0 0 0.7"/>  <!-- Red X-axis -->
    <geom name="target_palm_y" type="cylinder" ... rgba="0 1 0 0.7"/>  <!-- Green Y-axis -->
    <geom name="target_palm_z" type="cylinder" ... rgba="0 0 1 0.7"/>  <!-- Blue Z-axis -->
</body>

<!-- Target fingertip markers -->
<body name="target_fftip" pos="0.0 0.0 0.1">
    <geom name="target_fftip_geom" type="sphere" size="0.008"
          conaffinity="0" contype="0" rgba="1 0.5 0 0.6"/>
</body>
<!-- ... similar for mftip, rftip, thtip -->
```

**Key properties:**
- `conaffinity="0" contype="0"`: Non-colliding (visual only)
- Semi-transparent colors: Easy to see through
- Initial positions: Will be updated dynamically

### 2. Visualization Update Method (`allegro_env.py`)

Added `set_target_hand_pose()` method (similar to `set_goal()` for cube):

```python
def set_target_hand_pose(self, target_palm_pos=None, target_palm_quat=None, target_finger_qpos=None):
    # Set target palm pose
    if target_palm_pos is not None and target_palm_quat is not None:
        self.model_.body('target_palm').pos = target_palm_pos
        self.model_.body('target_palm').quat = target_palm_quat

    # Compute and set target fingertip positions using forward kinematics
    if target_finger_qpos is not None:
        target_palm_pose = np.hstack([target_palm_pos, target_palm_quat])

        # Forward kinematics for each finger
        target_fftip_pos = np.array(self.fftp_pos_fd_fn(target_palm_pose, ff_qpos)).flatten()
        # ... similar for other fingers

        # Update visualization positions
        self.model_.body('target_fftip').pos = target_fftip_pos
        # ... similar for other fingertips

    # Apply changes to simulation
    mujoco.mj_forward(self.model_, self.data_)
```

**Error handling:**
- Uses `try-except KeyError` to gracefully handle missing bodies
- Allows the same `MjSimulator` class to work with different XML files

### 3. Initialization in Test Script (`test_hand_pose.py`)

Called once after environment initialization:

```python
env = MjSimulator(param)

# Set target hand pose visualization
env.set_target_hand_pose(
    target_palm_pos=param.target_palm_qpos_[0:3],
    target_palm_quat=param.target_palm_qpos_[3:7],
    target_finger_qpos=param.target_robot_qpos_
)
```

## Comparison with Cube Visualization

| Aspect | Cube Target | Hand Pose Target |
|--------|-------------|------------------|
| **XML bodies** | 1 body (`goal`) | 5 bodies (`target_palm` + 4 fingertips) |
| **Update method** | `set_goal()` | `set_target_hand_pose()` |
| **Computation** | Direct (given pos + quat) | FK required for fingertips |
| **Visualization** | Single box | Palm box + axes + 4 spheres |
| **Update frequency** | Once (static target) | Once (static target) |

## Visual Elements

### Target Palm
- **Box**: Approximate palm size and shape (green, 30% opacity)
- **RGB Axes**: Show target orientation clearly
  - Red cylinder: +X direction (3cm length)
  - Green cylinder: +Y direction (3cm length)
  - Blue cylinder: +Z direction (3cm length)

### Target Fingertips
- **Spheres**: 8mm diameter (orange, 60% opacity)
- **Positions**: Computed via forward kinematics
  - Uses the same FK functions (`fftp_pos_fd_fn`, etc.) as the MPC cost function
  - Ensures consistency between visualization and optimization

## Benefits

1. **Visual feedback**: User can see exactly what pose the controller is trying to achieve
2. **Debugging**: Easy to verify if target pose is feasible/reasonable
3. **Progress tracking**: Can visually assess how close the hand is to the target
4. **Consistency**: Uses same FK as cost function, so visualization matches optimization target

## Extension for Future Work

When adding cube manipulation back:
- The target palm/fingertip visualization shows hand goals
- The existing `goal` body (magenta cube) shows object goals
- Both can be displayed simultaneously for multi-objective tasks
