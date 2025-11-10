# Hand Pose Matching with MPC

This directory contains an MPC-based controller for matching target hand poses with an Allegro hand.

## Overview

Unlike the cube manipulation example (`examples/mpc/allegro/cube/`), this controller focuses on achieving target hand configurations rather than manipulating objects. The hand is controlled to reach randomized target poses including:
- **Palm pose**: 6 DOF (3D position + 3D orientation)
- **Finger joints**: 16 DOF (4 fingers × 4 joints each)

The cube object remains in the scene but is not actively manipulated.

## Key Features

1. **Randomized Target Generation**: Each trial generates a random target hand pose within feasible ranges
2. **Full Hand Control**: Controls all 22 DOF (palm 6 + fingers 16)
3. **Contact-Aware**: Retains contact detection infrastructure for future extensions
4. **Pose Matching Cost**: Optimizes for hand pose alignment rather than object manipulation

## Files

- `params.py`: Parameter configuration and cost functions for hand pose matching
  - `HandPoseMPCParams`: Defines target hand poses, control bounds, and MPC parameters
  - `init_cost_fns()`: Cost functions for palm position/orientation and finger joint matching

- `test_hand_pose.py`: Main test script for running hand pose matching trials
  - Runs multiple trials with randomized targets
  - Reports success/failure based on pose matching thresholds

- `../../../planning/mpc_hand_pose.py`: Custom MPC planner for hand pose matching
  - Extends the explicit MPC framework
  - Handles target hand pose (palm + fingers) as optimization parameters

- `../../../envs/xmls/env_allegro_hand_pose.xml`: MuJoCo scene with target hand visualization
  - Includes target palm visualization (semi-transparent green box + RGB axes)
  - Includes target fingertip markers (orange spheres)

- `../../../envs/allegro_env.py`: Added `set_target_hand_pose()` method
  - Visualizes target palm pose and computes target fingertip positions via FK

## Cost Function

The MPC optimization uses:

### Path Cost (control regularization):
- Palm linear control: weight = 0.05
- Palm angular control: weight = 0.05
- Finger control: weight = 0.1

### Final Cost (hand pose matching):
- Palm position error: weight = 100.0
- Palm quaternion error: weight = 50.0
- Finger joint error: weight = 10.0

## Target Pose Randomization

- **Palm position**: ±3cm in x,y, ±2cm in z from origin
- **Palm orientation**: ±11.5° roll/pitch, ±17° yaw
- **Finger joints**: ±0.5 rad variation from initial, clipped to [-0.3, 1.8] rad

## Success Criteria

A trial is considered successful if the hand maintains target pose within thresholds for 20 consecutive timesteps:
- Palm position error < 1cm
- Palm quaternion error < 0.05
- Finger joint error < 0.1 rad

## Visualization

The viewer shows:
- **Current hand**: The actual Allegro hand (white/black)
- **Target palm**: Semi-transparent green box with RGB coordinate axes
  - Red axis: X direction
  - Green axis: Y direction
  - Blue axis: Z direction
- **Target fingertips**: Orange semi-transparent spheres (4 total)
  - Computed via forward kinematics from target finger joint positions
- **Cube**: Yellow cube object (stationary, not being manipulated)

## Usage

```bash
python examples/mpc/allegro/hand_pose/test_hand_pose.py
```

## Comparison with Cube Manipulation

| Aspect | Cube Manipulation | Hand Pose Matching |
|--------|------------------|-------------------|
| Optimization target | Object pose | Hand configuration |
| Cost function | Object position/orientation + grasp closure | Palm pose + finger joints |
| Contact usage | Essential for manipulation | Available but not primary |
| Complexity | Higher (indirect control) | Lower (direct control) |

## Future Extensions

The contact detection infrastructure is preserved to enable:
- Collision avoidance with the cube
- Future multi-objective tasks (pose matching + object manipulation)
- Contact-aware motion planning
