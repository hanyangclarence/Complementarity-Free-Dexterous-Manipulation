# Shadow Hand MPC for Cube Reorientation

This directory contains the implementation of Model Predictive Control (MPC) for in-hand object reorientation using the Shadow Dexterous Hand with 22 finger DOFs.

## Overview

The Shadow hand has been integrated into the complementarity-free manipulation framework, replacing the Allegro hand with a more complex dexterous hand.

## Key Differences from Allegro Hand

### Joint Structure
- **Allegro Hand**: 16 finger DOFs (4 fingers × 4 joints each)
- **Shadow Hand**: 22 finger DOFs
  - First Finger (FF): 4 joints
  - Middle Finger (MF): 4 joints
  - Ring Finger (RF): 4 joints
  - Little Finger (LF): 5 joints (includes metacarpal)
  - Thumb (TH): 5 joints

### System Dimensions
| Parameter | Allegro | Shadow | Change |
|-----------|---------|--------|--------|
| Finger DOFs | 16 | 22 | +6 |
| Total DOFs (with palm) | 22 | 28 | +6 |
| `n_robot_qpos_` | 16 | 22 | +6 |
| `n_qpos_` | 30 | 36 | +6 |
| `n_qvel_` | 28 | 34 | +6 |
| `n_cmd_` | 22 | 28 | +6 |

## Files Created

### Core Files
1. **envs/xmls/shadow_hand_right.xml** - Shadow hand MuJoCo model with 22 finger DOFs
2. **envs/xmls/env_shadow_cube.xml** - Complete environment (Shadow hand + cube)
3. **envs/shadow_env.py** - Shadow hand environment wrapper
4. **envs/shadow_fkin.py** - Forward kinematics for Shadow hand (5 fingers)
5. **contact/shadow_collision_detection.py** - Contact detection for Shadow hand
6. **examples/mpc/shadow/cube/params.py** - MPC parameters for Shadow hand
7. **examples/mpc/shadow/cube/test.py** - Test script for Shadow hand MPC

### Forward Kinematics
The forward kinematics for the Shadow hand are implemented using CasADi symbolic expressions, similar to the Allegro hand but with the correct Shadow hand link lengths and joint configurations:

- **First Finger (FF)**: 4-joint chain from palm
- **Middle Finger (MF)**: 4-joint chain from palm
- **Ring Finger (RF)**: 4-joint chain from palm
- **Little Finger (LF)**: 5-joint chain (metacarpal + 4 finger joints)
- **Thumb (TH)**: 5-joint complex chain with different base orientation

### Joint Order
Shadow hand joint order (22 DOFs):
```
FF: ffj3, ffj2, ffj1, ffj0  (4)
MF: mfj3, mfj2, mfj1, mfj0  (4)
RF: rfj3, rfj2, rfj1, rfj0  (4)
LF: lfj4, lfj3, lfj2, lfj1, lfj0  (5)
TH: thj4, thj3, thj2, thj1, thj0  (5)
```

## Running the Code

To run the Shadow hand MPC test:

```bash
python examples/mpc/shadow/cube/test.py
```

## Parameters

The MPC parameters have been adjusted for the Shadow hand:
- **Initial joint configuration**: Adjusted for Shadow hand geometry
- **Control bounds**: Same as Allegro (±0.2 rad per step for fingers)
- **Cost weights**: Similar to Allegro but account for 5 fingertips
- **MPC horizon**: 4 steps (same as Allegro)

## Notes

1. The Shadow hand has **5 fingertips** (vs 4 for Allegro), so the cost function includes all 5 contact points.
2. The little finger and thumb each have 5 DOFs, providing additional dexterity.
3. The forward kinematics are more complex due to the metacarpal joint in the little finger and the complex thumb base.
4. Initial joint positions have been tuned for a stable grasp configuration.

## Verification

The MuJoCo model has been verified:
- **Model loads successfully**: ✓
- **Number of DOFs**: 34 (6 palm + 22 fingers + 6 object)
- **Number of actuators**: 28 (6 palm + 22 fingers)

## Future Work

- Tune MPC weights for Shadow hand characteristics
- Adjust initial joint positions for better grasping
- Test with different objects and manipulation tasks
- Compare performance with Allegro hand baseline
