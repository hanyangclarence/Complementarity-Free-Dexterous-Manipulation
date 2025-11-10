# Shadow Hand Pose Tracking

This example demonstrates MPC-based tracking of target hand poses using the Shadow Dexterous Hand.

## Overview

The task is to control the Shadow hand (palm + fingers) to reach a target pose consisting of:
- **Target palm pose**: position (3D) + orientation (quaternion)
- **Target finger configuration**: 22 joint angles (FF: 4, MF: 4, RF: 4, LF: 5, TH: 5)

Unlike object manipulation tasks, this example focuses on hand pose matching without requiring contact with objects.

## Files

- `params.py`: MPC parameters including:
  - System dimensions (22 finger DOFs + 6 palm DOFs)
  - Randomized target hand poses
  - Cost function weights for pose matching
  - Control bounds and state bounds

- `test_hand_pose.py`: Main test script that:
  - Initializes the Shadow hand environment
  - Sets random target hand poses
  - Runs MPC rollouts to reach targets
  - Visualizes target poses with transparent markers

- `env_shadow_hand_pose.xml`: MuJoCo XML scene with:
  - Shadow hand model
  - Target palm visualization (green transparent box + coordinate axes)
  - Target fingertip visualizations (5 orange spheres)
  - Dummy cube object (placed far away to avoid interference)

## Key Differences from Allegro Hand

| Feature | Allegro Hand | Shadow Hand |
|---------|--------------|-------------|
| Finger DOFs | 16 (4×4) | 22 (4+4+4+5+5) |
| Fingertips | 4 | 5 (includes little finger) |
| State dimension | 30 | 36 |
| Velocity dimension | 28 | 34 |
| Command dimension | 22 | 28 |

## Running the Example

```bash
python examples/mpc/shadow/hand_pose/test_hand_pose.py
```

The script will run 20 trials with randomized target hand poses.

## Success Criteria

A trial succeeds when the hand maintains the target pose for 20 consecutive timesteps with:
- Palm position error < 1 cm
- Palm quaternion error < 0.05
- Finger joint error < 0.1 rad

## Visualization

- **Green transparent box + axes**: Target palm pose
- **Orange spheres**: Target fingertip positions (computed via forward kinematics)
- **Yellow cube**: Dummy object (ignored in this task)
