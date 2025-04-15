import numpy as np
import matplotlib.pyplot as plt

# Load data
states = np.load("./DL25SP/train/states.npy", mmap_mode='r')
actions = np.load("./DL25SP/train/actions.npy")

# Select a random trajectory
traj_idx = np.random.randint(len(states))
trajectory = states[traj_idx]  # (L, 2, 64, 64)  L: trajectory_length
trajectory_actions = actions[traj_idx]  # (L-1, 2)

# Extract agent positions over time
agent_frames = trajectory[:, 0]  # (L, 64, 64)
wall_frame = trajectory[0, 1]    # (L, 64, 64)
for l in range(1, trajectory.shape[0]):
    assert np.array_equal(trajectory[l, 1], wall_frame), "Wall frame should be the same for all time steps"

positions = []
positions_action = []
for t in range(agent_frames.shape[0]):
    print(f"------------------time step {t}-------------------")

    agent_mask = agent_frames[t]
    pos = np.argwhere(agent_mask > 0)
    assert pos.size > 0, "There should be at least one agent position per time step"
    if pos.size > 0:
        yx = pos.mean(axis=0)  # Y, X
        xy = yx[::-1]
        positions.append(xy)  # X, Y
        # positions.append(pos[0][::-1])
        print(f"agent detected at: {xy}")

    if t == 0:
        positions_action.append(xy)

    if t != 0:
        print("action difference:", xy - positions[t - 1] - trajectory_actions[t - 1])
        print("action position at:", trajectory_actions[t - 1] + positions[t - 1])
        positions_action.append(trajectory_actions[t - 1] + positions[t - 1])


positions = np.array(positions)
positions_action = np.array(positions_action)

# Plot
fig, ax = plt.subplots(figsize=(6, 6))

# Plot wall image (in black)
ax.imshow(1 - wall_frame, cmap="gray", interpolation="nearest")

# Plot trajectory path
ax.plot(positions[:, 0], positions[:, 1], 'o--', color='dodgerblue', markersize=3, linewidth=1)
ax.plot(positions_action[:, 0], positions_action[:, 1], 'o--', color='green', markersize=3, linewidth=1)

# Mark last agent position (last timestep) in red
ax.scatter(positions[-1, 0], positions[-1, 1], color="red", s=30, label="Agent")
ax.scatter(positions_action[-1, 0], positions_action[-1, 1], color="red", s=30, label="Action")

# Mark first agent position (first timestep) in black
ax.scatter(positions[0, 0], positions[0, 1], color="black", s=30, label="Start")
ax.scatter(positions_action[0, 0], positions_action[0, 1], color="black", s=30, label="Action")

# Style: remove grid, invert y, set aspect
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect("equal")
ax.invert_yaxis()
ax.set_facecolor("white")
plt.tight_layout()
plt.show()



