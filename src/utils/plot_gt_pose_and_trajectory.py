import numpy as np
import matplotlib.pyplot as plt
import sys

# python plot_pose_and_trajectory.py <path_to_txt>

# Define the initial pose of the camera
# R = pr.matrix_from_axis_angle(pr.random_axis_angle())
# path_to_pose = "/home/udit/ws/cmu/sem4/vlr/project/SfmLearner-Pytorch/data/dump/2011_09_26_drive_0001_sync_02/poses.txt"
path_to_pose = sys.argv[1]
poses = np.loadtxt(path_to_pose, delimiter=" ").reshape(-1, 3, 4)

# Plot the trajectory and the pose of the camera
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory
x = poses[:, 0, 3].reshape(-1, 1)
y = poses[:, 1, 3].reshape(-1, 1)
z = poses[:, 2, 3].reshape(-1, 1)
ax.plot(x, y, z, color='c', label="gt_pose")

# Plot the pose of the camera at the last step of the trajectory
R = poses[:, :, 0:3]
u, v, w = R[:, :, 0], R[:, :, 1], R[:, :, 2]

for pose in poses:
    x, y, z = pose[:, -1]
    u, v, w = pose[:, 0], pose[:, 1], pose[:, 2]
    ax.quiver(x, y, z, u, v, w, length=1, normalize=True, color=['r', 'g', 'b', 'r', 'r', 'g', 'g', 'b', 'b'])

# Set the limits of the plot
max_range = np.max(np.abs(poses[:, :, 3]))
ax.set_xlim([-max_range, max_range])
ax.set_ylim([-max_range, max_range])
ax.set_zlim([-max_range, max_range])

ax.legend()

# Set the labels of the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
