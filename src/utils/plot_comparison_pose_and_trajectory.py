import numpy as np
import matplotlib.pyplot as plt
import sys

# python plot_pose_and_trajectory.py <path_to_gt_txt> <path_to_pred_txt>
def pose_to_matrix(pose):
    # Convert pose to homogeneous transformation matrix
    xyz, quat = pose[:3], pose[3:]
    qx, qy, qz, qw = quat
    rotation_matrix = np.array([[1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                                [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
                                [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]])
    translation_vector = np.array([xyz]).T
    homogeneous_matrix = np.identity(4)
    homogeneous_matrix[:3, :3] = rotation_matrix
    homogeneous_matrix[:3, 3:] = translation_vector
    return homogeneous_matrix

path_to_pose = sys.argv[1]
poses = np.loadtxt(path_to_pose, delimiter=" ").reshape(-1, 3, 4)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory
x = poses[:, 0, 3].reshape(-1, 1)
y = poses[:, 1, 3].reshape(-1, 1)
z = poses[:, 2, 3].reshape(-1, 1)
ax.plot(x, y, z, color='c', label="gt_pose", alpha = 1)

# Plot the pose of the camera at the last step of the trajectory
R = poses[:, :, 0:3]
u, v, w = R[:, :, 0], R[:, :, 1], R[:, :, 2]

for pose in poses:
    x, y, z = pose[:, -1]
    u, v, w = pose[:, 0], pose[:, 1], pose[:, 2]
    ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True, color=['r', 'g', 'b', 'r', 'r', 'g', 'g', 'b', 'b'])

path_to_pose = sys.argv[2]
poses_quat = np.loadtxt(path_to_pose, delimiter=" ").reshape(-1, 7)

poses = [pose_to_matrix(poses_quat[-1])]
for pose_quat in poses_quat:
    poses.append(poses[-1] @ pose_to_matrix(pose_quat))

poses = np.stack(poses)

# Plot the trajectory
x = poses[:, 0, 3].reshape(-1, 1)
y= poses[:, 1, 3].reshape(-1, 1)
z = poses[:, 2, 3].reshape(-1, 1)
ax.plot(x, y, z, color='r', label="pred_path", alpha = 1)

# Plot the pose of the camera at the last step of the trajectory
R = poses[:, :, 0:3]
u, v, w = R[:, :, 0], R[:, :, 1], R[:, :, 2]

for pose in poses:
    x, y, z = pose[0:3, -1]
    u, v, w = pose[0:3, 0], pose[0:3, 1], pose[0:3, 2]
    ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True, color=['r', 'g', 'b', 'r', 'r', 'g', 'g', 'b', 'b'])

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