import torch
import torch.nn.functional as F

def quat_to_rotMat(Q):
    """
    Quaternion to rotation matrix
    """
    # normalize Q - (B x 4)
    norm_Q = Q / Q.norm(p=2, dim=1, keepdim=True)
    # Extract the values from Q
    b, _ = Q.shape
    qx, qy, qz, qw = norm_Q[:, 0], norm_Q[:, 1], norm_Q[:, 2], norm_Q[:, 3]

    x2, y2, z2, w2 = qx**2, qy**2, qz**2, qw**2
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz

    rot_mat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(b, 3, 3)
                            
    return rot_mat

def inverse_warp(imgs, pose, depth, intrinsics):
        '''
        imgs - (B x C x H x W)
        pose - (B x 7) - translation and quaternion
        depth - (B x 1 x H x W)
        intrinsics - (3 x 3)
        '''
        # generate pixel indices
        b, _, h, w = imgs.shape
        x = torch.arange(0, h, dtype= torch.float)
        y = torch.arange(0, w, dtype= torch.float)
        x_inds, y_inds =  torch.meshgrid(x, y, indexing='ij')
        # homogeneous pixel indices
        indices = torch.stack((x_inds, y_inds, torch.ones_like(x_inds))).unsqueeze(0) # (1 x 3 x H x W)

        # # convert to rot mat and get transformation matrix
        translation, quat = pose[:, :3], pose[:, 3:]
        translation = translation.unsqueeze(-1) # (B x 3 x 1)
        rot_mat = quat_to_rotMat(quat)
        transform_mat = torch.cat((rot_mat, translation), dim=2) # (B x 3 x 4)
        # transform_mat = pose.reshape(b, 3, 4) # (B x 3 x 4)

        # convert to 3D coords
        intrinsics_inv = torch.linalg.inv(intrinsics).unsqueeze(0) # (1 x 3 x 3)
        indices = indices.flatten(start_dim=-2).to(intrinsics.device) # (1 x 3 x H*W)
        cam_coords = (intrinsics_inv @ indices).reshape(1, 3, h, w).repeat(b, 1, 1, 1) # (B x 3 x H x W)
        # multiply with depth
        depth_cam_coords = cam_coords * depth # (B x 3 x H x W)
        # homogenize
        depth_cam_coords_homo = torch.cat((depth_cam_coords, torch.ones((b, 1, h, w), device=intrinsics.device)), dim=1) # (B x 4 x H x W)
        depth_cam_coords_homo = depth_cam_coords_homo.flatten(start_dim=-2) # (B x 4 x H*W)

        # warp to source frame
        proj_tgt_to_src = intrinsics.unsqueeze(0) @ transform_mat # (B x 3 x 4)
        warped_pixels = proj_tgt_to_src @ depth_cam_coords_homo # (B x 3 x H*W)
        # normalize and un-homo
        warped_pixels = warped_pixels[:, :-1, :] / warped_pixels[:, -1, :].clamp(min=1e-3).unsqueeze(1) # (B x 2 x H*W)
        grid_x = warped_pixels[:, 0] # (B x H*W)
        grid_y = warped_pixels[:, 1] # (B x H*W)

        warped_pixels = warped_pixels.reshape(b, 2, h, w) # (B x 2 x H x W)

        grid_x = ((2*grid_x) / (h - 1)) - 1
        grid_y = ((2*grid_y) / (w - 1)) - 1
        grid = torch.stack((grid_x, grid_y), dim=2).reshape(b, h, w, 2) # (B x H x W x 2)

        warped_imgs = F.grid_sample(imgs, grid, padding_mode='zeros', align_corners=True)
        valid_pixels = grid.abs().max(dim=-1)[0] <= 1

        return warped_imgs, valid_pixels