import numpy as np
import json
import torch

def get_relative_pose(cam_params):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]

        cam_to_origin = 0
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -cam_to_origin],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses

class Camera(object):
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)

def parse_matrix(matrix_str):
    rows = matrix_str.strip().split('] [')
    matrix = []
    for row in rows:
        row = row.replace('[', '').replace(']', '')
        matrix.append(list(map(float, row.split())))
    return np.array(matrix)

def transform_camera_params(num_frames, cam_type):
    tgt_camera_path = "./asset/camera_extrinsics.json"
    with open(tgt_camera_path, 'r') as file:
        cam_data = json.load(file)

    cam_idx = list(range(num_frames))
    traj = [parse_matrix(cam_data[f"frame{idx}"][f"cam{int(cam_type):02d}"]) for idx in cam_idx]
    traj = np.stack(traj).transpose(0, 2, 1)
    c2ws = []
    for c2w in traj:
        c2w = c2w[:, [1, 2, 0, 3]]
        c2w[:3, 1] *= -1.
        c2w[:3, 3] /= 100
        c2ws.append(c2w)
    tgt_cam_params = [Camera(cam_param) for cam_param in c2ws]
    relative_poses = []
    for i in range(len(tgt_cam_params)):
        relative_pose = get_relative_pose([tgt_cam_params[0], tgt_cam_params[i]])
        relative_poses.append(torch.as_tensor(relative_pose)[:,:3,:][1])
    pose_embedding = torch.stack(relative_poses, dim=0)  # 81x3x4
    instristics = "0 0.532139961 0.946026558 0.5 0.5 0 0"
    with open(f"./asset/recam{(int(cam_type)):02d}.txt", 'w') as f:
        f.write("\n")
        for camera_params in pose_embedding:
            extrinsic = camera_params.flatten().cpu().numpy().tolist()
            line = f"{instristics} " + " ".join(map(lambda x: f"{x:.10f}", extrinsic)) + "\n"
            f.write(line)

if __name__ == "__main__":
    for cam in range(10):
        transform_camera_params(81, cam + 1)