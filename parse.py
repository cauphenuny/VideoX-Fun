import numpy as np
import json

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
    instristics = "0 0.532139961 0.946026558 0.5 0.5 0 0"
    with open(f"./asset/recam{(int(cam_type)):02d}.txt", 'w') as f:
        f.write("\n")
        for camera_params in c2ws:
            extrinsic = camera_params[:3].flatten()
            line = f"{instristics} " + " ".join(map(str, extrinsic)) + "\n"
            f.write(line)

if __name__ == "__main__":
    for cam in range(10):
        transform_camera_params(81, cam + 1)