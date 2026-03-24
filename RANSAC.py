import os
import argparse
import numpy as np
import pandas as pd
import open3d as o3d
import shutil
from scipy.spatial.transform import Rotation as R

def load_points3D(path): # COLMAP points3D.txt
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    result = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#"):
            i += 1
            continue
        points = line.split()
        point3d_id = int(points[0])
        x, y, z = map(float, points[1:4])
        r, g, b = map(int, points[4:7])
        error = float(points[7])
        track = points[8:]
        for j in range(0, len(track), 2):
            image_id = int(track[j])
            point2d_id = int(track[j+1])
            result.append({
                "point3d_id" : point3d_id,
                "x" : x,
                "y" : y,
                "z" : z,
                "r" : r,
                "g" : g,
                "b" : b,
                "error" : error,
                "image_id" : image_id,
                "point2d_id" : point2d_id
            })
        i += 1
    return pd.DataFrame(result)

def save_points3D(path, df): # COLMAP points3D.txt
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        grouped = df.groupby('point3d_id')
        for point_id, group in grouped:
            x, y, z = group.iloc[0][['x_aligned', 'y_aligned', 'z_aligned']]
            r = group.iloc[0]['r']
            g = group.iloc[0]['g']
            b = group.iloc[0]['b']
            error = group.iloc[0]['error']
            track_list = []
            for _, row in group.iterrows():
                track_list.append(f"{int(row['image_id'])} {int(row['point2d_id'])}")
            track_str = ' '.join(track_list)
            line = f"{int(point_id)} {x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)} {error:.6f} {track_str}\n"
            f.write(line)

def load_images(path): # COLMAP images.txt
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    result = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#"):
            i += 1
            continue
        image = line.split()
        image_id = int(image[0])
        qw, qx, qy, qz = map(float, image[1:5])
        tx, ty, tz = map(float, image[5:8])
        camera_id = image[-2]
        image_name = image[-1]
        if i+1 < len(lines):
            points2d = lines[i+1].strip().split()
            for j in range(0, len(points2d), 3):
                x = float(points2d[j])
                y = float(points2d[j+1])
                point3d_id = int(points2d[j+2])
                if point3d_id != -1:
                    result.append({
                        "image_id" : image_id,
                        "image_name" : image_name,
                        "image_x" : x,
                        "image_y" : y,
                        "point3d_id" : point3d_id, 
                        "point2d_id" : j // 3, 
                        "qw" : qw,
                        "qx" : qx,
                        "qy" : qy,
                        "qz" : qz,
                        "tx" : tx,
                        "ty" : ty,
                        "tz" : tz,
                        "qx" : qx,
                        "camera_id" : camera_id      
                    })
        i += 2
    return pd.DataFrame(result)

def save_images(path, df): # COLMAP images.txt
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for image_id in df["image_id"].unique():
            row = df[df["image_id"] == image_id].iloc[0]
            qw, qx, qy, qz = row["qw_aligned"], row["qx_aligned"], row["qy_aligned"], row["qz_aligned"]
            # tx, ty, tz = row["tx_aligned"], row["ty_aligned"], row["tz_aligned"]
            tx, ty, tz = row["tx"], row["ty"], row["tz"]
            camera_id = row["camera_id"]
            image_name = row["image_name"]
            line1 = f"{int(image_id)} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {int(camera_id)} {image_name}\n"
            f.write(line1)
            points = df[df["image_id"] == image_id]
            line2 = " ".join(f"{x} {y} {int(pid)}" for x, y, pid in zip(points["image_x"], points["image_y"], points["point3d_id"]))
            f.write(line2 + "\n") 

def align_normal_to_target(normal, target=np.array([0,1,0])):
    normal = normal / np.linalg.norm(normal)
    target = target / np.linalg.norm(target)
    v = np.cross(normal, target)
    c = np.dot(normal, target)
    s = np.linalg.norm(v)
    if s == 0:
        return np.eye(3)
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s**2))
    return R

def transform(row, R_mat):
    r_cam = R.from_quat([row['qx'], row['qy'], row['qz'], row['qw']]).as_matrix()
    r_new = (R_mat @ r_cam.T).T
    quat_new = R.from_matrix(r_new).as_quat()  # [qx, qy, qz, qw] 
    # t = np.array([row['tx'], row['ty'], row['tz']])
    # t_new = (R_mat @ r_cam.T).T @ (R_mat @ r_cam.T) @ t
    return pd.Series({
        'qw_aligned': quat_new[3],
        'qx_aligned': quat_new[0],
        'qy_aligned': quat_new[1],
        'qz_aligned': quat_new[2]
        # 'tx_aligned': t_new[0],
        # 'ty_aligned': t_new[1],
        # 'tz_aligned': t_new[2]
    })

def main():
    parser = argparse.ArgumentParser(description="COLMAP rotation COLMAP/*.txt -> sparse/0/*.txt")
    parser.add_argument("--input_path", type=str, required=True, help="path")
    args = parser.parse_args()

    points3D_input = os.path.join(args.input_path, "COLMAP/points3D.txt")
    points3D_output = os.path.join(args.input_path, "sparse/0/points3D.txt")
    
    images_input = os.path.join(args.input_path, "COLMAP/images.txt")
    images_output = os.path.join(args.input_path, "sparse/0/images.txt")
    
    cameras_input = os.path.join(args.input_path, "COLMAP/cameras.txt")
    cameras_output = os.path.join(args.input_path, "sparse/0/cameras.txt")
    
    if not os.path.exists(points3D_input):
        print(f"Error: {points3D_input} does not exist.")
        return
    
    if not os.path.exists(images_input):
        print(f"Error: {images_input} does not exist.")
        return  
    
    if not os.path.exists(cameras_input):
        print(f"Error: {cameras_input} does not exist.")
        return 
    
    os.makedirs(os.path.dirname(points3D_output), exist_ok=True)
    
    shutil.copy2(cameras_input, cameras_output)
    
    df_points3D = load_points3D(points3D_input)
    df_images = load_images(images_input)
    
    points = df_points3D[["x", "y", "z"]].drop_duplicates().values
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"{a}x + {b}y + {c}z + {d} = 0")
    
    normal = np.array([a, b, c])
    R_mat = align_normal_to_target(normal, target=np.array([0,1,0]))
    # print(R_mat)
    
    points =  df_points3D[["x", "y", "z"]].values
    points_aligned = (R_mat @ points.T).T
    df_points3D['x_aligned'] = points_aligned[:, 0]
    df_points3D['y_aligned'] = points_aligned[:, 1]
    df_points3D['z_aligned'] = points_aligned[:, 2]
    
    save_points3D(points3D_output, df_points3D)
    
    df_quat_pos = df_images.apply(transform, axis=1, args=(R_mat,))
    df_images = pd.concat([df_images, df_quat_pos], axis=1)
    
    save_images(images_output, df_images)
    
    print(f"{args.input_path} Done.")

if __name__ == '__main__':
    main()