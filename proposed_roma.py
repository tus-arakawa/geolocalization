from romatch import roma_outdoor
import torch
import cv2
import numpy as np
import argparse
from PIL import Image
from pathlib import Path
import pandas as pd
import csv
import os
import time

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', type=str, help='choose path') # test
    parser.add_argument('--img_name', type=str, help='choose image name') # top
    return parser.parse_args()

def estimate_similarity(kpts0, kpts1, thresh=3.0, conf=0.99999):
    H_est, inliers = cv2.estimateAffinePartial2D(kpts0, kpts1, ransacReprojThreshold=thresh, confidence=conf, method=cv2.RANSAC)
    if H_est is None:
        return np.eye(3) * 0, np.empty((0))
    H_est = np.concatenate([H_est, np.array([[0, 0, 1]])], axis=0) # 3 * 3
    return H_est, inliers

def warp_img(img0, img1, H, save_path, alpha=0.5):
    H = np.linalg.inv(H)
    img0 = np.copy(img0).astype(np.uint8)
    img1 = np.copy(img1).astype(np.uint8)
    img_warpped = cv2.warpAffine(np.array(img1), H[:2, :], (img0.shape[1], img0.shape[0]), flags=cv2.INTER_LINEAR)
    blended = cv2.addWeighted(img_warpped, alpha, img0, 1.0 - alpha, 0.0)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), blended)
    
def rot_points(pts, w, h, k):
    if k % 4 == 0:
        return pts.copy()
    k = k % 4
    x, y = pts[:,0], pts[:,1]
    if k == 1:
        xp, yp = y, (w - 1) - x
    elif k == 2:
        xp, yp = (w - 1) - x, (h - 1) - y
    else:  # k == 3
        xp, yp = (h - 1) - y, x
    return np.stack([xp, yp], axis=1)

def rot(img, angle):
    if angle == 0:   return img
    if angle == 90:  return img.transpose(Image.ROTATE_270)   # 時計回り
    if angle == 180: return img.transpose(Image.ROTATE_180)
    if angle == 270: return img.transpose(Image.ROTATE_90)
    raise ValueError(angle)
    
def fix_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
if __name__ == "__main__":
    fix_seed()
    args = parse_args()
    dir_path = args.path
    img_name = args.img_name
    dirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    dirs.sort()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = roma_outdoor(device=device)
    
    for d in dirs:
        img0_path = Path(args.path) / d / "images" / f"{d}.jpg"
        img1_path = Path(args.path) / d / "images" / f"{img_name}.png"
        img0 = cv2.imread(img0_path)
        img1 = cv2.imread(img1_path)
        h0, w0 = img0.shape[:2]
        h1, w1 = img1.shape[:2]
        
        best_angle = None
        best_inliers_sum = -1
        best_inliers = None
        best_H = None
        best_R = None
        best_mkpts0 = None
        best_mkpts1 = None
        best_mconf = None
        
        img0_pil = Image.open(img0_path).convert("RGB")
        img1_pil = Image.open(img1_path).convert("RGB")

        for angle in [0, 90, 180, 270]:
            img1_rot = rot(img1_pil, angle) 
            warp, certainty = model.match(img0_pil, img1_rot, device=device)
            matches, certainty = model.sample(warp, certainty)
            mkpts0, mkpts1 = model.to_pixel_coordinates(matches, h0, w0, h1, w1)
            mkpts0 = mkpts0.cpu().numpy()
            mkpts1 = mkpts1.cpu().numpy()
            mconf = certainty.cpu().numpy()
            H_est, inliers = estimate_similarity(mkpts0, mkpts1) # H_est: img0->img1
            
            if inliers is None:
                num_inliers = 0
            else:
                num_inliers = inliers.sum()
            if (best_inliers_sum < num_inliers):
                best_inliers_sum = num_inliers
                best_inliers = inliers.copy()
                best_angle = angle
                best_H = H_est.copy()
                best_mkpts0 = mkpts0.copy()
                best_mkpts1 = mkpts1.copy()
                best_mconf = mconf.copy()
            
        if best_angle == 0:
            best_R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        elif best_angle == 90:
            best_R = np.array([[0, -1, h1-1], [1, 0, 0], [0, 0, 1]])  
        elif best_angle == 180:
            best_R = np.array([[-1, 0, w1-1], [0, -1, h1-1], [0, 0, 1]])            
        else:
            best_R = np.array([[0, 1, 0], [-1, 0, w1-1], [0, 0, 1]])
        
        H = np.linalg.inv(best_R) @ best_H
        warp_img(img0, img1, H, save_path=Path(args.path)/d/img_name/"aligned.png")
        
        pts = rot_points(best_mkpts1, w1, h1, best_angle/90)
        
        inliers_reval = best_inliers.ravel()
        df = pd.DataFrame({
            "map_x": best_mkpts0[:, 0],
            "map_y": best_mkpts0[:, 1],
            "3dgs_x": pts[:, 0],
            "3dgs_y": pts[:, 1],
            "confidence": best_mconf,
            "inlier": inliers_reval
        })
        df.to_csv(Path(args.path)/d/img_name/"matches.csv", index=False)
        

        np.save(Path(args.path)/d/img_name/f"{d}.npy", H)
