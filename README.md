# Geolocalization 
"Geolocalization of Unmanned Aerial Vehicle Images and Mapping onto Satellite Images Utilizing 3D Gaussian Splatting"

### Directory Structure
```text
project/
├─ test/
│  ├─ 0000/
│  │  ├─ COLMAP/
│  │  │  ├─ cameras.txt
│  │  │  ├─ images.txt
│  │  │  └─ points3D.txt
│  │  ├─ images/
│  │  │  ├─ 0000.jpg          # wide-area satellite image 
│  │  │  └─ top.png           # 3DGS-rendered image
│  │  └─ sparse/0/            # after running RANSAC.py
│  │     ├─ cameras.txt
│  │     ├─ images.txt
│  │     └─ points3D.txt
│  │
│  ├─ 0001/
│  │
```
