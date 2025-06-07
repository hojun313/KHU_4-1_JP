import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d
import numpy as np
import math
import os
import cv2
from scipy.interpolate import griddata
from scipy import fftpack


def find_marker(gray):
    mask = cv2.inRange(gray, 0, 70)
    return mask

def dilate(img, ksize=5, iter=1):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(img, kernel, iterations=iter)

def erode(img, ksize=5):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.erode(img, kernel, iterations=1)

def matching_rows(A,B):
    matches=[i for i in range(B.shape[0]) if np.any(np.all(A==B[i],axis=1))]
    if len(matches)==0:
        return B[matches]
    return np.unique(B[matches],axis=0)

def interpolate_gradients(gx, gy, img, cm, markermask):

    cmcm = np.zeros(img.shape[:2])
    ind1 = np.vstack(np.where(cm)).T
    ind2 = np.vstack(np.where(markermask)).T
    ind2not = np.vstack(np.where(~markermask)).T
    ind3 = matching_rows(ind1, ind2)
    cmcm[(ind3[:, 0], ind3[:, 1])] = 1.
    ind4 = ind1[np.all(np.any((ind1 - ind3[:, None]), axis=2), axis=0)]

    try:
        points_known = ind2
        values_gx_known = gx[(ind2[:, 0], ind2[:, 1])]
        values_gy_known = gy[(ind2[:, 0], ind2[:, 1])]
        points_interp = ind2not

        if len(points_known) > 0 and len(points_interp) > 0:
            gx_interpolated_values = griddata(points_known, values_gx_known, points_interp, method='nearest')
            gy_interpolated_values = griddata(points_known, values_gy_known, points_interp, method='nearest')

            gx_out = gx.copy()
            gy_out = gy.copy()
            gx_out[(ind2not[:, 0], ind2not[:, 1])] = gx_interpolated_values
            gy_out[(ind2not[:, 0], ind2not[:, 1])] = gy_interpolated_values
        else:
            gx_out = gx.copy()
            gy_out = gy.copy()

    except Exception as e:
        print(f"interpolate_gradients 오류: {e}")
        gx_out = gx.copy()
        gy_out = gy.copy()

    return gx_out, gy_out


def interpolate_grad(img, mask):
    mask_around = (dilate(mask, ksize=3, iter=2) > 0) & ~(mask != 0)
    mask_around = mask_around.astype(bool)

    x, y = np.arange(img.shape[0]), np.arange(img.shape[1])
    yy, xx = np.meshgrid(y, x)

    mask_to_interp = mask_around 
    points_known = np.vstack([xx[~mask.astype(bool)], yy[~mask.astype(bool)]]).T
    values_known = img[~mask.astype(bool)]
    points_interp = np.vstack([xx[mask.astype(bool)], yy[mask.astype(bool)]]).T

    if len(points_known) > 0 and len(points_interp) > 0:
        method = "nearest"
        x_interp_values = griddata(points_known, values_known, points_interp, method=method)

        x_interp_values[x_interp_values != x_interp_values] = 0.0

        ret = img.copy()
        ret[mask.astype(bool)] = x_interp_values
    else:
        ret = img.copy()

    return ret

def demark(gx, gy, markermask):
    gx_interp = interpolate_grad(gx.copy(), markermask)
    gy_interp = interpolate_grad(gy.copy(), markermask)
    return gx_interp, gy_interp

def get_features(img,pixels,features,imgw,imgh):
    features[:,3], features[:,4] = pixels[:,0] / imgh, pixels[:,1] / imgw
    for k in range(len(pixels)):
        i,j = pixels[k]
        rgb = img[i, j] / 255.
        features[k,:3] = rgb


def poisson_dct_neumaan(gx,gy):

    gxx = 1 * (gx[:, (list(range(1, gx.shape[1])) + [gx.shape[1] - 1])] - gx[:, ([0] + list(range(gx.shape[1] - 1)))])
    gyy = 1 * (gy[(list(range(1,gy.shape[0]))+[gy.shape[0]-1]), :] - gy[([0]+list(range(gy.shape[0]-1))), :])

    f = gxx + gyy

    b = np.zeros(gx.shape)
    b[0,1:-2] = -gy[0,1:-2]
    b[-1,1:-2] = gy[-1,1:-2]
    b[1:-2,0] = -gx[1:-2,0]
    b[1:-2,-1] = gx[1:-2,-1]
    b[0,0] = (1/np.sqrt(2))*(-gy[0,0] - gx[0,0])
    b[0,-1] = (1/np.sqrt(2))*(-gy[0,-1] + gx[0,-1])
    b[-1,-1] = (1/np.sqrt(2))*(gy[-1,-1] + gx[-1,-1])
    b[-1,0] = (1/np.sqrt(2))*(gy[-1,0]-gx[-1,0])

    f[0,1:-2] = f[0,1:-2] - b[0,1:-2]
    f[-1,1:-2] = f[-1,1:-2] - b[-1,1:-2]
    f[1:-2,0] = f[1:-2,0] - b[1:-2,0]
    f[1:-2,-1] = f[1:-2,-1] - b[1:-2,-1]

    f[0,-1] = f[0,-1] - np.sqrt(2) * b[0,-1]
    f[-1,-1] = f[-1,-1] - np.sqrt(2) * b[-1,-1]
    f[-1,0] = f[-1,0] - np.sqrt(2) * b[-1,0]
    f[0,0] = f[0,0] - np.sqrt(2) * b[0,0]

    tt = fftpack.dct(f, norm='ortho')
    fcos = fftpack.dct(tt.T, norm='ortho').T

    x_freq_indices = np.arange(1, f.shape[1] + 1)
    y_freq_indices = np.arange(1, f.shape[0] + 1)
    X_freq_grid, Y_freq_grid = np.meshgrid(x_freq_indices, y_freq_indices, copy=True)

    denom = 4 * ( (np.sin(0.5*math.pi*X_freq_grid/(f.shape[1])))**2 + (np.sin(0.5*math.pi*Y_freq_grid/(f.shape[0])))**2)

    denom[0,0] = 1e-9

    f = -fcos / denom

    tt = fftpack.idct(f, norm='ortho')
    img_tt = fftpack.idct(tt.T, norm='ortho').T

    return img_tt

class RGB2NormNet(nn.Module):
    def __init__(self):
        super(RGB2NormNet, self).__init__()
        input_size = 5
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,64)
        self.fc4 = nn.Linear(64,2)
        self.drop_layer = nn.Dropout(p=0.05)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc2(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc3(x))
        x = self.drop_layer(x)
        x = self.fc4(x)
        return x

class Reconstruction3D:
    def __init__(self, dev):
        self.cpuorgpu = "cpu"
        self.dm_zero_counter = 0
        self.dm_zero = np.zeros((dev.imgh, dev.imgw))
        print(f"Reconstruction3D initialized with dm_zero shape: {self.dm_zero.shape}")
        pass

    def load_nn(self, net_path, cpuorgpu):

        self.cpuorgpu = cpuorgpu
        device = torch.device(cpuorgpu)

        if not os.path.isfile(net_path):
            print('Error opening ', net_path, ' does not exist')
            return


        net = RGB2NormNet().float().to(device)

        if cpuorgpu=="cuda":
            checkpoint = torch.load(net_path, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
            net.load_state_dict(checkpoint['state_dict'])
        else:
            checkpoint = torch.load(net_path, map_location=torch.device('cpu'))
            net.load_state_dict(checkpoint['state_dict'])

        self.net = net

        return self.net

    def get_depthmap(self, frame, mask_markers, cm=None):
        MARKER_INTERPOLATE_FLAG = mask_markers

        if (cm is None):
            cm = np.ones(frame.shape[:2], dtype=bool)

        imgh = frame.shape[:2][0]
        imgw = frame.shape[:2][1]


        if MARKER_INTERPOLATE_FLAG:
            if frame.ndim == 3 and frame.shape[2] == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif frame.ndim == 2:
                gray_frame = frame
            else:
                print(f"경고: find_marker에 예상치 못한 프레임 차원: {frame.ndim}")
                gray_frame = frame[:,:,0] if frame.ndim == 3 else frame

            markermask = find_marker(gray_frame)
            cm = ~markermask.astype(bool)

        nx = np.zeros((imgh, imgw))
        ny = np.zeros((imgh, imgw))
        dm = np.zeros((imgh, imgw))

        cm_indices = np.where(cm)

        if cm_indices[0].shape[0] != 0:
            rgb = frame[cm_indices] / 255.

            pxpos = np.vstack(cm_indices).T

            pxpos_normalized = pxpos.copy().astype(np.float32)
            pxpos_normalized[:, 0] = pxpos[:, 0] / imgh
            pxpos_normalized[:, 1] = pxpos[:, 1] / imgw


            features = np.column_stack((rgb, pxpos_normalized))

            features = torch.from_numpy(features).float().to(self.cpuorgpu)

            with torch.no_grad():
                self.net.eval()
                out = self.net(features)

            nx[cm_indices] = out[:, 0].cpu().detach().numpy()
            ny[cm_indices] = out[:, 1].cpu().detach().numpy()

            nz = np.sqrt(1 - nx ** 2 - ny ** 2)
            if np.isnan(nz).any():
                print ('경고: nz 계산 중 NaN 발생. 평균값으로 대체합니다.')
                nz[np.isnan(nz)] = np.nanmean(nz)

            small_val = np.finfo(float).eps
            gx = -nx / (nz + small_val * (nz == 0))
            gy = -ny / (nz + small_val * (nz == 0))

        else:
            gx = np.zeros((imgh, imgw))
            gy = np.zeros((imgh, imgw))
            print("경고: 접촉 영역(cm)이 비어 있습니다. 기울기를 0으로 설정합니다.")


        if MARKER_INTERPOLATE_FLAG:
            dilated_mm = dilate(markermask, ksize=3, iter=2) > 0
            interp_mask = dilated_mm & ~markermask.astype(bool)

            gx_interp, gy_interp = demark(gx, gy, dilated_mm.astype(np.uint8) * 255)

        else:
            gx_interp, gy_interp = gx, gy



        dm = poisson_dct_neumaan(gx_interp, gy_interp)
        dm = np.reshape(dm, (imgh, imgw))

        if self.dm_zero_counter < 15:
            self.dm_zero += dm 
            print ('zeroing depth. do not touch the gel!')
            if self.dm_zero_counter == 14:
                self.dm_zero /= 15
                print ('Depth zeroing complete. Ok to touch me now!')
        self.dm_zero_counter += 1

        if self.dm_zero_counter > 15:
            dm = dm - self.dm_zero

        gx_min, gx_max = gx.min(), gx.max()
        gy_min, gy_max = gy.min(), gy.max()
        gx_interp_min, gx_interp_max = gx_interp.min(), gx_interp.max()
        gy_interp_min, gy_interp_max = gy_interp.min(), gy_interp.max()

        gx = (gx - gx_min) / ((gx_max - gx_min) + np.finfo(float).eps)
        gy = (gy - gy_min) / ((gy_max - gy_min) + np.finfo(float).eps)
        gx_interp = (gx_interp - gx_interp_min) / ((gx_interp_max - gx_interp_min) + np.finfo(float).eps)
        gy_interp = (gy_interp - gy_interp_min) / ((gy_interp_max - gy_interp_min) + np.finfo(float).eps)


        return dm


class Visualize3D:
    def __init__(self, n, m, save_path, mmpp):
        self.n, self.m = n, m
        self.init_open3D()
        self.cnt = 0
        self.save_path = save_path
        print(f"Visualize3D initialized with dimensions n={self.n} (Height), m={self.m} (Width)")
        pass

    def init_open3D(self):
        x_coords = np.arange(self.m)
        y_coords = np.arange(self.n)
        self.X, self.Y = np.meshgrid(x_coords, y_coords, indexing='xy')

        Z = np.zeros((self.n, self.m))

        self.points = np.zeros([self.n * self.m, 3])
        self.points[:, 0] = np.ndarray.flatten(self.X)
        self.points[:, 1] = np.ndarray.flatten(self.Y)

        self.depth2points(Z)

        self.pcd = open3d.geometry.PointCloud()
        self.pcd.points = open3d.utility.Vector3dVector(self.points)
        self.pcd.colors = open3d.utility.Vector3dVector(np.zeros_like(self.points))


        self.vis = open3d.visualization.Visualizer()
        self.vis.create_window(width=640, height=480, window_name="3D Visualization")
        self.vis.add_geometry(self.pcd)

    def depth2points(self, Z):
        if Z.shape[:2] != (self.n, self.m):
            print(f"경고: depth2points에 전달된 Z 형상 {Z.shape[:2]}이 Visualize3D 초기화 크기 ({self.n}, {self.m})와 다릅니다.")

        self.points[:, 2] = np.ndarray.flatten(Z)


    def update(self, Z):
        self.depth2points(Z) 
        dy_dz, dx_dz = np.gradient(Z)
        dx, dy = dx_dz * 0.5, dy_dz * 0.5
        np_colors_2d = dx + 0.5

        np_colors_2d[np_colors_2d < 0] = 0
        np_colors_2d[np_colors_2d > 1] = 1

        np_colors_flattened = np.ndarray.flatten(np_colors_2d)
        colors = np.zeros([self.points.shape[0], 3])
        for _ in range(3): colors[:,_] = np_colors_flattened

        self.pcd.points = open3d.utility.Vector3dVector(self.points)
        self.pcd.colors = open3d.utility.Vector3dVector(colors)

        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

        if self.save_path != '':
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            file_name = os.path.join(self.save_path, "pc_{:05d}.pcd".format(self.cnt))
            try:
                open3d.io.write_point_cloud(file_name, self.pcd)
            except Exception as e:
                print(f"경고: 포인트 클라우드 저장 중 오류 발생 - {file_name}: {e}")


        self.cnt += 1

    def save_pointcloud(self):
        if self.save_path != '':
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            file_name = os.path.join(self.save_path, "pc_{:05d}.pcd".format(self.cnt))
            try:
                open3d.io.write_point_cloud(file_name, self.pcd)
            except Exception as e:
                print(f"경고: 포인트 클라우드 저장 중 오류 발생 - {file_name}: {e}")
        print("포인트 클라우드 수동 저장 기능 완료.")