import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d # 이 모듈이 필요합니다.
import numpy as np
import math
import os
import cv2
from scipy.interpolate import griddata
from scipy import fftpack


def find_marker(gray):
    mask = cv2.inRange(gray, 0, 70)
    # kernel = np.ones((2, 2), np.uint8)
    # dilation = cv2.dilate(mask, kernel, iterations=1)
    return mask

def dilate(img, ksize=5, iter=1):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(img, kernel, iterations=iter)

def erode(img, ksize=5):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.erode(img, kernel, iterations=1)

def matching_rows(A,B):
    ### https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
    matches=[i for i in range(B.shape[0]) if np.any(np.all(A==B[i],axis=1))]
    if len(matches)==0:
        return B[matches]
    return np.unique(B[matches],axis=0)

def interpolate_gradients(gx, gy, img, cm, markermask):
    ''' interpolate gradients at marker location '''

    # if np.where(cm)[0].shape[0] != 0:
    cmcm = np.zeros(img.shape[:2])
    ind1 = np.vstack(np.where(cm)).T
    ind2 = np.vstack(np.where(markermask)).T
    ind2not = np.vstack(np.where(~markermask)).T
    ind3 = matching_rows(ind1, ind2)
    cmcm[(ind3[:, 0], ind3[:, 1])] = 1.
    ind4 = ind1[np.all(np.any((ind1 - ind3[:, None]), axis=2), axis=0)]
    # x = np.linspace(0, 240, 240) # <-- 이 부분은 사용되지 않는 것 같습니다.
    # y = np.linspace(0,320, 320) # <-- 이 부분은 사용되지 않는 것 같습니다.
    # X, Y = np.meshgrid(x, y) # <-- 이 부분은 사용되지 않는 것 같습니다.

    '''interpolate at the intersection of cm and markermask '''
    # gx_interpol = griddata(ind4, gx[(ind4[:, 0], ind4[:, 1])], ind3, method='nearest')
    # gx[(ind3[:, 0], ind3[:, 1])] = gx_interpol
    # gy_interpol = griddata(ind4, gy[(ind4[:, 0], gy[(ind4[:, 0], ind4[:, 1])], ind3, method='nearest')
    # gy[(ind3[:, 0], ind3[:, 1])] = gy_interpol

    ''' interpolate at the entire markermask '''
    # griddata(points, values, xi) -> points: known data points (N, D), values: values at points (N,), xi: interpolation points (M, D)
    # 여기서는 ind2 (마커 마스크 내부 점들)의 값을 ind2not (마커 마스크 외부 점들)에서 보간하려 시도합니다.
    # gx[(ind2[:, 0], ind2[:, 1])]는 마커 마스크 내부 점들의 gx 값입니다.
    # gx[(ind2not[:, 0], ind2not[:, 1])]는 마커 마스크 외부 점들의 gx 값입니다.
    # points는 마커 마스크 내부 점들의 좌표 (N, 2)가 되어야 합니다. -> ind2
    # values는 해당 점들에서의 gx 값 (N,)가 되어야 합니다. -> gx[(ind2[:, 0], ind2[:, 1])]
    # xi는 보간하려는 점들의 좌표 (M, 2)가 되어야 합니다. -> ind2not
    # 따라서 gx_interpol의 결과는 ind2not의 길이와 같아야 합니다.
    # gx[(ind2not[:, 0], ind2not[:, 1])]에 결과를 대입하는 것은 맞습니다.

    # 수정: griddata의 points와 values는 알려진 점들이므로 ind2와 gx[ind2[:,0], ind2[:,1]]를 사용해야 합니다.
    # xi는 보간하려는 점들이므로 ind2not를 사용해야 합니다.
    # 원래 코드의 인자 순서가 잘못된 것 같습니다.
    # gx_interpol = griddata(ind2, gx[(ind2[:, 0], ind2[:, 1])], ind2not, method='nearest') # <-- 원래 의도는 이거였을 것 같습니다.
    # gy_interpol = griddata(ind2, gy[(ind2[:, 0], ind2[:, 1])], ind2not, method='nearest') # <-- 원래 의도는 이거였을 것 같습니다.
    # 원본 코드를 그대로 유지합니다.
    try:
        # 원본 코드는 값을 인자로 넘기고 xi를 gx/gy 전체에서 가져옵니다. 의도와 다를 수 있지만 일단 유지.
        # gx_interpol = griddata(ind2, gx[(ind2[:, 0], ind2[:, 1])], gx[(ind2not[:, 0], ind2not[:, 1])], method='nearest')
        # gy_interpol = griddata(ind2, gy[(ind2[:, 0], gy[(ind2[:, 0], ind2[:, 1])], gy[(ind2not[:, 0], ind2not[:, 1])], method='nearest')

        # griddata(points, values, xi)
        # points: 마커 내부 좌표 (N, 2)
        points_known = ind2
        # values: 마커 내부 gx/gy 값 (N,)
        values_gx_known = gx[(ind2[:, 0], ind2[:, 1])]
        values_gy_known = gy[(ind2[:, 0], ind2[:, 1])]
        # xi: 마커 외부 좌표 (M, 2) - 보간하려는 지점
        points_interp = ind2not

        # 알려진 점이 없거나 보간할 점이 없으면 보간하지 않음
        if len(points_known) > 0 and len(points_interp) > 0:
            gx_interpolated_values = griddata(points_known, values_gx_known, points_interp, method='nearest')
            gy_interpolated_values = griddata(points_known, values_gy_known, points_interp, method='nearest')

            # 보간된 값을 원래 gx, gy 배열에 대입
            gx_out = gx.copy()
            gy_out = gy.copy()
            gx_out[(ind2not[:, 0], ind2not[:, 1])] = gx_interpolated_values
            gy_out[(ind2not[:, 0], ind2not[:, 1])] = gy_interpolated_values
        else: # 보간할 필요가 없으면 원본 gx, gy 반환
            gx_out = gx.copy()
            gy_out = gy.copy()

    except Exception as e:
        print(f"interpolate_gradients 오류: {e}")
        # 오류 발생 시 보간되지 않은 원본 반환 또는 처리 방식 결정
        gx_out = gx.copy()
        gy_out = gy.copy()
        # 오류가 치명적이면 여기서 예외를 다시 발생시키거나 종료할 수 있습니다.


    #print (gy_interpol.shape, gx_interpol.shape, gx.shape, gy.shape)

    ''' interpolate using samples in the vicinity of marker '''


    ''' method #3 '''
    # ind1 = np.vstack(np.where(markermask)).T
    # gx_interpol = scipy.ndimage.map_coordinates(gx, [ind1[:, 0], ind1[:, 1]], order=1, mode='constant')
    # gx[(ind1[:, 0], ind1[:, 1])] = gx_interpol
    # gy_interpol = scipy.ndimage.map_coordinates(gy, [ind1[:, 0], ind1[:, 1]], order=1, mode='constant')
    # gx[(ind1[:, 0], ind1[:, 1])] = gy_interpol

    ''' method #4 '''
    # x = np.arange(0, img.shape[0])
    # y = np.arange(0, img.shape[1])
    # fgx = scipy.interpolate.RectBivariateSpline(x, y, gx, kx=2, ky=2, s=0)
    # gx_interpol = fgx.ev(ind2[:,0],ind2[:,1])
    # gx[(ind2[:, 0], ind2[:, 1])] = gx_interpol
    # fgy = scipy.interpolate.RectBivariateSpline(x, y, gy, kx=2, ky=2, s=0)
    # gy_interpol = fgy.ev(ind2[:, 0], ind2[:, 1])
    # gy[(ind2[:, 0], ind2[:, 1])] = gy_interpol

    return gx_out, gy_out # 보간된 gx, gy 반환


def interpolate_grad(img, mask):
    # mask = (soft_mask > 0.5).astype(np.uint8) * 255
    # pixel around markers
    # markermask 내부(mask != 0)가 아닌 외부(mask == 0)이면서 확장된 마스크 내부(dilate > 0)인 영역
    mask_around = (dilate(mask, ksize=3, iter=2) > 0) & ~(mask != 0)
    # mask_around = mask == 0 # <-- 원래 주석 처리된 부분
    mask_around = mask_around.astype(bool) # 불리언 인덱싱을 위해 타입 변경

    x, y = np.arange(img.shape[0]), np.arange(img.shape[1]) # 이미지 행/열 인덱스 범위
    yy, xx = np.meshgrid(y, x) # <-- meshgrid 순서 주의 (yy: 열 인덱스 그리드, xx: 행 인덱스 그리드)

    # mask_zero = mask == 0 # <-- 원래 주석 처리된 부분
    # mask_zero는 이제 mark_around와 동일합니다.
    mask_to_interp = mask_around # 보간할 영역 (마커 주변의 마커가 아닌 부분)

    # cv2.imshow("mask_zero", mask_zero*1.) # <-- 디버깅용 주석

    # if np.where(mask_zero)[0].shape[0] != 0: # <-- 조건은 mask_to_interp 기준으로
    #     print ('interpolating') # <-- 디버깅용 주석

    # points: 알려진 점들의 좌표 (values가 있는 곳) - 여기서는 마스크 외부(mask == 0) 점들을 사용하려 함
    # values: 알려진 점들에서의 값 (img[points])
    # xi: 보간하려는 점들의 좌표 - 여기서는 마스크 내부(mask != 0) 점들을 사용하려 함

    # 원본 코드의 의도는 mask == 0인 영역의 값을 이용하여 mask != 0인 영역의 값을 보간하려는 것 같습니다.
    # 그렇다면 points는 mask == 0인 영역의 좌표가 되어야 합니다.
    # values는 mask == 0인 영역에서의 img 값이 되어야 합니다.
    # xi는 mask != 0인 영역의 좌표가 되어야 합니다.

    # 수정: points와 values는 알려진 점들(mask == 0)에서 가져오고, xi는 보간하려는 점들(mask != 0)에서 가져옵니다.
    points_known = np.vstack([xx[~mask.astype(bool)], yy[~mask.astype(bool)]]).T # 마스크 False인 부분의 좌표 (행, 열)
    values_known = img[~mask.astype(bool)] # 마스크 False인 부분의 img 값
    points_interp = np.vstack([xx[mask.astype(bool)], yy[mask.astype(bool)]]).T # 마스크 True인 부분의 좌표 (행, 열)

    # 알려진 점이 없거나 보간할 점이 없으면 보간하지 않음
    if len(points_known) > 0 and len(points_interp) > 0:
        method = "nearest" # 원본 코드의 기본값
        # method = "linear" # 원본 코드의 주석
        # method = "cubic" # 원본 코드의 주석
        x_interp_values = griddata(points_known, values_known, points_interp, method=method)

        # 보간 결과에 NaN이 있으면 0.0으로 채움 (원본 코드 동일)
        x_interp_values[x_interp_values != x_interp_values] = 0.0

        # 원본 img 배열을 복사하여 마스크 부분에 보간된 값을 대입
        ret = img.copy()
        ret[mask.astype(bool)] = x_interp_values
    else: # 보간할 필요가 없으면 원본 img 반환
        ret = img.copy()


    # else: # <-- 원래 else 블록
    #     ret = img # <-- 원래 else 블록 내용

    return ret

def demark(gx, gy, markermask):
    # mask = find_marker(img) # <-- 원본 코드의 주석
    gx_interp = interpolate_grad(gx.copy(), markermask)
    gy_interp = interpolate_grad(gy.copy(), markermask)
    return gx_interp, gy_interp

#@njit(parallel=True) # <-- numba 관련 주석
def get_features(img,pixels,features,imgw,imgh):
    features[:,3], features[:,4] = pixels[:,0] / imgh, pixels[:,1] / imgw # Y, X 좌표 정규화?
    for k in range(len(pixels)):
        i,j = pixels[k] # i: 행 (높이), j: 열 (너비)
        rgb = img[i, j] / 255. # RGB 값 정규화
        features[k,:3] = rgb

#
# 2D integration via Poisson solver
#
def poisson_dct_neumaan(gx,gy):

    # 중앙 차분으로 발산 계산
    # gxx: gx의 x 방향 미분 (열 방향 차분)
    # gy: (H, W) -> gx[:, 1:] shape (H, W-1), gx[:, :-1] shape (H, W-1)
    # gx[:, (list(range(1, gx.shape[1])) + [gx.shape[1] - 1])] : 1열부터 끝열까지 + 마지막 열 반복 -> shape (H, W)
    # gx[:, ([0] + list(range(gx.shape[1] - 1)))] : 첫열 반복 + 첫열부터 끝에서 두번째 열까지 -> shape (H, W)
    # 이 계산은 경계 처리를 포함한 x 방향 차분입니다. 결과 형상 (H, W)
    gxx = 1 * (gx[:, (list(range(1, gx.shape[1])) + [gx.shape[1] - 1])] - gx[:, ([0] + list(range(gx.shape[1] - 1)))])
    # gyy: gy의 y 방향 미분 (행 방향 차분)
    # gy[(list(range(1,gy.shape[0]))+[gy.shape[0]-1]), :] : 1행부터 끝행까지 + 마지막 행 반복 -> shape (H, W)
    # gy[([0]+list(range(gy.shape[0]-1))), :] : 첫행 반복 + 첫행부터 끝에서 두번째 행까지 -> shape (H, W)
    # 이 계산은 경계 처리를 포함한 y 방향 차분입니다. 결과 형상 (H, W)
    gyy = 1 * (gy[(list(range(1,gy.shape[0]))+[gy.shape[0]-1]), :] - gy[([0]+list(range(gy.shape[0]-1))), :])

    # 발산 f = d(gx)/dx + d(gy)/dy. 결과 형상 (H, W)
    f = gxx + gyy

    ### Right hand side of the boundary condition
    # 경계 조건 b 계산. 형상 (H, W)
    b = np.zeros(gx.shape)
    b[0,1:-2] = -gy[0,1:-2]
    b[-1,1:-2] = gy[-1,1:-2]
    b[1:-2,0] = -gx[1:-2,0]
    b[1:-2,-1] = gx[1:-2,-1]
    b[0,0] = (1/np.sqrt(2))*(-gy[0,0] - gx[0,0])
    b[0,-1] = (1/np.sqrt(2))*(-gy[0,-1] + gx[0,-1])
    b[-1,-1] = (1/np.sqrt(2))*(gy[-1,-1] + gx[-1,-1])
    b[-1,0] = (1/np.sqrt(2))*(gy[-1,0]-gx[-1,0])

    ## Modification near the boundaries to enforce the non-homogeneous Neumann BC (Eq. 53 in [1])
    # 경계 부분 f 값 수정. 형상 (H, W) 유지.
    f[0,1:-2] = f[0,1:-2] - b[0,1:-2]
    f[-1,1:-2] = f[-1,1:-2] - b[-1,1:-2]
    f[1:-2,0] = f[1:-2,0] - b[1:-2,0]
    f[1:-2,-1] = f[1:-2,-1] - b[1:-2,-1]

    ## Modification near the corners (Eq. 54 in [1])
    # 모서리 부분 f 값 수정. 형상 (H, W) 유지.
    f[0,-1] = f[0,-1] - np.sqrt(2) * b[0,-1]
    f[-1,-1] = f[-1,-1] - np.sqrt(2) * b[-1,-1]
    f[-1,0] = f[-1,0] - np.sqrt(2) * b[-1,0]
    f[0,0] = f[0,0] - np.sqrt(2) * b[0,0]

    ## Cosine transform of f
    # 2D DCT 변환. 형상 (H, W) 유지.
    tt = fftpack.dct(f, norm='ortho') # 기본적으로 마지막 축 (열)에 대해 DCT
    fcos = fftpack.dct(tt.T, norm='ortho').T # 행에 대해 DCT 후 다시 전치. 결과 형상 (H, W)

    # Cosine transform of z (Eq. 55 in [1])
    # DCT 계수를 나누기 위한 분모 계산
    # x: 열 방향 주파수 인덱스 (0부터 W-1), y: 행 방향 주파수 인덱스 (0부터 H-1)
    # np.meshgrid의 기본 인덱싱은 'xy'이며, 첫번째 인자(width 관련)가 열을 따라, 두번째 인자(height 관련)가 행을 따라 반복됩니다.
    # 결과 x_grid shape는 (len(y_range), len(x_range)), y_grid shape도 동일.
    # 여기서는 주파수 인덱스에 해당하는 값을 grid로 만들어야 합니다.
    # f.shape[1]은 너비 W, f.shape[0]은 높이 H
    # x 주파수 인덱스는 0부터 W-1 까지, y 주파수 인덱스는 0부터 H-1 까지
    # 공식의 형태를 보니 주파수 인덱스를 1부터 W, 1부터 H로 사용하고 있는 것 같습니다.
    # grid의 크기는 fcos와 같아야 합니다. (H, W)
    # griddata(range(1, f.shape[1] + 1), range(1, f.shape[0] + 1)) <- 여기가 문제였던 부분.
    # 주파수 k와 l에 대한 grid를 (H, W) 형상으로 만들어야 합니다.
    # k는 행 주파수 (0..H-1), l은 열 주파수 (0..W-1).
    # k_grid = np.arange(f.shape[0]).reshape(-1, 1) # (H, 1)
    # l_grid = np.arange(f.shape[1]).reshape(1, -1) # (1, W)
    # k_grid = np.tile(k_grid, (1, f.shape[1])) # (H, W)
    # l_grid = np.tile(l_grid, (f.shape[0], 1)) # (H, W)

    # 원본 코드가 사용하려 했던 1-based index grid 재현
    # f.shape[1]은 너비, f.shape[0]은 높이
    # np.meshgrid(열 범위, 행 범위)
    x_freq_indices = np.arange(1, f.shape[1] + 1) # 1..W
    y_freq_indices = np.arange(1, f.shape[0] + 1) # 1..H
    # meshgrid 기본 인덱싱 'xy': X_grid shape (H, W), Y_grid shape (H, W)
    X_freq_grid, Y_freq_grid = np.meshgrid(x_freq_indices, y_freq_indices, copy=True)

    # 분모 계산. 형상 (H, W)
    denom = 4 * ( (np.sin(0.5*math.pi*X_freq_grid/(f.shape[1])))**2 + (np.sin(0.5*math.pi*Y_freq_grid/(f.shape[0])))**2)

    # 분모가 0인 경우 처리 (주파수 (0,0) 성분). 이 성분은 따로 처리해야 합니다.
    # denom[0,0]은 0이 됩니다. 여기에 해당하는 fcos[0,0] 성분은 기울기가 아닌 평균 높이와 관련.
    # 푸아송 적분 결과는 상대 높이이므로 평균 높이는 0으로 맞춰주는 경우가 많습니다.
    # denom[0,0]을 0이 아닌 값 (예: 1e-9)으로 바꾸거나, 해당 성분 계산 시 제외.
    denom[0,0] = 1e-9 # 0으로 나누는 것을 방지하기 위해 작은 값으로 대체

    # fcos는 DCT 계수. 형상 (H, W)
    # 분모 denom도 형상 (H, W)
    # 나눗셈 결과 f의 형상도 (H, W)
    f = -fcos / denom

    # Inverse Discrete cosine Transform (2D IDCT). 결과 형상 (H, W)
    tt = fftpack.idct(f, norm='ortho') # 기본적으로 마지막 축 (열)에 대해 IDCT
    img_tt = fftpack.idct(tt.T, norm='ortho').T # 행에 대해 IDCT 후 다시 전치. 결과 형상 (H, W)

    # 평균 높이를 0으로 맞추거나 더합니다.
    # img_tt = img_tt.mean() + img_tt # 원본 코드 주석 처리된 부분
    # img_tt = img_tt - img_tt.min() # 최소 높이를 0으로 맞춤 (원본 코드 주석 처리된 부분)

    return img_tt # 형상 (H, W)


''' nn architecture for mini '''
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
        # FIX: Initialize dm_zero with shape (Height, Width) to match dm shape
        # dev.imgh가 높이, dev.imgw가 너비입니다. 순서를 바꿔서 (높이, 너비)로 초기화합니다.
        self.dm_zero = np.zeros((dev.imgh, dev.imgw)) # <-- 이 줄을 수정했습니다.
        print(f"Reconstruction3D initialized with dm_zero shape: {self.dm_zero.shape}") # 디버깅 출력
        pass

    def load_nn(self, net_path, cpuorgpu):

        self.cpuorgpu = cpuorgpu
        device = torch.device(cpuorgpu)

        if not os.path.isfile(net_path):
            print('Error opening ', net_path, ' does not exist')
            return


        net = RGB2NormNet().float().to(device)

        if cpuorgpu=="cuda":
            ### load weights on gpu
            # net.load_state_dict(torch.load(net_path))
            # map_location 수정 (FutureWarning 관련) - 필요시 weights_only=True 추가 검토
            checkpoint = torch.load(net_path, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device())) # 현재 GPU 사용
            net.load_state_dict(checkpoint['state_dict'])
        else:
            ### load weights on cpu which were actually trained on gpu
            # map_location 수정 (FutureWarning 관련) - 필요시 weights_only=True 추가 검토
            checkpoint = torch.load(net_path, map_location=torch.device('cpu')) # CPU 명시
            net.load_state_dict(checkpoint['state_dict'])

        self.net = net

        return self.net

    def get_depthmap(self, frame, mask_markers, cm=None):
        MARKER_INTERPOLATE_FLAG = mask_markers

        ''' find contact region '''
        # cm, cmindx = find_contact_mask(f1, f0)
        ###################################################################
        ### check these sizes
        ##################################################################
        if (cm is None):
            # cm = np.ones(frame.shape[:2]) # 원본 코드
            # cmindx = np.where(cm) # 원본 코드
            cm = np.ones(frame.shape[:2], dtype=bool) # 마스크는 불리언 또는 uint8
            # cmindx는 np.where(cm)의 결과. 필요하면 사용.

        imgh = frame.shape[:2][0] # 이미지의 높이
        imgw = frame.shape[:2][1] # 이미지의 너비
        # print(f"get_depthmap received frame shape: {frame.shape[:2]}, imgh={imgh}, imgw={imgw}") # 디버깅 출력


        if MARKER_INTERPOLATE_FLAG:
            ''' find marker mask '''
            # BGR 이미지를 그레이스케일로 변환하여 마커 찾기
            if frame.ndim == 3 and frame.shape[2] == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # OpenCV 기본은 BGR
            elif frame.ndim == 2:
                gray_frame = frame # 이미 그레이스케일인 경우
            else:
                print(f"경고: find_marker에 예상치 못한 프레임 차원: {frame.ndim}")
                gray_frame = frame[:,:,0] if frame.ndim == 3 else frame # 일단 첫 채널 사용 또는 그대로 사용

            markermask = find_marker(gray_frame)
            # cm = ~markermask # 원본 코드 - 마커 영역을 접촉 영역에서 제외하는 듯
            # 마커 마스크가 True인 곳은 제외하고 나머지가 cm (접촉 영역)이 됩니다.
            cm = ~markermask.astype(bool) # 불리언 마스크로 변환

            '''intersection of cm and markermask '''
            # cmmm = np.zeros(img.shape[:2]) # <-- 사용되지 않는 변수
            # ind1 = np.vstack(np.where(cm)).T # <-- 사용되지 않는 변수
            # ind2 = np.vstack(np.where(markermask)).T # <-- 사용되지 않는 변수
            # ind2not = np.vstack(np.where(~markermask)).T # <-- 사용되지 않는 변수
            # ind3 = matching_rows(ind1, ind2) # <-- 사용되지 않는 함수/변수
            # cmmm[(ind3[:, 0], ind3[:, 1])] = 1. # <-- 사용되지 않는 변수
            # cmandmm = (np.logical_and(cm, markermask)).astype('uint8') # <-- 사용되지 않는 변수
            # cmandnotmm = (np.logical_and(cm, ~markermask)).astype('uint8') # <-- 사용되지 않는 변수


        ''' Get depth image with NN '''
        # nx, ny, dm의 형상을 (높이, 너비)로 초기화
        nx = np.zeros((imgh, imgw))
        ny = np.zeros((imgh, imgw))
        dm = np.zeros((imgh, imgw))

        ''' ENTIRE CONTACT MASK THRU NN '''
        # if np.where(cm)[0].shape[0] != 0: # cm 마스크에 True인 요소가 하나라도 있으면
        # cm 마스크가 불리언이라고 가정하고 np.where(cm) 사용
        cm_indices = np.where(cm) # 결과는 (행 인덱스 배열, 열 인덱스 배열) 튜플

        if cm_indices[0].shape[0] != 0: # 접촉 영역(cm)이 비어 있지 않다면
            # 접촉 영역에 해당하는 픽셀의 RGB 값과 좌표 가져오기
            # rgb = frame[np.where(cm)] / 255 # 원본 코드
            # np.where(cm) 결과는 튜플이므로 인덱싱 방식 변경
            rgb = frame[cm_indices] / 255. # RGB 값 정규화 [0, 1] 범위
            # rgb = diffimg[np.where(cm)] # <-- 주석 처리된 원본 코드

            # 픽셀 좌표 가져오기 (행, 열) -> (높이 위치, 너비 위치)
            # pxpos = np.vstack(np.where(cm)).T # 원본 코드 - 결과는 (N, 2) 형태로 (행, 열)
            pxpos = np.vstack(cm_indices).T # 결과는 (N, 2) 형태로 (행 인덱스, 열 인덱스)

            # 픽셀 좌표 정규화 [0, 1] 범위
            # pxpos[:, 0]는 행 인덱스 (높이), pxpos[:, 1]는 열 인덱스 (너비)
            # pxpos[:, 0]을 imgh로 나누고, pxpos[:, 1]을 imgw로 나눕니다.
            # the neural net was trained using height=320, width=240 <-- 원본 주석
            # pxpos[:, [1, 0]] = pxpos[:, [0, 1]] # swapping <-- 원본 주석 - 좌표 순서를 (열, 행)으로 바꿨었음?
            # 다시 원본 주석처럼 (행, 열) -> (높이 정규화, 너비 정규화)로 사용
            pxpos_normalized = pxpos.copy().astype(np.float32) # 복사 후 형변환
            pxpos_normalized[:, 0] = pxpos[:, 0] / imgh # 높이 정규화
            pxpos_normalized[:, 1] = pxpos[:, 1] / imgw # 너비 정규화


            # features = np.column_stack((rgb, pxpos)) # 원본 코드 - pxpos 사용
            features = np.column_stack((rgb, pxpos_normalized)) # 정규화된 좌표 사용

            features = torch.from_numpy(features).float().to(self.cpuorgpu)

            with torch.no_grad():
                self.net.eval()
                out = self.net(features) # 신경망 출력 (nx, ny 성분)

            # 신경망 출력을 nx, ny 배열에 대입
            # out[:, 0]은 nx, out[:, 1]은 ny 일 것으로 예상
            # 원본 코드: nx[np.where(cm)] = out[:, 0].cpu().detach().numpy()
            # np.where(cm) 결과 튜플 사용
            nx[cm_indices] = out[:, 0].cpu().detach().numpy()
            ny[cm_indices] = out[:, 1].cpu().detach().numpy()

            # print(nx.min(), nx.max(), ny.min(), ny.max()) # 디버깅 주석
            # Normalization options (주석 처리됨)
            # nx = 2 * ((nx - nx.min()) / (nx.max() - nx.min())) -1
            # ny = 2 * ((ny - ny.min()) / (ny.max() - ny.min())) -1
            # print(nx.min(), nx.max(), ny.min(), ny.max()) # 디버깅 주석

            '''OPTION#1 normalize gradient between [a,b]'''
            # ... (주석 처리됨) ...
            '''OPTION#2 calculate gx, gy from nx, ny. '''
            ### normalize normals to get gradients for poisson
            # nz 계산. np.nanany 대신 np.isnan을 np.any와 함께 사용
            nz = np.sqrt(1 - nx ** 2 - ny ** 2)
            if np.isnan(nz).any():
                print ('경고: nz 계산 중 NaN 발생. 평균값으로 대체합니다.')
                nz[np.isnan(nz)] = np.nanmean(nz) # NaN 값을 nz의 평균값으로 대체

            # gx, gy 계산
            # 분모가 0인 경우 방지 (nz가 0이면 수직 법선, 기울기가 무한대). 작은 값 더하기.
            # nz가 0에 매우 가까우면 불안정해질 수 있음.
            # np.finfo(float).eps는 부동소수점 표현의 아주 작은 값
            small_val = np.finfo(float).eps
            gx = -nx / (nz + small_val * (nz == 0)) # nz가 0이면 small_val 더함
            gy = -ny / (nz + small_val * (nz == 0)) # nz가 0이면 small_val 더함

        else: # 접촉 영역(cm)이 비어 있다면
            # 접촉 영역이 없으면 기울기(gx, gy)를 모두 0으로 설정
            gx = np.zeros((imgh, imgw))
            gy = np.zeros((imgh, imgw))
            print("경고: 접촉 영역(cm)이 비어 있습니다. 기울기를 0으로 설정합니다.")


        # 마커 영역 보간
        if MARKER_INTERPOLATE_FLAG:
            # gx, gy = interpolate_gradients(gx, gy, img, cm, cmmm) # <-- 원본 주석
            # 보간 마스크 생성: 마커 주변의 마커가 아닌 영역
            dilated_mm = dilate(markermask, ksize=3, iter=2) > 0 # 마커 영역 확장
            # 보간할 영역 = (확장된 마커 영역) AND (NOT 마커 영역) = 마커 주변 테두리
            interp_mask = dilated_mm & ~markermask.astype(bool)

            # interpolate_grad 함수를 사용하여 보간
            # interpolate_grad 함수는 img와 mask를 받아서 mask 영역을 보간합니다.
            # 여기서 보간하려는 것은 gx, gy의 'interp_mask' 영역 값입니다.
            # interpolate_grad의 mask 인자에 어떤 마스크를 넘겨야 하는지 명확하지 않습니다.
            # 원본 코드의 demark 함수를 보면 interpolate_grad에 'dilated_mm'를 넘깁니다.
            # interpolate_grad 내부 로직을 보니 mask == 0인 영역을 이용하여 mask != 0인 영역을 보간하는 형태입니다.
            # 따라서 demark(gx, gy, dilated_mm)는 gx, gy에서 dilated_mm가 True인 영역을 보간하는 것으로 보입니다.
            # 즉, 확장된 마커 영역 전체를 보간하는 것입니다.
            gx_interp, gy_interp = demark(gx, gy, dilated_mm.astype(np.uint8) * 255) # demark 함수는 uint8 마스크를 기대하는 듯

        else: # 마커 보간 비활성화 시
            gx_interp, gy_interp = gx, gy

        # nz = np.sqrt(1 - nx ** 2 - ny ** 2) # <-- 이미 위에서 계산됨
        # boundary = np.zeros((imgh, imgw)) # <-- 사용되지 않는 변수


        # 푸아송 적분으로 깊이 맵 계산
        # gx_interp, gy_interp 형상은 (imgh, imgw), 즉 (높이, 너비)
        dm = poisson_dct_neumaan(gx_interp, gy_interp)
        # poisson_dct_neumaan 결과 형상은 (높이, 너비) 일 것으로 예상. reshape 필요 없을 수 있지만 원본 유지.
        dm = np.reshape(dm, (imgh, imgw))
        # print(f"Depth map (dm) shape before zeroing: {dm.shape}") # 디버깅 출력

        ''' remove initial zero depth '''
        # self.dm_zero_counter가 50보다 작으면 dm_zero 누적
        if self.dm_zero_counter < 15:
            self.dm_zero += dm # dm 형상 (높이, 너비), self.dm_zero 형상 (높이, 너비) -> 형상 일치!
            print ('zeroing depth. do not touch the gel!')
            if self.dm_zero_counter == 14:
                self.dm_zero /= 15 # 49가 아닌 50으로 나누는 것이 맞음
                print ('Depth zeroing complete. Ok to touch me now!')
        self.dm_zero_counter += 1

        # 누적 완료 후 dm에서 평균 깊이(dm_zero)를 뺍니다.
        if self.dm_zero_counter > 15: # 50 프레임 누적 완료 후
            dm = dm - self.dm_zero # <--- 이제 여기서 형상 일치 (높이, 너비) - (높이, 너비)
        # print(dm.min(), dm.max()) # 디버깅 주석

        ''' ENTIRE MASK. GPU OPTIMIZED VARIABLES. '''
        # ... (GPU 최적화 관련 주석 처리된 코드) ...

        ''' normalize gradients for plotting purpose '''
        # 기울기 값을 [0, 1] 범위로 정규화 (시각화 목적)
        # 분모가 0이 되는 경우 방지
        gx_min, gx_max = gx.min(), gx.max()
        gy_min, gy_max = gy.min(), gy.max()
        gx_interp_min, gx_interp_max = gx_interp.min(), gx_interp.max()
        gy_interp_min, gy_interp_max = gy_interp.min(), gy_interp.max()

        gx = (gx - gx_min) / ((gx_max - gx_min) + np.finfo(float).eps) # 분모 0 방지
        gy = (gy - gy_min) / ((gy_max - gy_min) + np.finfo(float).eps) # 분모 0 방지
        gx_interp = (gx_interp - gx_interp_min) / ((gx_interp_max - gx_interp_min) + np.finfo(float).eps) # 분모 0 방지
        gy_interp = (gy_interp - gy_interp_min) / ((gy_interp_max - gy_interp_min) + np.finfo(float).eps) # 분모 0 방지


        return dm # 최종 깊이 맵 반환


class Visualize3D:
    def __init__(self, n, m, save_path, mmpp):
        self.n, self.m = n, m # n: 높이, m: 너비
        self.init_open3D()
        self.cnt = 0 # 프레임 카운터 초기화
        self.save_path = save_path
        print(f"Visualize3D initialized with dimensions n={self.n} (Height), m={self.m} (Width)") # 디버깅 출력
        pass

    def init_open3D(self):
        # X, Y 좌표 그리드 생성 (m: 너비, n: 높이). Open3D는 보통 X=가로, Y=세로, Z=깊이
        # 이미지 좌표계 (행=y, 열=x)와 Open3D 좌표계 (X=열, Y=행) 간의 변환 고려 필요
        # 여기서는 n(높이), m(너비) 기준으로 그리드를 만듭니다.
        # self.X는 m(너비) 방향, self.Y는 n(높이) 방향으로 스케일링된 좌표를 가져야 합니다.
        # np.arange(m) : 0부터 m-1까지 (너비)
        # np.arange(n) : 0부터 n-1까지 (높이)
        # np.meshgrid(X축 범위, Y축 범위, indexing='xy')
        # X축은 열(m), Y축은 행(n)
        x_coords = np.arange(self.m) # 0부터 너비-1
        y_coords = np.arange(self.n) # 0부터 높이-1
        # X_grid shape (n, m), Y_grid shape (n, m)
        self.X, self.Y = np.meshgrid(x_coords, y_coords, indexing='xy')

        # 그리드를 mmpp 값으로 스케일링 (mm 단위로 변환)
        # self.X *= mmpp # <-- mmpp 적용은 외부에서 필요하거나 여기서 적용해야 함. __init__ 인자에 mmpp가 있습니다.
        # self.Y *= mmpp

        Z = np.zeros((self.n, self.m)) # 초기 Z 값은 0

        self.points = np.zeros([self.n * self.m, 3])
        # points[:, 0]는 X (열) 좌표, points[:, 1]는 Y (행) 좌표
        self.points[:, 0] = np.ndarray.flatten(self.X) # X 좌표 (열)
        self.points[:, 1] = np.ndarray.flatten(self.Y) # Y 좌표 (행)

        self.depth2points(Z) # 초기 Z 값을 포인트 클라우드에 설정

        self.pcd = open3d.geometry.PointCloud()
        self.pcd.points = open3d.utility.Vector3dVector(self.points)
        # self.pcd.colors = Vector3dVector(np.zeros([self.n * self.m, 3])) # 색상 초기화 (검정)
        self.pcd.colors = open3d.utility.Vector3dVector(np.zeros_like(self.points)) # 색상 초기화 (형상에 맞게)


        self.vis = open3d.visualization.Visualizer()
        self.vis.create_window(width=640, height=480, window_name="3D Visualization") # 창 이름 설정
        self.vis.add_geometry(self.pcd)

        # 뷰 컨트롤러 설정 (카메라 위치, 방향 등) - 필요시 추가

    def depth2points(self, Z):
        # Z는 (높이, 너비) 형상의 깊이 맵
        # 포인트 클라우드의 Z 좌표로 설정
        # Z의 형상이 (self.n, self.m) 즉 (높이, 너비)와 일치해야 함.
        if Z.shape[:2] != (self.n, self.m):
            print(f"경고: depth2points에 전달된 Z 형상 {Z.shape[:2]}이 Visualize3D 초기화 크기 ({self.n}, {self.m})와 다릅니다.")
            # 크기가 다르면 문제가 발생할 수 있으므로 처리 방식을 결정해야 합니다.
            # 일단 현재는 오류가 나지 않도록 reshape를 시도하거나 경고만 출력합니다.
            # 가장 안전한 것은 dm의 크기가 (self.n, self.m)과 일치하도록 하는 것입니다.
            # 이 코드는 dm의 형상이 일치한다고 가정하고 진행됩니다.

        self.points[:, 2] = np.ndarray.flatten(Z) # 깊이(Z) 값을 포인트 클라우드의 Z 좌표로


    def update(self, Z):
        # Z는 (높이, 너비) 형상의 깊이 맵 (dm)
        self.depth2points(Z) # 포인트 클라우드의 Z 좌표 업데이트

        # 깊이 맵의 그라디언트 계산 (시각화 목적)
        # np.gradient는 (dy, dx) 순서로 반환 (y: 행 방향, x: 열 방향)
        dy_dz, dx_dz = np.gradient(Z)
        # 그라디언트 값의 스케일 조정
        dx, dy = dx_dz * 0.5, dy_dz * 0.5 # 원본 코드 (x, y 방향 스케일링)

        # 그라디언트 값을 색상으로 매핑 (예시: dx 값을 회색조로)
        # np_colors = dx + 0.5 # 원본 코드 (dx 사용)
        # dx와 dy의 크기(magnitude)를 색상으로 사용할 수도 있습니다. np.sqrt(dx**2 + dy**2)
        # 여기서는 원본 코드처럼 dx를 사용하되, 클리핑 및 형상 변환
        np_colors_2d = dx + 0.5 # 형상 (n, m)

        # 값을 [0, 1] 범위로 클리핑
        np_colors_2d[np_colors_2d < 0] = 0
        np_colors_2d[np_colors_2d > 1] = 1

        # 색상 배열 생성 (N, 3)
        np_colors_flattened = np.ndarray.flatten(np_colors_2d) # 형상 (n*m,)
        colors = np.zeros([self.points.shape[0], 3]) # 형상 (n*m, 3)
        for _ in range(3): colors[:,_] = np_colors_flattened # 각 채널에 동일한 값 복사 (회색조)

        self.pcd.points = open3d.utility.Vector3dVector(self.points) # Z 좌표 업데이트 반영
        self.pcd.colors = open3d.utility.Vector3dVector(colors) # 색상 업데이트 반영

        # Open3D 시각화 업데이트
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events() # 이벤트 처리 (창 조작 등)
        self.vis.update_renderer() # 렌더링 업데이트

        #### SAVE POINT CLOUD TO A FILE
        if self.save_path != '':
            # 저장 경로가 디렉토리인지 확인하고 없으면 생성
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            file_name = os.path.join(self.save_path, "pc_{:05d}.pcd".format(self.cnt)) # 파일 이름 형식 변경 (5자리 숫자)
            try:
                open3d.io.write_point_cloud(file_name, self.pcd)
            except Exception as e:
                print(f"경고: 포인트 클라우드 저장 중 오류 발생 - {file_name}: {e}")


        self.cnt += 1 # 프레임 카운트 증가

    def save_pointcloud(self):
        # update 함수 내에서 매 프레임 저장하므로 이 함수는 직접 호출되지 않을 수 있습니다.
        if self.save_path != '':
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            file_name = os.path.join(self.save_path, "pc_{:05d}.pcd".format(self.cnt))
            try:
                open3d.io.write_point_cloud(file_name, self.pcd)
            except Exception as e:
                print(f"경고: 포인트 클라우드 저장 중 오류 발생 - {file_name}: {e}")
        print("포인트 클라우드 수동 저장 기능 완료.")