import cv2
import numpy as np
from scipy.stats import wasserstein_distance, skew, kurtosis
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.transform import resize as sk_resize
import os # os 모듈 임포트 추가

def load_and_preprocess_image(image_path, target_size=None, normalize_to_uint8=True):
    """
    이미지를 로드하고 전처리합니다.

    Args:
        image_path (str): 이미지 파일 경로.
        target_size (tuple, optional): (width, height) 형태의 목표 이미지 크기.
                                       None이면 크기 조정을 하지 않습니다.
        normalize_to_uint8 (bool): True이면 이미지를 0-255 범위의 uint8로 정규화합니다.
                                   GLCM, LBP 등에 필요합니다.

    Returns:
        numpy.ndarray: 전처리된 이미지.
    """
    # 이미지를 그레이스케일로 로드합니다. 높이맵은 채널이 하나여야 합니다.
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    if target_size is not None and img.shape[:2] != (target_size[1], target_size[0]):
        # skimage.transform.resize는 (height, width) 순서를 사용합니다.
        # 또한, 값 범위를 [0,1]로 변경하므로, 원래 범위로 복원하거나 uint8로 변환해야 합니다.
        img_float = sk_resize(img, (target_size[1], target_size[0]), anti_aliasing=True, preserve_range=False)
        if normalize_to_uint8:
            img = (img_float * 255).astype(np.uint8)
        else: # 원래 값 범위를 유지하려고 시도 (주의: resize 후 정확히 유지되지 않을 수 있음)
            img = img_float
            # 만약 원본 이미지의 min/max를 알고 있다면, 그 범위로 스케일링하는 것이 더 정확합니다.
            # 여기서는 간단히 0-255로 가정하고, 필요시 사용자가 조정해야 합니다.
            if not normalize_to_uint8 and (img.min() < 0 or img.max() > (255 if img.dtype == np.uint8 else 1.0001)): # float의 경우 1.0 초과 체크
                 print(f"경고: {image_path} 리사이즈 후 값 범위가 예상과 다를 수 있습니다. 현재 범위: [{img.min():.2f}, {img.max():.2f}]")


    elif normalize_to_uint8 and img.dtype != np.uint8:
        # 이미지가 uint8이 아니면 0-255 범위로 정규화 후 uint8로 변환
        if img.max() > 1.0: # 이미 0-255 범위라고 가정 (또는 그 이상)
            # 0-1 범위로 먼저 정규화 후 0-255로 스케일링 하는 것이 더 안전할 수 있습니다.
            # 여기서는 사용자가 입력 이미지의 범위를 알고 있다고 가정합니다.
            # 만약 다양한 범위의 float 이미지가 입력될 수 있다면, min-max scaling이 필요합니다.
            img = np.clip(img, 0, 255).astype(np.uint8)
        else: # 0-1 범위라고 가정
            img = (img * 255).astype(np.uint8)

    return img

def compare_histograms_emd(img1, img2, bins=256):
    """
    두 이미지의 높이 값 히스토그램을 EMD(Earth Mover's Distance)로 비교합니다.

    Args:
        img1 (numpy.ndarray): 첫 번째 이미지.
        img2 (numpy.ndarray): 두 번째 이미지.
        bins (int): 히스토그램 계산 시 사용할 빈(bin)의 수.

    Returns:
        float: 두 히스토그램 간의 EMD 값.
    """
    hist1 = cv2.calcHist([img1], [0], None, [bins], [0, bins]).flatten() # range 수정: [0, bins-1] -> [0, bins] for 0-255
    hist2 = cv2.calcHist([img2], [0], None, [bins], [0, bins]).flatten() # range 수정: [0, bins-1] -> [0, bins] for 0-255

    # 히스토그램 정규화 (확률 분포로 만들기)
    # 합이 0인 경우를 대비하여 작은 값(epsilon)을 더해줌
    epsilon = 1e-9
    hist1 = hist1 / (hist1.sum() + epsilon)
    hist2 = hist2 / (hist2.sum() + epsilon)

    # 값의 범위 (픽셀 값 자체)
    values = np.arange(bins)

    return wasserstein_distance(values, values, hist1, hist2)

def calculate_global_stats_diff(img1, img2):
    """
    두 이미지의 전역 통계적 모멘트(평균, 분산, 왜도, 첨도) 차이를 계산합니다.

    Args:
        img1 (numpy.ndarray): 첫 번째 이미지.
        img2 (numpy.ndarray): 두 번째 이미지.

    Returns:
        dict: 각 통계량의 절대 차이를 담은 딕셔너리.
    """
    stats1 = {
        'mean': np.mean(img1),
        'variance': np.var(img1),
        'skewness': skew(img1.flatten()),
        'kurtosis': kurtosis(img1.flatten()) # Fisher's definition (normal = 0)
    }
    stats2 = {
        'mean': np.mean(img2),
        'variance': np.var(img2),
        'skewness': skew(img2.flatten()),
        'kurtosis': kurtosis(img2.flatten())
    }

    diff = {key: abs(stats1[key] - stats2[key]) for key in stats1}
    return diff

def compare_psd_mse(img1, img2):
    """
    두 이미지의 2D 푸리에 변환 기반 전력 스펙트럼 밀도(PSD)를 MSE로 비교합니다.

    Args:
        img1 (numpy.ndarray): 첫 번째 이미지.
        img2 (numpy.ndarray): 두 번째 이미지.

    Returns:
        float: 두 PSD 간의 Mean Squared Error.
    """
    if img1.shape != img2.shape:
        raise ValueError("PSD 비교를 위해서는 이미지 크기가 동일해야 합니다.")

    f1 = np.fft.fft2(img1)
    fshift1 = np.fft.fftshift(f1)
    psd1 = np.abs(fshift1)**2

    f2 = np.fft.fft2(img2)
    fshift2 = np.fft.fftshift(f2)
    psd2 = np.abs(fshift2)**2

    mse = np.mean((psd1 - psd2)**2)
    return mse

def compare_haralick_features_euclidean(img1_uint8, img2_uint8, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    두 이미지(uint8)의 GLCM 기반 하라릭 특징 벡터 간 유클리드 거리를 계산합니다.

    Args:
        img1_uint8 (numpy.ndarray): 첫 번째 이미지 (uint8 타입).
        img2_uint8 (numpy.ndarray): 두 번째 이미지 (uint8 타입).
        distances (list): GLCM 계산 시 사용할 픽셀 간 거리.
        angles (list): GLCM 계산 시 사용할 각도 (라디안).

    Returns:
        float: 두 하라릭 특징 벡터 간의 평균 유클리드 거리.
    """
    assert img1_uint8.dtype == np.uint8, "img1은 uint8 타입이어야 합니다."
    assert img2_uint8.dtype == np.uint8, "img2은 uint8 타입이어야 합니다."

    glcm1 = graycomatrix(img1_uint8, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    glcm2 = graycomatrix(img2_uint8, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

    haralick_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

    features1 = []
    for prop in haralick_props:
        features1.append(np.mean(graycoprops(glcm1, prop)))

    features2 = []
    for prop in haralick_props:
        features2.append(np.mean(graycoprops(glcm2, prop)))

    features1 = np.array(features1)
    features2 = np.array(features2)

    return np.linalg.norm(features1 - features2)


def compare_lbp_histograms_emd(img1_uint8, img2_uint8, P=8, R=1, method='uniform', bins=None):
    """
    두 이미지(uint8)의 LBP 히스토그램을 EMD로 비교합니다.

    Args:
        img1_uint8 (numpy.ndarray): 첫 번째 이미지 (uint8 타입).
        img2_uint8 (numpy.ndarray): 두 번째 이미지 (uint8 타입).
        P (int): LBP 계산 시 사용할 이웃 픽셀 수.
        R (int): LBP 계산 시 사용할 원의 반지름.
        method (str): LBP 계산 방법 ('default', 'ror', 'uniform', 'nri_uniform').
        bins (int, optional): 히스토그램 빈 수. 'uniform' LBP의 경우 P+2, 그 외에는 2^P.
                              None이면 자동으로 설정됩니다.

    Returns:
        float: 두 LBP 히스토그램 간의 EMD 값.
    """
    assert img1_uint8.dtype == np.uint8, "img1은 uint8 타입이어야 합니다."
    assert img2_uint8.dtype == np.uint8, "img2은 uint8 타입이어야 합니다."

    lbp1 = local_binary_pattern(img1_uint8, P, R, method)
    lbp2 = local_binary_pattern(img2_uint8, P, R, method)

    if bins is None:
        if method == 'uniform':
            bins = P + 2
        else:
            bins = 2**P

    hist_range = (0, bins) # 이론적 최대값을 사용 (일반적)
    # LBP 패턴 값은 0부터 시작하므로 range는 (0, bins)가 맞습니다.
    # np.histogram의 bins는 [range[0], range[1]) 구간을 나누는 개수 또는 경계 배열입니다.
    # 정수형 패턴의 경우, 각 패턴 값이 하나의 빈이 되도록 하는 것이 일반적입니다.
    # 따라서 bins는 패턴의 종류 수, range는 (최소 패턴값, 최대 패턴값 + 1)이 됩니다.
    # 'uniform' LBP의 경우 패턴 값은 0부터 P+1까지 총 P+2개 입니다.
    # 'default' LBP의 경우 0부터 2^P-1까지 총 2^P개 입니다.
    # 현재 bins가 패턴의 종류 수로 설정되므로, range는 (0, bins)가 적절합니다.
    # (np.arange(bins)가 0부터 bins-1까지의 값을 생성하므로 일치)

    hist1, _ = np.histogram(lbp1.ravel(), bins=bins, range=hist_range, density=True)
    hist2, _ = np.histogram(lbp2.ravel(), bins=bins, range=hist_range, density=True)

    values = np.arange(hist_range[0], hist_range[1]) # LBP 패턴 값들

    return wasserstein_distance(values, values, hist1, hist2)


def calculate_all_haptic_losses(image_path1, image_path2, target_size=(256, 256)):
    """
    두 높이맵 이미지 간의 다양한 햅틱 관련 손실/차이 지표를 계산합니다.

    Args:
        image_path1 (str): 첫 번째 높이맵 이미지 파일 경로.
        image_path2 (str): 두 번째 높이맵 이미지 파일 경로.
        target_size (tuple): 이미지 비교를 위해 통일할 크기 (width, height).
                             None이면 원본 크기 사용 (PSD 등 일부 비교에 문제 발생 가능).

    Returns:
        dict: 각 비교 방법에 따른 손실/차이 값을 담은 딕셔너리.
              값이 작을수록 두 이미지가 해당 측면에서 유사함을 의미합니다.
    """
    img1_uint8 = load_and_preprocess_image(image_path1, target_size=target_size, normalize_to_uint8=True)
    img2_uint8 = load_and_preprocess_image(image_path2, target_size=target_size, normalize_to_uint8=True)

    img1_float = img1_uint8.astype(np.float32) / 255.0
    img2_float = img2_uint8.astype(np.float32) / 255.0

    # target_size가 None일 때 PSD 비교를 위한 이미지 크기 조정 부분은
    # load_and_preprocess_image에서 target_size를 강제하거나,
    # 여기서 명시적으로 처리하는 것이 좋습니다.
    # 현재는 load_and_preprocess_image에서 target_size가 None이면 원본 크기를 사용합니다.
    # compare_psd_mse 함수 내에서 크기 불일치 시 에러를 발생시킵니다.
    # 따라서, 이 부분의 별도 크기 조정 로직은 제거하거나,
    # compare_psd_mse 호출 전에 명시적으로 크기를 맞추는 로직으로 변경할 수 있습니다.
    # 여기서는 compare_psd_mse 내부의 에러 핸들링에 의존합니다.
    # img1_uint8_psd, img2_uint8_psd, img1_float_psd, img2_float_psd 변수들은
    # img1_uint8, img2_uint8, img1_float, img2_float와 동일하게 사용합니다.

    losses = {}

    losses['histogram_emd'] = compare_histograms_emd(img1_uint8, img2_uint8)
    losses['global_stats_diff'] = calculate_global_stats_diff(img1_float, img2_float)

    try:
        # PSD 비교는 크기가 동일해야 하므로, target_size가 설정되었거나 원본 크기가 같은 경우에만 유효합니다.
        # load_and_preprocess_image에서 target_size가 None이 아니고 크기가 다르면 resize를 수행합니다.
        # target_size가 None이고 크기가 다르면 compare_psd_mse에서 ValueError가 발생합니다.
        img1_psd_input = load_and_preprocess_image(image_path1, target_size=target_size, normalize_to_uint8=False)
        img2_psd_input = load_and_preprocess_image(image_path2, target_size=target_size, normalize_to_uint8=False)
        if img1_psd_input.shape != img2_psd_input.shape and target_size is None:
             print(f"경고: 원본 이미지 크기가 다르고 target_size가 지정되지 않아 PSD 비교를 건너<0xEB><0><0xA9>니다. ({img1_psd_input.shape} vs {img2_psd_input.shape})")
             losses['psd_mse'] = float('nan') # 비교 불가능함을 명시
        else:
             losses['psd_mse'] = compare_psd_mse(img1_psd_input, img2_psd_input)

    except ValueError as e:
        print(f"PSD 비교 오류: {e}")
        losses['psd_mse'] = float('inf')
    except FileNotFoundError as e: # load_and_preprocess_image에서 발생 가능
        print(f"PSD 비교 위한 이미지 로드 오류: {e}")
        losses['psd_mse'] = float('inf')


    losses['haralick_euclidean'] = compare_haralick_features_euclidean(img1_uint8, img2_uint8)
    losses['lbp_emd'] = compare_lbp_histograms_emd(img1_uint8, img2_uint8, P=8, R=1)

    return losses

if __name__ == '__main__':
    # --- 사용 예시 ---
    # 비교할 두 이미지 파일의 경로를 여기에 입력하세요.
    image_path_ground_truth = "Validate_H-Map/ground_truth.png"  # 예: "original_carpet.png"
    image_path_generated = "Validate_H-Map/1.output_heightmap.png"    # 예: "model_output_carpet.png"

    # (선택 사항) 이미지 처리 크기 설정 (None으로 하면 원본 크기 사용)
    # 크기를 지정하면 모든 이미지가 이 크기로 조절되어 비교됩니다. (일관성 및 속도에 영향)
    # PSD 비교 등을 위해서는 동일 크기로 만드는 것이 좋습니다.
    target_processing_size = (256, 256) # 예: (128, 128) 또는 None

    print(f"--- '{os.path.basename(image_path_ground_truth)}' vs '{os.path.basename(image_path_generated)}' ---")

    # 파일 존재 여부 확인
    if not os.path.exists(image_path_ground_truth):
        print(f"오류: 원본 이미지 파일을 찾을 수 없습니다 - {image_path_ground_truth}")
        print("image_path_ground_truth 변수에 올바른 파일 경로를 입력하세요.")
    elif not os.path.exists(image_path_generated):
        print(f"오류: 생성된 이미지 파일을 찾을 수 없습니다 - {image_path_generated}")
        print("image_path_generated 변수에 올바른 파일 경로를 입력하세요.")
    else:
        try:
            haptic_losses = calculate_all_haptic_losses(image_path_ground_truth, image_path_generated, target_size=target_processing_size)
            
            # 각 지표에 대한 설명 문자열 정의
            descriptions = {
                'histogram_emd': "전체 높이 값 분포 유사도 (EMD, 작을수록 유사)",
                'global_stats_diff': "전역 통계량 차이:",
                'mean': "  - 평균 높이 차이 (작을수록 유사)",
                'variance': "  - 높이 분산(전체적 거칠기/대비) 차이 (작을수록 유사)",
                'skewness': "  - 높이 분포 왜도(비대칭성) 차이 (작을수록 유사)",
                'kurtosis': "  - 높이 분포 첨도(뾰족함) 차이 (작을수록 유사)",
                'psd_mse': "공간 주파수 분포(패턴 스케일) 유사도 (MSE, 작을수록 유사)",
                'haralick_euclidean': "미세 텍스처 패턴(GLCM 기반) 유사도 (유클리드 거리, 작을수록 유사)",
                'lbp_emd': "국부 미세 패턴(LBP 기반) 분포 유사도 (EMD, 작을수록 유사)"
            }

            for metric, value in haptic_losses.items():
                if metric == 'global_stats_diff':
                    print(f"{descriptions[metric]}")
                    for k, v_item in value.items():
                        print(f"  {k:<10}: {v_item:>10.4f} ({descriptions.get(k, '')})")
                else:
                    print(f"{metric:<20}: {value:>10.4f} ({descriptions.get(metric, '')})")

        except FileNotFoundError as e:
            # calculate_all_haptic_losses 내부에서도 FileNotFoundError를 처리하지만,
            # 경로 설정 자체가 잘못되었을 경우를 위해 여기서도 잡을 수 있습니다.
            print(f"이미지 처리 중 오류 발생: {e}")
            print("이미지 경로가 올바른지 다시 확인해주세요.")
        except Exception as e:
            print(f"알 수 없는 오류 발생: {e}")
