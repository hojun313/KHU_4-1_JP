import cv2
import numpy as np
from scipy.stats import wasserstein_distance, skew, kurtosis
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.transform import resize as sk_resize
import os

def load_and_preprocess_image(image_path, target_size=None, normalize_to_uint8=True):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    if target_size is not None and img.shape[:2] != (target_size[1], target_size[0]):
        img_float = sk_resize(img, (target_size[1], target_size[0]), anti_aliasing=True, preserve_range=False)
        if normalize_to_uint8:
            img = (img_float * 255).astype(np.uint8)
        else:
            img = img_float
            if not normalize_to_uint8 and (img.min() < 0 or img.max() > (255 if img.dtype == np.uint8 else 1.0001)):
                 print(f"경고: {image_path} 리사이즈 후 값 범위가 예상과 다를 수 있습니다. 현재 범위: [{img.min():.2f}, {img.max():.2f}]")


    elif normalize_to_uint8 and img.dtype != np.uint8:
        if img.max() > 1.0:
            img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = (img * 255).astype(np.uint8)

    return img

def compare_histograms_emd(img1, img2, bins=256):
    hist1 = cv2.calcHist([img1], [0], None, [bins], [0, bins]).flatten()
    hist2 = cv2.calcHist([img2], [0], None, [bins], [0, bins]).flatten()

    epsilon = 1e-9
    hist1 = hist1 / (hist1.sum() + epsilon)
    hist2 = hist2 / (hist2.sum() + epsilon)

    values = np.arange(bins)

    return wasserstein_distance(values, values, hist1, hist2)

def calculate_global_stats_diff(img1, img2):
    stats1 = {
        'mean': np.mean(img1),
        'variance': np.var(img1),
        'skewness': skew(img1.flatten()),
        'kurtosis': kurtosis(img1.flatten())
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
    assert img1_uint8.dtype == np.uint8, "img1은 uint8 타입이어야 합니다."
    assert img2_uint8.dtype == np.uint8, "img2은 uint8 타입이어야 합니다."

    lbp1 = local_binary_pattern(img1_uint8, P, R, method)
    lbp2 = local_binary_pattern(img2_uint8, P, R, method)

    if bins is None:
        if method == 'uniform':
            bins = P + 2
        else:
            bins = 2**P

    hist_range = (0, bins)

    hist1, _ = np.histogram(lbp1.ravel(), bins=bins, range=hist_range, density=True)
    hist2, _ = np.histogram(lbp2.ravel(), bins=bins, range=hist_range, density=True)

    values = np.arange(hist_range[0], hist_range[1])

    return wasserstein_distance(values, values, hist1, hist2)


def calculate_all_haptic_losses(image_path1, image_path2, target_size=(256, 256)):
    img1_uint8 = load_and_preprocess_image(image_path1, target_size=target_size, normalize_to_uint8=True)
    img2_uint8 = load_and_preprocess_image(image_path2, target_size=target_size, normalize_to_uint8=True)

    img1_float = img1_uint8.astype(np.float32) / 255.0
    img2_float = img2_uint8.astype(np.float32) / 255.0

    losses = {}

    losses['histogram_emd'] = compare_histograms_emd(img1_uint8, img2_uint8)
    losses['global_stats_diff'] = calculate_global_stats_diff(img1_float, img2_float)

    try:
        img1_psd_input = load_and_preprocess_image(image_path1, target_size=target_size, normalize_to_uint8=False)
        img2_psd_input = load_and_preprocess_image(image_path2, target_size=target_size, normalize_to_uint8=False)
        if img1_psd_input.shape != img2_psd_input.shape and target_size is None:
             print(f"경고: 원본 이미지 크기가 다르고 target_size가 지정되지 않아 PSD 비교를 건너뜁니다. ({img1_psd_input.shape} vs {img2_psd_input.shape})")
             losses['psd_mse'] = float('nan')
        else:
             losses['psd_mse'] = compare_psd_mse(img1_psd_input, img2_psd_input)

    except ValueError as e:
        print(f"PSD 비교 오류: {e}")
        losses['psd_mse'] = float('inf')
    except FileNotFoundError as e:
        print(f"PSD 비교 위한 이미지 로드 오류: {e}")
        losses['psd_mse'] = float('inf')


    losses['haralick_euclidean'] = compare_haralick_features_euclidean(img1_uint8, img2_uint8)
    losses['lbp_emd'] = compare_lbp_histograms_emd(img1_uint8, img2_uint8, P=8, R=1)

    return losses

if __name__ == '__main__':
    image_path_ground_truth = "Validate_H-Map/ground_truth.png"
    image_path_generated = "Validate_H-Map/1.output_heightmap.png"

    target_processing_size = (256, 256)

    print(f"--- '{os.path.basename(image_path_ground_truth)}' vs '{os.path.basename(image_path_generated)}' ---")

    if not os.path.exists(image_path_ground_truth):
        print(f"오류: 원본 이미지 파일을 찾을 수 없습니다 - {image_path_ground_truth}")
        print("image_path_ground_truth 변수에 올바른 파일 경로를 입력하세요.")
    elif not os.path.exists(image_path_generated):
        print(f"오류: 생성된 이미지 파일을 찾을 수 없습니다 - {image_path_generated}")
        print("image_path_generated 변수에 올바른 파일 경로를 입력하세요.")
    else:
        try:
            haptic_losses = calculate_all_haptic_losses(image_path_ground_truth, image_path_generated, target_size=target_processing_size)
            
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
            print(f"이미지 처리 중 오류 발생: {e}")
            print("이미지 경로가 올바른지 다시 확인해주세요.")
        except Exception as e:
            print(f"알 수 없는 오류 발생: {e}")
