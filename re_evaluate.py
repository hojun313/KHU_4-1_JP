import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import f1_score, precision_score, recall_score
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
from PIL import Image
import lpips
import torch
import torchvision.transforms as T
import argparse
import os

def load_image_grayscale_numpy(image_path, target_size=(256, 256)):
    """이미지를 로드하고 흑백 NumPy 배열로 변환 후 리사이즈 (주로 GLCM, Canny용)."""
    if not os.path.exists(image_path):
        print(f"오류: 이미지 파일을 찾을 수 없습니다 - {image_path}")
        return None
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"오류: 이미지를 로드할 수 없습니다 - {image_path}")
        return None
    
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img_as_ubyte(img_resized) # GLCM은 0-255 uint8 타입 필요

def load_image_tensor_for_lpips(image_path, target_size=(256, 256), device='cpu'):
    """LPIPS 평가를 위해 이미지를 로드하고 [-1, 1] 범위의 텐서로 변환합니다."""
    if not os.path.exists(image_path):
        print(f"오류: 이미지 파일을 찾을 수 없습니다 - {image_path}")
        return None
    try:
        img_pil = Image.open(image_path).convert("RGB") # LPIPS는 RGB 이미지를 기대
    except Exception as e:
        print(f"오류: PIL 이미지 로드/변환 중 오류 ({image_path}): {e}")
        return None

    # LPIPS는 보통 [-1, 1] 범위의 이미지를 사용
    transform = T.Compose([
        T.Resize(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])
    tensor = transform(img_pil).unsqueeze(0).to(device) # 배치 차원 추가 및 디바이스로 이동
    return tensor

def get_canny_edges(image_numpy, low_threshold=50, high_threshold=150):
    """Canny 엣지 검출을 수행합니다 (입력: NumPy 배열)."""
    if image_numpy is None: return None
    return cv2.Canny(image_numpy, low_threshold, high_threshold)

def calculate_edge_ssim(edge_map1, edge_map2):
    if edge_map1 is None or edge_map2 is None: return None
    if edge_map1.shape != edge_map2.shape: return None
    win_size = min(7, min(edge_map1.shape) - 1)
    if win_size < 3: return None
    return ssim(edge_map1.astype(float), edge_map2.astype(float), data_range=255, win_size=win_size)

def calculate_edge_f1_precision_recall(edge_map1_binary, edge_map2_binary):
    if edge_map1_binary is None or edge_map2_binary is None: return None, None, None
    if edge_map1_binary.shape != edge_map2_binary.shape: return None, None, None
    y_true = (edge_map1_binary.flatten() / 255).astype(int)
    y_pred = (edge_map2_binary.flatten() / 255).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return f1, precision, recall

def calculate_lpips_score(img_tensor1, img_tensor2, lpips_model):
    """두 텐서 이미지 간의 LPIPS 점수를 계산합니다."""
    if img_tensor1 is None or img_tensor2 is None: return None
    with torch.no_grad():
        score = lpips_model(img_tensor1, img_tensor2)
    return score.item()

def calculate_glcm_features(image_numpy):
    """흑백 NumPy 이미지에서 GLCM 특징을 추출합니다."""
    if image_numpy is None: return None
    distances = [1, 3, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(image_numpy, distances=distances, angles=angles, symmetric=True, normed=True)
    
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    features = np.array([graycoprops(glcm, prop).mean() for prop in props])
    return features

def main(args):
    print(f"--- 원본 하이트맵: {args.ground_truth_path} ---")
    print(f"--- 생성된 하이트맵: {args.predicted_path} ---")
    DEVICE = "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    print(f"사용 장치: {DEVICE}")

    # --- 이미지 로드 ---
    # Canny, GLCM, 원본SSIM용 (흑백 NumPy, 0-255)
    gt_img_numpy = load_image_grayscale_numpy(args.ground_truth_path, (args.image_size, args.image_size))
    pred_img_numpy = load_image_grayscale_numpy(args.predicted_path, (args.image_size, args.image_size))
    
    # LPIPS용 (RGB 텐서, -1~1, GPU/CPU)
    gt_tensor_lpips = load_image_tensor_for_lpips(args.ground_truth_path, (args.image_size, args.image_size), DEVICE)
    pred_tensor_lpips = load_image_tensor_for_lpips(args.predicted_path, (args.image_size, args.image_size), DEVICE)

    if gt_img_numpy is None or pred_img_numpy is None or gt_tensor_lpips is None or pred_tensor_lpips is None:
        print("하나 이상의 이미지 로드 실패로 평가를 중단합니다.")
        return

    # --- 평가 결과 저장용 딕셔너리 ---
    results = {}

    # --- 1. 엣지 기반 평가 ---
    print("\n--- 엣지 기반 평가 수행 중 ---")
    gt_edges = get_canny_edges(gt_img_numpy, args.canny_low, args.canny_high)
    pred_edges = get_canny_edges(pred_img_numpy, args.canny_low, args.canny_high)

    if gt_edges is not None and pred_edges is not None:
        if args.save_edge_maps:
            os.makedirs(args.output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(args.output_dir, "edge_ground_truth.png"), gt_edges)
            cv2.imwrite(os.path.join(args.output_dir, "edge_predicted.png"), pred_edges)
            print(f"엣지 맵 저장 완료: {args.output_dir}")

        results['Edge_SSIM'] = calculate_edge_ssim(gt_edges, pred_edges)
        f1, precision, recall = calculate_edge_f1_precision_recall(gt_edges, pred_edges)
        results['Edge_F1_Score'] = f1
        results['Edge_Precision'] = precision
        results['Edge_Recall'] = recall
    else:
        print("엣지 검출 실패로 엣지 기반 평가를 건너<0xEB><0><0x88>니다.")

    # --- 2. LPIPS 평가 ---
    print("\n--- LPIPS 평가 수행 중 ---")
    try:
        lpips_model = lpips.LPIPS(net=args.lpips_net, verbose=False).to(DEVICE)
        for param in lpips_model.parameters(): param.requires_grad = False
        results['LPIPS_Score'] = calculate_lpips_score(gt_tensor_lpips, pred_tensor_lpips, lpips_model)
    except Exception as e:
        print(f"LPIPS 평가 중 오류: {e}")
        results['LPIPS_Score'] = None
        
    # --- 3. GLCM 특징 비교 ---
    print("\n--- GLCM 특징 비교 수행 중 ---")
    try:
        glcm_features_gt = calculate_glcm_features(gt_img_numpy)
        glcm_features_pred = calculate_glcm_features(pred_img_numpy)
        
        if glcm_features_gt is not None and glcm_features_pred is not None:
            glcm_feature_names = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 'ASM']
            results['GLCM_Features_GT'] = dict(zip(glcm_feature_names, glcm_features_gt))
            results['GLCM_Features_Pred'] = dict(zip(glcm_feature_names, glcm_features_pred))
            results['GLCM_Cosine_Similarity'] = cosine_similarity(glcm_features_gt.reshape(1, -1), glcm_features_pred.reshape(1, -1))[0,0]
        else:
            print("GLCM 특징 계산 실패로 GLCM 비교를 건너<0xEB><0><0x88>니다.")
    except Exception as e:
        print(f"GLCM 특징 비교 중 오류: {e}")

    # --- 4. (추가) 원본 하이트맵 간 SSIM ---
    print("\n--- 원본 하이트맵 SSIM 평가 수행 중 ---")
    # skimage.metrics.ssim은 0-255 uint8 이미지도 잘 처리함
    # 단, data_range를 명시해주는 것이 좋음
    win_size_orig = min(7, min(gt_img_numpy.shape) - 1)
    if win_size_orig < 3:
        results['Original_SSIM'] = None
        print("이미지가 너무 작아 원본 SSIM을 계산할 수 없습니다.")
    else:
        results['Original_SSIM'] = ssim(gt_img_numpy, pred_img_numpy, data_range=255, win_size=win_size_orig)


    # --- 최종 결과 출력 ---
    print("\n\n========== 최종 평가 결과 ==========")
    if results.get('Edge_SSIM') is not None:
        print(f"엣지 맵 간 SSIM 점수:       {results['Edge_SSIM']:.4f}")
    if results.get('Edge_F1_Score') is not None:
        print(f"엣지 맵 간 F1 점수:        {results['Edge_F1_Score']:.4f}")
        print(f"엣지 맵 간 Precision:      {results['Edge_Precision']:.4f}")
        print(f"엣지 맵 간 Recall:         {results['Edge_Recall']:.4f}")
    if results.get('Original_SSIM') is not None:
        print(f"원본 하이트맵 간 SSIM 점수: {results['Original_SSIM']:.4f}")
    if results.get('LPIPS_Score') is not None:
        print(f"LPIPS 점수 ({args.lpips_net} net):    {results['LPIPS_Score']:.4f} (낮을수록 좋음)")
    
    if results.get('GLCM_Features_GT') and results.get('GLCM_Features_Pred'):
        print("\nGLCM 특징 (Ground Truth):")
        for name, val in results['GLCM_Features_GT'].items(): print(f"  - {name:<15}: {val:.4f}")
        print("GLCM 특징 (Predicted):")
        for name, val in results['GLCM_Features_Pred'].items(): print(f"  - {name:<15}: {val:.4f}")
    if results.get('GLCM_Cosine_Similarity') is not None:
        print(f"GLCM 특징 벡터 코사인 유사도: {results['GLCM_Cosine_Similarity']:.4f} (1에 가까울수록 좋음)")
    print("===================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="하이트맵 이미지 유사도 종합 평가 (엣지, LPIPS, GLCM)")
    parser.add_argument("--ground_truth_path", type=str, default="IOFiles/Output/carpet3_groundtruth.png",
                        help="원본(Ground Truth) 하이트맵 이미지 경로")
    parser.add_argument("--predicted_path", type=str, default="IOFiles/Output/output_heightmap_1000.png",
                        help="생성된(Predicted) 하이트맵 이미지 경로")
    parser.add_argument("--output_dir", type=str, default="IOFiles/Edge/edge_eval_outputs",
                        help="엣지 맵 이미지 저장 시 사용할 디렉토리")
    
    parser.add_argument("--image_size", type=int, default=256, 
                        help="평가 전 이미지 리사이즈 크기 (가로, 세로 동일)")
    parser.add_argument("--canny_low", type=int, default=15, 
                        help="Canny 엣지 검출기의 낮은 임계값")
    parser.add_argument("--canny_high", type=int, default=30, 
                        help="Canny 엣지 검출기의 높은 임계값")
    parser.add_argument("--save_edge_maps", action='store_true', default=True,
                        help="생성된 엣지 맵을 이미지 파일로 저장할지 여부")
    
    parser.add_argument("--lpips_net", type=str, default='alex', choices=['alex', 'vgg'], help="LPIPS 계산에 사용할 네트워크 (alex 또는 vgg)")
    parser.add_argument("--use_gpu", action='store_true', default=False, help="GPU 사용 여부 (LPIPS 계산 시)")
    
    args = parser.parse_args()
    main(args)