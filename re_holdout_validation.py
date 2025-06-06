import os
import argparse
import torch
import glob
from PIL import Image
import torchvision.transforms as T
import torchvision
import lpips #
import numpy as np
import pandas as pd # 결과를 표로 정리하고 CSV로 저장하기 위함

# 기존 스크립트에서 필요한 함수들을 가져옵니다.
# 경로 문제가 발생하지 않도록 이 스크립트와 re_model.py 등이 같은 위치에 있거나, PYTHONPATH에 설정되어 있어야 합니다.
from re_model import create_model #
# re_predict.py의 함수들
from re_predict import load_trained_model, preprocess_image, postprocess_and_save_image #
# re_evaluate.py의 함수들
from re_evaluate import (
    load_image_grayscale_numpy, load_image_tensor_for_lpips, #
    get_canny_edges, calculate_edge_ssim, calculate_lpips_score, #
    calculate_edge_f1_precision_recall, calculate_glcm_features #
)
# re_evaluate.py에는 ssim이 skimage.metrics에서 직접 임포트되어 사용됩니다.
from skimage.metrics import structural_similarity as skimage_ssim #
# re_evaluate.py 에는 cosine_similarity 임포트가 없었으므로, sklearn에서 가져오거나 직접 구현합니다.
# 여기서는 sklearn.metrics.pairwise.cosine_similarity를 사용한다고 가정합니다.
try:
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity # (스크립트에 직접 없었으나 GLCM 코사인 유사도 언급)
except ImportError:
    print("scikit-learn이 설치되어 있지 않아 GLCM 코사인 유사도 계산이 불가능할 수 있습니다. (pip install scikit-learn)")
    sklearn_cosine_similarity = None

def calculate_glcm_cosine_similarity_metric(vec1, vec2):
    """GLCM 특징 벡터 간의 코사인 유사도를 계산합니다."""
    if vec1 is None or vec2 is None or sklearn_cosine_similarity is None:
        return None
    # re_evaluate.py에서 [0,0]으로 접근한 것을 보아, (1, N) 형태로 입력되어 (1,1) 결과가 나오는 것을 가정
    return sklearn_cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0, 0]


def run_prediction_for_material(model, material_input_path, material_prediction_output_path, image_size, device):
    """단일 재질의 입력 이미지에 대해 예측을 수행하고 결과를 저장합니다."""
    print(f"    예측 수행: '{material_input_path}' -> '{material_prediction_output_path}'")
    input_tensor = preprocess_image(material_input_path, image_size) #
    if input_tensor is None:
        print(f"      오류: 이미지 전처리 실패 - {material_input_path}")
        return False
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        prediction_tensor = model(input_tensor) #
    
    output_dir_pred = os.path.dirname(material_prediction_output_path)
    if output_dir_pred and not os.path.exists(output_dir_pred):
        os.makedirs(output_dir_pred, exist_ok=True)
        
    postprocess_and_save_image(prediction_tensor, material_prediction_output_path) #
    print(f"    예측 이미지 저장 완료: '{material_prediction_output_path}'")
    return True

def run_evaluation_for_material_pair(gt_path, pred_path, image_size, device, lpips_model_instance, canny_low_thresh, canny_high_thresh):
    """단일 재질의 정답-예측 쌍에 대해 모든 평가 지표를 계산합니다."""
    print(f"    평가 수행: GT='{gt_path}', Pred='{pred_path}'")
    eval_metrics = {}

    # NumPy 이미지 로드 (Canny, GLCM, 원본SSIM용)
    gt_img_np = load_image_grayscale_numpy(gt_path, (image_size, image_size)) #
    pred_img_np = load_image_grayscale_numpy(pred_path, (image_size, image_size)) #
    
    # LPIPS용 텐서 로드
    gt_tensor_lp = load_image_tensor_for_lpips(gt_path, (image_size, image_size), device) #
    pred_tensor_lp = load_image_tensor_for_lpips(pred_path, (image_size, image_size), device) #

    if gt_img_np is None or pred_img_np is None or gt_tensor_lp is None or pred_tensor_lp is None:
        print(f"      경고: 평가용 이미지 중 하나 이상을 로드할 수 없습니다. (GT: {gt_path}, Pred: {pred_path})")
        return {"error": "Image loading failed for evaluation"}

    # 1. 엣지 기반 평가
    gt_edges_map = get_canny_edges(gt_img_np, canny_low_thresh, canny_high_thresh) #
    pred_edges_map = get_canny_edges(pred_img_np, canny_low_thresh, canny_high_thresh) #

    if gt_edges_map is not None and pred_edges_map is not None:
        eval_metrics['Edge_SSIM'] = calculate_edge_ssim(gt_edges_map, pred_edges_map) #
        f1, prec, rec = calculate_edge_f1_precision_recall(gt_edges_map, pred_edges_map) #
        eval_metrics['Edge_F1'] = f1
        eval_metrics['Edge_Precision'] = prec
        eval_metrics['Edge_Recall'] = rec
    
    # 2. LPIPS 평가
    if lpips_model_instance:
        eval_metrics['LPIPS'] = calculate_lpips_score(gt_tensor_lp, pred_tensor_lp, lpips_model_instance) #
        
    # 3. GLCM 특징 비교
    glcm_features_gt = calculate_glcm_features(gt_img_np) #
    glcm_features_pred = calculate_glcm_features(pred_img_np) #
    if glcm_features_gt is not None and glcm_features_pred is not None:
        glcm_prop_names = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 'ASM']
        for idx, name in enumerate(glcm_prop_names):
            eval_metrics[f'GLCM_GT_{name}'] = glcm_features_gt[idx]
            eval_metrics[f'GLCM_Pred_{name}'] = glcm_features_pred[idx]
        eval_metrics['GLCM_Cosine_Similarity'] = calculate_glcm_cosine_similarity_metric(glcm_features_gt, glcm_features_pred) # (함수 호출)

    # 4. 원본 하이트맵 간 SSIM
    # re_evaluate.py의 ssim은 skimage.metrics.structural_similarity를 사용
    win_size = min(7, min(gt_img_np.shape) - 1) # re_evaluate.py의 Original_SSIM win_size 로직
    if win_size >=3: # re_evaluate.py 조건
        eval_metrics['Original_SSIM'] = skimage_ssim(gt_img_np, pred_img_np, data_range=255, win_size=win_size) #
    else:
        eval_metrics['Original_SSIM'] = None

    return eval_metrics

def find_test_files(material_base_dir, input_pattern="input_*.png", gt_heightmap_glob_pattern="output/*/heightmaps/*.png"):
    """주어진 재질 폴더에서 입력 이미지와 정답 하이트맵 경로를 찾습니다."""
    input_img_paths = glob.glob(os.path.join(material_base_dir, input_pattern))
    if not input_img_paths:
        return None, None
    
    # 여러 입력 이미지가 있을 경우 첫 번째 것을 사용하거나, 특정 규칙 적용 가능
    input_img_path = input_img_paths[0]
    
    # 정답 하이트맵 탐색 (re_dataset.py의 exclude_heightmap_indices_up_to 로직은 여기서는 단순화)
    # 테스트용 정답 하이트맵은 보통 명확히 하나로 정해져 있을 것을 가정합니다.
    # 예를 들어, 파일명이 특정 패턴을 따르거나, 가장 마지막 프레임 등.
    # 여기서는 첫 번째로 찾아지는 것을 사용하며, 실제 데이터에 맞게 수정이 필요할 수 있습니다.
    gt_heightmap_full_pattern = os.path.join(material_base_dir, gt_heightmap_glob_pattern)
    all_possible_gt_paths = glob.glob(gt_heightmap_full_pattern)
    
    # re_dataset.py의 exclude 로직을 간단히 적용해볼 수 있습니다.
    # exclude_indices_up_to = 15 # 예시 (argparse로 받을 수도 있음)
    # excluded_suffixes = [f"_{i:05d}" for i in range(exclude_indices_up_to + 1)]
    
    valid_gt_paths = []
    for p in all_possible_gt_paths:
        # filename_no_ext = os.path.splitext(os.path.basename(p))[0]
        # if not any(filename_no_ext.endswith(suffix) for suffix in excluded_suffixes):
        #     valid_gt_paths.append(p)
        # 단순화를 위해, 여기서는 모든 찾은 것을 유효하다고 가정 (테스트셋은 특정 GT가 있을 것이므로)
        valid_gt_paths.append(p)

    if not valid_gt_paths:
        return input_img_path, None
        
    # 어떤 GT를 사용할지 결정 (예: 가장 이름이 짧거나, 특정 이름 포함 등)
    # 여기서는 첫 번째 유효한 GT 사용
    gt_heightmap_path = valid_gt_paths[0] 
    
    return input_img_path, gt_heightmap_path

def holdout_validation_main(args):
    DEVICE = "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    print(f"사용 장치: {DEVICE}")

    # 1. 학습된 모델 로드
    print(f"학습된 모델 로드 중: '{args.checkpoint_path}'")
    try:
        trained_model = load_trained_model(args.checkpoint_path, DEVICE, args.encoder) #
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return
    trained_model.eval() # 추론 모드

    # 2. LPIPS 모델 로드 (평가에 필요시)
    lpips_instance = None
    if args.eval_lpips:
        try:
            print(f"LPIPS 모델({args.lpips_net}) 로드 중...")
            lpips_instance = lpips.LPIPS(net=args.lpips_net, verbose=False).to(DEVICE) #
            for param in lpips_instance.parameters(): # re_evaluate.py 및 re_train.py 참고
                param.requires_grad = False
        except Exception as e:
            print(f"LPIPS 모델 로드 실패: {e}. LPIPS 평가는 건넙니다.")

    # 3. 테스트 재질 폴더 목록 가져오기
    test_material_folders = [d for d in glob.glob(os.path.join(args.test_data_root, "*")) if os.path.isdir(d)]
    if not test_material_folders:
        print(f"오류: 테스트 데이터 루트 '{args.test_data_root}'에서 재질 폴더를 찾을 수 없습니다.")
        return
    print(f"총 {len(test_material_folders)}개의 테스트 재질 폴더를 찾았습니다.")

    # 예측 결과를 저장할 디렉토리 생성
    os.makedirs(args.predictions_dir, exist_ok=True)

    all_materials_eval_results = []

    for material_folder in test_material_folders:
        material_name = os.path.basename(material_folder)
        print(f"\n--- '{material_name}' 재질 처리 시작 ---")

        # 현재 재질의 입력 이미지와 정답 하이트맵 경로 찾기
        # 이 부분은 사용자의 데이터 구조에 맞게 `find_test_files` 함수 내부를 잘 조정해야 합니다.
        input_path, gt_path = find_test_files(material_folder, 
                                              gt_heightmap_glob_pattern=args.gt_glob_pattern)

        if not input_path:
            print(f"  경고: '{material_name}'에서 입력 이미지를 찾지 못했습니다. 건넙니다.")
            all_materials_eval_results.append({"material_id": material_name, "error": "Input image not found"})
            continue
        if not gt_path:
            print(f"  경고: '{material_name}'에서 정답 하이트맵을 찾지 못했습니다. 건넙니다.")
            all_materials_eval_results.append({"material_id": material_name, "error": "Ground truth heightmap not found"})
            continue
            
        # 예측 하이트맵 저장 경로 설정
        predicted_output_path = os.path.join(args.predictions_dir, f"{material_name}_predicted.png")

        # 4. 예측 수행
        prediction_successful = run_prediction_for_material(trained_model, input_path, predicted_output_path, 
                                                            args.image_size, DEVICE)
        
        if not prediction_successful or not os.path.exists(predicted_output_path):
            print(f"  오류: '{material_name}'에 대한 예측 실패 또는 예측 파일이 생성되지 않았습니다. 평가를 건넙니다.")
            all_materials_eval_results.append({"material_id": material_name, "error": "Prediction failed or output not found"})
            continue

        # 5. 평가 수행
        current_material_metrics = run_evaluation_for_material_pair(
            gt_path, predicted_output_path, args.image_size, DEVICE, 
            lpips_instance, args.canny_low, args.canny_high
        )
        current_material_metrics["material_id"] = material_name # 재질 ID 추가
        all_materials_eval_results.append(current_material_metrics)
        print(f"  '{material_name}' 재질 평가 완료.")

    # 6. 전체 결과 취합 및 요약
    if not all_materials_eval_results:
        print("수행된 평가가 없습니다.")
        return

    results_dataframe = pd.DataFrame(all_materials_eval_results)
    
    print("\n\n========== 홀드아웃 검증 종합 결과 ==========")
    print("--- 개별 재질 평가 결과 ---")
    # 모든 열이 보이도록 Pandas 출력 옵션 설정 (선택 사항)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(results_dataframe.to_string(index=False))

    # 오류가 없는 숫자형 결과에 대해서만 평균 및 표준편차 계산
    numeric_cols_results_df = results_dataframe.select_dtypes(include=np.number)
    if not numeric_cols_results_df.empty:
        mean_perf = numeric_cols_results_df.mean()
        std_perf = numeric_cols_results_df.std()
        
        summary_df = pd.DataFrame({'Mean': mean_perf, 'Std': std_perf})
        print("\n\n--- 전체 평균 성능 (오류 발생 재질 제외) ---")
        print(summary_df)
    else:
        print("\n평균 성능을 계산할 수 있는 숫자형 결과가 없습니다.")

    # 결과를 CSV 파일로 저장
    if args.output_csv:
        results_dataframe.to_csv(args.output_csv, index=False)
        print(f"\n전체 평가 결과가 '{args.output_csv}' 파일에 저장되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="학습된 모델을 사용하여 홀드아웃 테스트 세트에 대한 자동 예측 및 평가를 수행합니다.")
    
    # 모델 및 예측 관련 인자
    parser.add_argument("--checkpoint_path", type=str, default="training_outputs/efb7_lr1e-4_bs40_lpips_0/checkpoints/checkpoint_epoch_700.pth.tar", 
                        help="학습된 모델의 체크포인트 파일 경로 (.pth.tar)")
    parser.add_argument("--encoder", type=str, default="efficientnet-b7",
                        help="모델 학습 시 사용한 U-Net 인코더 이름 (re_model.py 참고)")
    parser.add_argument("--image_size", type=int, default=256, 
                        help="모델 입력 이미지 크기 (학습 시와 동일하게)")
    parser.add_argument("--use_gpu", action='store_true', default=False, 
                        help="GPU를 사용하여 예측 및 평가 수행")

    # 데이터 경로 관련 인자
    parser.add_argument("--test_data_root", type=str, default="./test_dataset",
                        help="홀드아웃 테스트 재질들의 상위 폴더 경로 (예: ./test_materials_holdout)")
    parser.add_argument("--predictions_dir", type=str, default="./test_dataset/0.holdout_predictions", 
                        help="생성된 예측 하이트맵 이미지들을 저장할 디렉토리")
    parser.add_argument("--gt_glob_pattern", type=str, default="output/*/heightmaps/*.png",
                        help="각 재질 폴더 내에서 정답 하이트맵을 찾기 위한 glob 패턴. (예: 'output/condition_A/heightmaps/gt.png')")


    # 평가 관련 인자 (re_evaluate.py 참고)
    parser.add_argument("--eval_lpips", action='store_true', default=True, 
                        help="LPIPS 지각 손실 평가 수행 여부 (플래그 지정 시 True)")
    parser.add_argument("--lpips_net", type=str, default='alex', choices=['alex', 'vgg'], 
                        help="LPIPS 계산에 사용할 네트워크 (alex 또는 vgg)")
    parser.add_argument("--canny_low", type=int, default=15, 
                        help="Canny 엣지 검출기의 낮은 임계값")
    parser.add_argument("--canny_high", type=int, default=30, 
                        help="Canny 엣지 검출기의 높은 임계값")
    
    # 결과 저장 관련 인자
    parser.add_argument("--output_csv", type=str, default="holdout_validation_results.csv", 
                        help="평가 결과를 저장할 CSV 파일 이름")
    
    args = parser.parse_args()
    holdout_validation_main(args)