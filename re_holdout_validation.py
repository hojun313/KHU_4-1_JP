import os
import argparse
import torch
import glob
from PIL import Image
import torchvision.transforms as T
import torchvision
import lpips
import numpy as np
import pandas as pd

from re_model import create_model
from re_predict import load_trained_model, preprocess_image, postprocess_and_save_image
from re_evaluate import (
    load_image_grayscale_numpy, load_image_tensor_for_lpips,
    get_canny_edges, calculate_edge_ssim, calculate_lpips_score,
    calculate_edge_f1_precision_recall, calculate_glcm_features
)
from skimage.metrics import structural_similarity as skimage_ssim
try:
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
except ImportError:
    print("scikit-learn이 설치되어 있지 않아 GLCM 코사인 유사도 계산이 불가능할 수 있습니다. (pip install scikit-learn)")
    sklearn_cosine_similarity = None

def calculate_glcm_cosine_similarity_metric(vec1, vec2):
    if vec1 is None or vec2 is None or sklearn_cosine_similarity is None:
        return None
    return sklearn_cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0, 0]


def run_prediction_for_material(model, material_input_path, material_prediction_output_path, image_size, device):
    print(f"    예측 수행: '{material_input_path}' -> '{material_prediction_output_path}'")
    input_tensor = preprocess_image(material_input_path, image_size)
    if input_tensor is None:
        print(f"      오류: 이미지 전처리 실패 - {material_input_path}")
        return False
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        prediction_tensor = model(input_tensor)
    
    output_dir_pred = os.path.dirname(material_prediction_output_path)
    if output_dir_pred and not os.path.exists(output_dir_pred):
        os.makedirs(output_dir_pred, exist_ok=True)
        
    postprocess_and_save_image(prediction_tensor, material_prediction_output_path)
    print(f"    예측 이미지 저장 완료: '{material_prediction_output_path}'")
    return True

def run_evaluation_for_material_pair(gt_path, pred_path, image_size, device, lpips_model_instance, canny_low_thresh, canny_high_thresh):
    print(f"    평가 수행: GT='{gt_path}', Pred='{pred_path}'")
    eval_metrics = {}

    gt_img_np = load_image_grayscale_numpy(gt_path, (image_size, image_size))
    pred_img_np = load_image_grayscale_numpy(pred_path, (image_size, image_size))
    
    gt_tensor_lp = load_image_tensor_for_lpips(gt_path, (image_size, image_size), device)
    pred_tensor_lp = load_image_tensor_for_lpips(pred_path, (image_size, image_size), device)

    if gt_img_np is None or pred_img_np is None or gt_tensor_lp is None or pred_tensor_lp is None:
        print(f"      경고: 평가용 이미지 중 하나 이상을 로드할 수 없습니다. (GT: {gt_path}, Pred: {pred_path})")
        return {"error": "Image loading failed for evaluation"}

    gt_edges_map = get_canny_edges(gt_img_np, canny_low_thresh, canny_high_thresh)
    pred_edges_map = get_canny_edges(pred_img_np, canny_low_thresh, canny_high_thresh)

    if gt_edges_map is not None and pred_edges_map is not None:
        eval_metrics['Edge_SSIM'] = calculate_edge_ssim(gt_edges_map, pred_edges_map)
        f1, prec, rec = calculate_edge_f1_precision_recall(gt_edges_map, pred_edges_map)
        eval_metrics['Edge_F1'] = f1
        eval_metrics['Edge_Precision'] = prec
        eval_metrics['Edge_Recall'] = rec
    
    if lpips_model_instance:
        eval_metrics['LPIPS'] = calculate_lpips_score(gt_tensor_lp, pred_tensor_lp, lpips_model_instance)
        
    glcm_features_gt = calculate_glcm_features(gt_img_np)
    glcm_features_pred = calculate_glcm_features(pred_img_np)
    if glcm_features_gt is not None and glcm_features_pred is not None:
        glcm_prop_names = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 'ASM']
        for idx, name in enumerate(glcm_prop_names):
            eval_metrics[f'GLCM_GT_{name}'] = glcm_features_gt[idx]
            eval_metrics[f'GLCM_Pred_{name}'] = glcm_features_pred[idx]
        eval_metrics['GLCM_Cosine_Similarity'] = calculate_glcm_cosine_similarity_metric(glcm_features_gt, glcm_features_pred)

    win_size = min(7, min(gt_img_np.shape) - 1)
    if win_size >=3:
        eval_metrics['Original_SSIM'] = skimage_ssim(gt_img_np, pred_img_np, data_range=255, win_size=win_size)
    else:
        eval_metrics['Original_SSIM'] = None

    return eval_metrics

def find_test_files(material_base_dir, input_pattern="input_*.png", gt_heightmap_glob_pattern="output/*/heightmaps/*.png"):
    input_img_paths = glob.glob(os.path.join(material_base_dir, input_pattern))
    if not input_img_paths:
        return None, None
    
    input_img_path = input_img_paths[0]
    
    gt_heightmap_full_pattern = os.path.join(material_base_dir, gt_heightmap_glob_pattern)
    all_possible_gt_paths = glob.glob(gt_heightmap_full_pattern)
    
    
    valid_gt_paths = []
    for p in all_possible_gt_paths:
        valid_gt_paths.append(p)

    if not valid_gt_paths:
        return input_img_path, None
        
    gt_heightmap_path = valid_gt_paths[0] 
    
    return input_img_path, gt_heightmap_path

def holdout_validation_main(args):
    DEVICE = "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    print(f"사용 장치: {DEVICE}")

    print(f"학습된 모델 로드 중: '{args.checkpoint_path}'")
    try:
        trained_model = load_trained_model(args.checkpoint_path, DEVICE, args.encoder)
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return
    trained_model.eval()

    lpips_instance = None
    if args.eval_lpips:
        try:
            print(f"LPIPS 모델({args.lpips_net}) 로드 중...")
            lpips_instance = lpips.LPIPS(net=args.lpips_net, verbose=False).to(DEVICE)
            for param in lpips_instance.parameters():
                param.requires_grad = False
        except Exception as e:
            print(f"LPIPS 모델 로드 실패: {e}. LPIPS 평가는 건넙니다.")

    test_material_folders = [d for d in glob.glob(os.path.join(args.test_data_root, "*")) if os.path.isdir(d)]
    if not test_material_folders:
        print(f"오류: 테스트 데이터 루트 '{args.test_data_root}'에서 재질 폴더를 찾을 수 없습니다.")
        return
    print(f"총 {len(test_material_folders)}개의 테스트 재질 폴더를 찾았습니다.")

    os.makedirs(args.predictions_dir, exist_ok=True)

    all_materials_eval_results = []

    for material_folder in test_material_folders:
        material_name = os.path.basename(material_folder)
        print(f"\n--- '{material_name}' 재질 처리 시작 ---")

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
            
        predicted_output_path = os.path.join(args.predictions_dir, f"{material_name}_predicted.png")

        prediction_successful = run_prediction_for_material(trained_model, input_path, predicted_output_path, 
                                                            args.image_size, DEVICE)
        
        if not prediction_successful or not os.path.exists(predicted_output_path):
            print(f"  오류: '{material_name}'에 대한 예측 실패 또는 예측 파일이 생성되지 않았습니다. 평가를 건넙니다.")
            all_materials_eval_results.append({"material_id": material_name, "error": "Prediction failed or output not found"})
            continue

        current_material_metrics = run_evaluation_for_material_pair(
            gt_path, predicted_output_path, args.image_size, DEVICE, 
            lpips_instance, args.canny_low, args.canny_high
        )
        current_material_metrics["material_id"] = material_name
        all_materials_eval_results.append(current_material_metrics)
        print(f"  '{material_name}' 재질 평가 완료.")

    if not all_materials_eval_results:
        print("수행된 평가가 없습니다.")
        return

    results_dataframe = pd.DataFrame(all_materials_eval_results)
    
    print("\n\n========== 홀드아웃 검증 종합 결과 ==========")
    print("--- 개별 재질 평가 결과 ---")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(results_dataframe.to_string(index=False))

    numeric_cols_results_df = results_dataframe.select_dtypes(include=np.number)
    if not numeric_cols_results_df.empty:
        mean_perf = numeric_cols_results_df.mean()
        std_perf = numeric_cols_results_df.std()
        
        summary_df = pd.DataFrame({'Mean': mean_perf, 'Std': std_perf})
        print("\n\n--- 전체 평균 성능 (오류 발생 재질 제외) ---")
        print(summary_df)
    else:
        print("\n평균 성능을 계산할 수 있는 숫자형 결과가 없습니다.")

    if args.output_csv:
        results_dataframe.to_csv(args.output_csv, index=False)
        print(f"\n전체 평가 결과가 '{args.output_csv}' 파일에 저장되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="학습된 모델을 사용하여 홀드아웃 테스트 세트에 대한 자동 예측 및 평가를 수행합니다.")
    
    parser.add_argument("--checkpoint_path", type=str, default="training_outputs/efb7_lr1e-4_bs40_lpips_0/checkpoints/checkpoint_epoch_700.pth.tar", 
                        help="학습된 모델의 체크포인트 파일 경로 (.pth.tar)")
    parser.add_argument("--encoder", type=str, default="efficientnet-b7",
                        help="모델 학습 시 사용한 U-Net 인코더 이름 (re_model.py 참고)")
    parser.add_argument("--image_size", type=int, default=256, 
                        help="모델 입력 이미지 크기 (학습 시와 동일하게)")
    parser.add_argument("--use_gpu", action='store_true', default=False, 
                        help="GPU를 사용하여 예측 및 평가 수행")

    parser.add_argument("--test_data_root", type=str, default="./test_dataset",
                        help="홀드아웃 테스트 재질들의 상위 폴더 경로 (예: ./test_materials_holdout)")
    parser.add_argument("--predictions_dir", type=str, default="./test_dataset/0.holdout_predictions", 
                        help="생성된 예측 하이트맵 이미지들을 저장할 디렉토리")
    parser.add_argument("--gt_glob_pattern", type=str, default="output/*/heightmaps/*.png",
                        help="각 재질 폴더 내에서 정답 하이트맵을 찾기 위한 glob 패턴. (예: 'output/condition_A/heightmaps/gt.png')")


    parser.add_argument("--eval_lpips", action='store_true', default=True, 
                        help="LPIPS 지각 손실 평가 수행 여부 (플래그 지정 시 True)")
    parser.add_argument("--lpips_net", type=str, default='alex', choices=['alex', 'vgg'], 
                        help="LPIPS 계산에 사용할 네트워크 (alex 또는 vgg)")
    parser.add_argument("--canny_low", type=int, default=15, 
                        help="Canny 엣지 검출기의 낮은 임계값")
    parser.add_argument("--canny_high", type=int, default=30, 
                        help="Canny 엣지 검출기의 높은 임계값")
    
    parser.add_argument("--output_csv", type=str, default="holdout_validation_results.csv", 
                        help="평가 결과를 저장할 CSV 파일 이름")
    
    args = parser.parse_args()
    holdout_validation_main(args)