import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import argparse
import os
import torchvision

from re_model import create_model 

def load_trained_model(checkpoint_path, device, encoder_name="resnet34"):
    print(f"=> '{encoder_name}' 인코더를 사용하는 모델 생성 중...")
    model = create_model(device=device, encoder_name=encoder_name)
    
    if not os.path.exists(checkpoint_path):
        print(f"오류: 체크포인트 파일 '{checkpoint_path}'를 찾을 수 없습니다. 경로를 확인해주세요.")
        raise FileNotFoundError(f"체크포인트 파일 없음: {checkpoint_path}")
        
    print(f"=> 체크포인트 불러오기: '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "gen_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["gen_state_dict"])
        print("Pix2Pix 생성자(Generator) 가중치를 로드했습니다.")
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
        print("일반 모델 가중치를 로드했습니다.")
    else:
        model.load_state_dict(checkpoint)
        print("모델 state_dict 자체를 로드했습니다.")
        
    model.eval() 
    print("모델 및 가중치 로드 완료. 추론 모드로 설정됨.")
    return model

def preprocess_image(image_path, image_size=256):
    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"오류: 입력 이미지 파일 '{image_path}'를 찾을 수 없습니다.")
        return None
    except Exception as e:
        print(f"오류: 입력 이미지 '{image_path}' 로드 중 문제 발생: {e}")
        return None

    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])
    img_tensor = transform(img)
    return img_tensor.unsqueeze(0) 

def postprocess_and_save_image(pred_tensor, output_path):
    if pred_tensor.dim() == 4 and pred_tensor.size(0) == 1:
        pred_tensor = pred_tensor.squeeze(0) 
    if pred_tensor.dim() == 3 and pred_tensor.size(0) == 1:
        pred_tensor = pred_tensor.squeeze(0)

    pred_tensor_normalized_for_save = (pred_tensor * 0.5) + 0.5
    pred_tensor_normalized_for_save = torch.clamp(pred_tensor_normalized_for_save, 0, 1)
    
    try:
        torchvision.utils.save_image(pred_tensor_normalized_for_save.cpu(), output_path)
        print(f"예측된 하이트맵 저장 완료: '{output_path}'")
    except Exception as e:
        print(f"오류: 예측 이미지 저장 중 문제 발생 ({output_path}): {e}")

def predict(args):
    DEVICE = "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    print(f"추론 장치: {DEVICE}")

    try:
        model = load_trained_model(args.checkpoint_path, DEVICE, args.encoder)
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return

    if not os.path.exists(args.input_image_path):
        print(f"오류: 입력 이미지 파일 '{args.input_image_path}'가 기본값으로 설정되었거나 존재하지 않습니다. --input_image_path를 지정해주세요.")
        return
    else:
        input_tensor = preprocess_image(args.input_image_path, args.image_size)

    if input_tensor is None:
        return 
    input_tensor = input_tensor.to(DEVICE)

    print(f"입력 이미지로 추론 시작: '{args.input_image_path}'")
    with torch.no_grad(): 
        prediction_tensor = model(input_tensor)
    print("추론 완료.")

    output_dir = os.path.dirname(args.output_image_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"출력 디렉토리 생성: '{output_dir}'")
        except OSError as e:
            print(f"오류: 출력 디렉토리 '{output_dir}' 생성 실패: {e}")
            print(f"결과를 현재 디렉토리에 '{os.path.basename(args.output_image_path)}' 이름으로 저장 시도합니다.")
            args.output_image_path = os.path.basename(args.output_image_path)
            
    postprocess_and_save_image(prediction_tensor, args.output_image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="학습된 모델로 하이트맵 추론")
    
    parser.add_argument("--checkpoint_path", type=str, 
                        default="training_outputs/efb7_lr1e-4_bs40_lpips_0/checkpoints/checkpoint_epoch_100.pth.tar",
                        help="학습된 모델의 체크포인트 파일 경로 (예: checkpoints/epoch_100.pth.tar)")
    parser.add_argument("--input_image_path", type=str, 
                        default="test_dataset/Carpet_Carpet4/input_Carpet4.png",
                        help="입력 재질 이미지 파일 경로 (예: test_images/fabric.png)")
    parser.add_argument("--output_image_path", type=str, 
                        default="test_dataset/output_Carpet4_700.png",
                        help="예측된 하이트맵을 저장할 경로 (기본값: 현재 폴더의 predicted_heightmap.png)")
    
    parser.add_argument("--encoder", type=str, default="efficientnet-b7", 
                        help="학습 시 사용한 U-Net의 인코더 이름 (기본값: resnet34)")
    parser.add_argument("--image_size", type=int, default=256, 
                        help="학습 시 사용한 이미지 리사이즈 크기 (기본값: 256)")
    parser.add_argument("--use_gpu", action='store_true', default=False,
                        help="GPU를 사용하여 추론 (플래그 지정 시 True, 기본값: False, 즉 CPU 사용)")
    
    args = parser.parse_args()


    predict(args)