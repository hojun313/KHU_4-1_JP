import torch
import torch.nn as nn # 모델 구조를 불러올 때 필요할 수 있음
from PIL import Image
import torchvision.transforms as T # torchvision.transforms를 T로 사용
import argparse
import os

# re_model.py에서 모델 생성 함수 임포트
# re_model.py 파일이 re_predict.py와 같은 디렉토리에 있거나,
# Python 경로에 포함되어 있어야 합니다.
from re_model import create_model 

def load_trained_model(checkpoint_path, device, encoder_name="resnet34"):
    """학습된 모델 가중치를 불러옵니다."""
    print(f"=> '{encoder_name}' 인코더를 사용하는 모델 생성 중...")
    model = create_model(device=device, encoder_name=encoder_name)
    
    print(f"=> 체크포인트 불러오기: '{checkpoint_path}'")
    # map_location을 사용하여 CPU에서도 GPU 학습 모델을 불러올 수 있도록 함
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # state_dict 키가 있는지 확인 (일반적인 경우)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # 체크포인트 파일이 모델의 state_dict 자체일 경우
        model.load_state_dict(checkpoint)
        
    model.eval() # 추론 모드로 설정 (Dropout, BatchNorm 등의 동작 변경)
    print("모델 및 가중치 로드 완료. 추론 모드로 설정됨.")
    return model

def preprocess_image(image_path, image_size=256):
    """입력 이미지를 모델에 맞게 전처리합니다."""
    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"오류: 입력 이미지 파일 '{image_path}'를 찾을 수 없습니다.")
        return None
    except Exception as e:
        print(f"오류: 입력 이미지 '{image_path}' 로드 중 문제 발생: {e}")
        return None

    # 학습 시 사용했던 Texture 이미지 변환과 동일하게 적용
    # (re_train.py의 transform_texture 참고)
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # -1 ~ 1 범위로 정규화
    ])
    img_tensor = transform(img)
    return img_tensor.unsqueeze(0) # 배치 차원 추가 (B, C, H, W)

def postprocess_and_save_image(pred_tensor, output_path):
    """예측된 텐서를 이미지로 변환하고 저장합니다."""
    # 모델 출력은 1채널, [-1, 1] 범위로 가정
    # 이를 [0, 1] 범위로 변환 후 [0, 255] 범위의 흑백 이미지로 저장
    
    # 배치 차원 제거 (만약 있다면)
    if pred_tensor.dim() == 4 and pred_tensor.size(0) == 1:
        pred_tensor = pred_tensor.squeeze(0) 
        
    # 채널 차원 제거 (만약 1채널이고 [1, H, W] 형태라면 [H, W]로)
    if pred_tensor.dim() == 3 and pred_tensor.size(0) == 1:
        pred_tensor = pred_tensor.squeeze(0)

    # [-1, 1] -> [0, 1] 범위로 변환
    pred_tensor = (pred_tensor * 0.5) + 0.5
    
    # 값 범위를 [0, 1]로 제한 (혹시 모를 오버플로우 방지)
    pred_tensor = torch.clamp(pred_tensor, 0, 1)
    
    # PIL 이미지로 변환 (텐서가 CPU에 있어야 함)
    # torchvision.transforms.ToPILImage()는 (C,H,W) 또는 (H,W) 텐서를 기대
    # 현재 pred_tensor는 (H,W) 또는 (C,H,W) 형태일 수 있음
    if pred_tensor.dim() == 2: # (H,W)
        pass # 그대로 사용
    elif pred_tensor.dim() == 3 and pred_tensor.size(0) == 1: # (1,H,W)
        pass # 그대로 사용 (ToPILImage가 처리 가능)
    else:
        print(f"경고: 예측 텐서의 형태가 예상과 다릅니다 ({pred_tensor.shape}). 첫 번째 채널만 사용합니다.")
        pred_tensor = pred_tensor[0] # 첫 번째 채널만 사용 가정

    try:
        # ToPILImage는 입력 텐서가 [0,1] 범위일 때 자동으로 [0,255] uint8로 변환하지 않음.
        # 명시적으로 변환하거나, save_image 사용.
        # 여기서는 torchvision.utils.save_image를 사용하여 간단히 저장.
        # save_image는 자동으로 정규화를 풀고 저장해줌 (normalize=True 기본값)
        # 또는, 직접 PIL 이미지로 변환 후 저장:
        #   img_pil = T.ToPILImage()(pred_tensor.cpu())
        #   img_pil.save(output_path)
        
        # torchvision.utils.save_image는 [-1,1] 범위의 텐서도 잘 처리함 (normalize=True 설정 시)
        # 하지만 우리는 이미 [0,1]로 만들었으므로 normalize=False로 하거나,
        # 아니면 모델 출력 그대로(-1~1)를 전달하고 normalize=True로 해도 됨.
        # 여기서는 [0,1]로 변환했으므로, save_image가 다시 정규화하지 않도록 normalize=False를 쓰거나,
        # 아니면 그냥 저장 (기본적으로 [0,1] 범위로 가정하고 저장)
        
        # 가장 간단한 방법: torchvision.utils.save_image 사용
        # 이 함수는 자동으로 0-1 범위로 클램핑하고 0-255로 스케일링하여 저장
        torchvision.utils.save_image(pred_tensor.cpu(), output_path)
        print(f"예측된 하이트맵 저장 완료: '{output_path}'")
    except Exception as e:
        print(f"오류: 예측 이미지 저장 중 문제 발생 ({output_path}): {e}")


def predict(args):
    DEVICE = "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    print(f"추론 장치: {DEVICE}")

    # 1. 모델 불러오기
    try:
        model = load_trained_model(args.checkpoint_path, DEVICE, args.encoder)
    except FileNotFoundError:
        print(f"오류: 체크포인트 파일 '{args.checkpoint_path}'를 찾을 수 없습니다.")
        return
    except Exception as e:
        print(f"오류: 모델 로드 중 문제 발생: {e}")
        return

    # 2. 입력 이미지 전처리
    input_tensor = preprocess_image(args.input_image_path, args.image_size)
    if input_tensor is None:
        return # 전처리 실패 시 종료
    input_tensor = input_tensor.to(DEVICE)

    # 3. 추론 실행
    print("추론 시작...")
    with torch.no_grad(): # 그래디언트 계산 비활성화
        prediction_tensor = model(input_tensor)
    print("추론 완료.")

    # 4. 결과 후처리 및 저장
    # 출력 디렉토리가 없으면 생성
    output_dir = os.path.dirname(args.output_image_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"출력 디렉토리 생성: '{output_dir}'")
        
    postprocess_and_save_image(prediction_tensor, args.output_image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="학습된 모델로 하이트맵 추론")
    
    # --- 경로 관련 인자 ---
    # 중요: 아래 기본값들은 예시이며, 실제 환경에 맞게 수정하거나,
    # 사용자가 항상 지정하도록 required=True를 유지하는 것이 더 안전할 수 있습니다.
    parser.add_argument("--checkpoint_path", type=str, 
                        default="default_checkpoints/latest_checkpoint.pth.tar",
                        help="학습된 모델의 체크포인트 파일 경로 (기본값: default_checkpoints/latest_checkpoint.pth.tar)")
    parser.add_argument("--input_image_path", type=str, 
                        default="default_input/sample_texture.png",
                        help="입력 재질 이미지 파일 경로 (기본값: default_input/sample_texture.png)")
    parser.add_argument("--output_image_path", type=str, 
                        default="./predicted_heightmap.png",
                        help="예측된 하이트맵을 저장할 경로 (기본값: 현재 폴더의 predicted_heightmap.png)")
    
    # --- 모델 및 추론 설정 관련 인자 ---
    parser.add_argument("--encoder", type=str, default="resnet34", 
                        help="학습 시 사용한 U-Net의 인코더 이름 (기본값: resnet34)")
    parser.add_argument("--image_size", type=int, default=256, 
                        help="학습 시 사용한 이미지 리사이즈 크기 (기본값: 256)")
    
    # 만약 기본으로 GPU를 사용하게 하려면 default=True로 하고 action을 바꾸거나 로직 수정 필요.
    parser.add_argument("--use_gpu", action='store_true', default=False,
                        help="GPU를 사용하여 추론 (플래그 지정 시 True, 기본값: False, 즉 CPU 사용)")
    
    args = parser.parse_args()


    predict(args)