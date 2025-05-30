# 0. 필요한 라이브러리 임포트
import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.optim as optim
import torch.nn.functional as F
# AMP (Automatic Mixed Precision) 임포트
from torch.amp import autocast # CUDA 컨텍스트에서 사용
from torch.cuda.amp import GradScaler
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler # 다른 Scheduler도 가능
from diffusers.optimization import get_cosine_schedule_with_warmup
from transformers import CLIPTextModel, CLIPTokenizer # 필요시
import tqdm
import numpy as np # 후처리를 위해
import traceback # 오류 추적을 위해
import sys # 종료 시 메시지 출력을 위해

import argparse


from dataset import GelSightDataset # dataset.py 파일이 동일 경로에 있다고 가정

# 2. 설정 및 하이퍼파라미터 정의
# 데이터셋 루트 디렉토리. dataset.py의 _build_image_pairs 메서드 구조에 맞아야 합니다.
data_root_directory = "heightmap_dataset" # 예시 경로

target_image_size = (256, 256) # 학습 및 추론 이미지 크기

# 학습 관련 설정
batch_size = 40 # GPU VRAM 크기에 맞춰 배치 크기 설정
num_workers = 0 # 데이터 로딩 워커 수 (Windows는 0으로 설정 권장)
num_epochs = 100 # 총 학습 에폭 수

use_data_augmentation = True # 데이터 증강 사용 여부 설정

# 옵티마이저 및 스케줄러 설정
learning_rate = 1e-5
adamw_betas = (0.9, 0.999)
adamw_weight_decay = 1e-2
num_warmup_steps_ratio = 0.1 # 총 스텝의 몇 %를 웜업으로 사용할지

# 체크포인트 설정
output_dir = "heightmap_diffusion_checkpoints" # 체크포인트 저장 디렉토리 (하이트맵용으로 이름 변경)
save_interval_steps = 250 # 몇 스텝마다 체크포인트를 저장할지

# 추론 설정
num_inference_steps = 100 # 추론 단계 수 (많을수록 품질 향상 가능성 있으나 시간 증가)
# <--- !!!! 추론할 입력 이미지 경로 수정 !!!! (이제 흑백 이미지도 가능, 내부적으로 흑백 처리 후 3채널로 변환됨)
input_image_for_generation = "./IOFiles/Input/input_Carpet3.png" # 예시 경로
# <--- !!!! 생성 이미지 저장 경로 설정 !!!! (생성된 흑백 하이트맵 저장 경로)
output_heightmap_image_path = "./IOFiles/Output/output_heightmap.png" # 예시 경로


# 3. 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용할 디바이스: {device}")

# AMP GradScaler 초기화 (학습 모드에서만 사용되지만, 미리 초기화)
scaler = None
if device == "cuda":
    scaler = GradScaler()
    print("AMP GradScaler 초기화 완료.")
else:
    print("CUDA 디바이스를 찾을 수 없어 AMP GradScaler를 사용하지 않습니다.")


# 4. 데이터셋 및 DataLoader 초기화 (학습 모드에서만 필요)
# 데이터 로더는 학습 모드 진입 시 사용자 입력 후 유효성이 확인되면 생성합니다.
dataset = None
dataloader = None
# 데이터 로더 생성 로직은 메인 실행 블록으로 이동합니다.


# 5. Stable Diffusion 모델 컴포넌트 로드
model_id = "runwayml/stable-diffusion-v1-5"
print(f"\n{model_id} 모델 컴포넌트 로드 중...")

vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
vae = vae.to(device)
vae.eval() # VAE는 학습하지 않고 인코더/디코더로만 사용

unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
unet = unet.to(device)

scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")

# Text Encoder 및 Tokenizer 로드 (텍스트 조건 사용 안 함 -> eval 모드 유지)
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
text_encoder = text_encoder.to(device)
text_encoder.eval() # 텍스트 조건 사용 안 하므로 eval mode

print("모델 컴포넌트 로드 완료.")

# 6. UNet 입력 채널 수정 (입력 이미지 조건 추가)
# dataset.py에서 흑백 하이트맵(타겟)을 3채널로 확장하여 VAE에 입력합니다.
# 또한, 입력 이미지(조건)도 흑백으로 로드 후 3채널로 확장하여 VAE에 입력합니다.
# 따라서 노이즈된 타겟 잠재 표현은 4채널이 됩니다.
# 입력 이미지(조건) 잠재 표현도 4채널입니다.
# UNet은 노이즈된 타겟 잠재 표현(4채널)과 입력 이미지 잠재 표현(4채널)을 concatenate하여 받으므로,
# UNet의 입력 채널은 4 + 4 = 8 채널이 됩니다.
original_in_channels = unet.conv_in.in_channels
new_in_channels = 4 + 4 # VAE latent (4) + Input Image latent (4) = 8 채널

if original_in_channels != new_in_channels:
    print(f"UNet 입력 채널 수정: {original_in_channels} -> {new_in_channels}")
    with torch.no_grad():
        original_weight = unet.conv_in.weight
        original_bias = unet.conv_in.bias

        new_conv_in = torch.nn.Conv2d(
            new_in_channels,
            unet.conv_in.out_channels,
            kernel_size=unet.conv_in.kernel_size,
            stride=unet.conv_in.stride,
            padding=unet.conv_in.padding,
            bias=unet.conv_in.bias is not None
        )
        # 기존 가중치를 앞쪽 절반(원본 4채널에 해당)에 복사합니다.
        # 새롭게 추가된 4채널은 기본값으로 초기화됩니다. (일반적으로 충분함)
        new_conv_in.weight.data[:, :original_in_channels, :, :].copy_(original_weight.data)
        if original_bias is not None:
            new_conv_in.bias.data.copy_(original_bias.data)

    unet.conv_in = new_conv_in.to(device)
    print("UNet 입력 채널 수정 완료.")
else:
    print(f"UNet 입력 채널이 이미 {new_in_channels}개입니다 (수정 불필요).")


# 7. 옵티마이저 및 학습률 스케줄러 설정 (학습 모드 진입 시 설정)
optimizer = None
lr_scheduler = None
total_train_steps = 0
num_warmup_steps = 0
# 옵티마이저/스케줄러 초기화 로직은 학습 모드 진입 시 수행됩니다.


# 8. 체크포인트 로드 로직 (메인 실행 블록으로 이동)
global_step = 0 # 전체 학습 스텝 카운터
start_epoch = 0 # 학습 재개 시 시작 에폭
progress_bar = None # tqdm 진행 바


# 9. 이미지 생성(추론) 유틸리티 함수 정의
# VAE 디코더 출력 후처리 파이프라인 (텐서 -> PIL Image 변환 및 정규화 해제 -> 흑백 변환)
def postprocess_heightmap_image(image_tensor):
    """
    VAE 디코더에서 나온 3채널 출력을 받아 흑백 하이트맵 PIL Image로 변환합니다.
    """
    # -1 ~ 1 범위를 0 ~ 1 범위로 변환
    image_tensor = (image_tensor / 2) + 0.5
    # 채널 순서 (C, H, W)를 이미지 순서 (H, W, C)로 변경
    image_tensor = image_tensor.permute(0, 2, 3, 1).clamp(0, 1) # 값 범위를 0~1로 클램핑

    # PyTorch 텐서를 NumPy 배열로 변환 (CPU로 이동 후)
    image_np = image_tensor.detach().cpu().numpy()

    # 0 ~ 1 범위를 0 ~ 255 범위로 스케일링하고 uint8 타입으로 변환
    image_np = (image_np * 255).astype(np.uint8)

    # NumPy 배열을 PIL Image로 변환 (현재 RGB처럼 3채널 데이터임)
    # dataset에서 1채널을 3채널로 복사했기 때문에, R, G, B 채널 값은 모두 같을 것입니다.
    # PIL의 convert("L") 사용이 가장 간단하고 안전하게 흑백으로 변환합니다.
    if image_np.shape[0] > 0: # 배치에 이미지가 하나라도 있는 경우
        # 배치 단위 추론 시에도 첫 번째 이미지만 처리하도록 구현됨
        image_pil = Image.fromarray(image_np[0]).convert("L") # 최종적으로 흑백 (1채널) PIL 이미지
        # print("DEBUG: postprocess_heightmap_image 처리 완료.")
    else:
        print("경고: postprocess_heightmap_image에 빈 배치 텐서가 전달되었습니다.")
        return None
    return image_pil


def generate_heightmap_image(input_image_path, vae, unet, scheduler, device, target_image_size, output_path="generated_heightmap.png", num_inference_steps=50):
    """
    입력 이미지를 사용하여 흑백 하이트맵 이미지를 생성하는 함수.
    입력 이미지는 내부적으로 흑백으로 변환된 후 VAE 처리를 위해 3채널로 확장됩니다.

    Args:
        input_image_path (str): 하이트맵을 생성할 입력 이미지 파일 경로 (컬러/흑백 무관).
        vae (AutoencoderKL): 학습된 VAE 모델.
        unet (UNet2DConditionModel): 학습된 UNet 모델.
        scheduler (SchedulerMixin): Diffusion 스케줄러.
        device (str): 모델이 로드된 디바이스 이름 (예: 'cuda', 'cpu').
        target_image_size (tuple): 모델 학습에 사용된 이미지 크기.
        output_path (str): 생성된 흑백 하이트맵 이미지를 저장할 경로.
        num_inference_steps (int): 추론 단계 수.
    """
    print(f"\n흑백 하이트맵 이미지 생성 시작 (입력: {input_image_path})")
    print(f"생성된 이미지는 다음 경로에 저장될 예정입니다: {output_path}")


    # 추론을 위한 전처리 파이프라인 (입력 이미지용)
    # 입력 이미지는 흑백 변환 후 3채널로 확장되므로, Normalize는 3채널 기준으로 수행
    inference_transform = T.Compose([
        T.Resize(target_image_size),
        T.ToTensor(), # (C, H, W) 형태로 변환. C=3 예상.
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 3채널 정규화
    ])

    # 모델을 평가 모드로 설정
    unet.eval()
    vae.eval()
    text_encoder.eval()

    # 1. 입력 이미지 로드 및 전처리
    try:
        # 이미지를 흑백('L')으로 로드한 후, VAE 입력을 위해 RGB('RGB') 형태로 변환 (R=G=B가 됨)
        input_image_pil = Image.open(input_image_path).convert("RGB")
        input_image_tensor = inference_transform(input_image_pil).unsqueeze(0).to(device) # 배치 차원 추가 및 디바이스 이동
    except Exception as e:
        print(f"오류: 입력 이미지 로드 또는 전처리 중 문제 발생 - {input_image_path}")
        print(f"오류 내용: {e}")
        return # 함수 종료

    # 2. 입력 이미지 잠재 공간 변환 (VAE 인코딩)
    with torch.no_grad():
        # 3채널 (흑백이 복제된) 입력 이미지를 VAE로 인코딩 -> 4채널 잠재
        input_image_latent = vae.encode(input_image_tensor).latent_dist.sample() * vae.config.scaling_factor
        # print(f"DEBUG: Input image latent shape: {input_image_latent.shape}")


    # 3. 초기 노이즈 생성 (타겟 하이트맵의 잠재 공간 형태에 맞춤)
    # 타겟(하이트맵)은 흑백을 3채널로 확장해서 VAE에 넣었으므로 Latent shape는 [1, 4, H/8, W/8] 입니다.
    latent_shape = input_image_latent.shape # 입력 이미지 Latent와 동일한 형태 사용
    # device 문자열을 torch.device 객체로 변환하여 사용
    torch_device_obj = torch.device(device)
    generator = torch.Generator(device=torch_device_obj).manual_seed(42) # 재현 가능한 결과 생성을 위한 시드 고정 (선택 사항)
    initial_noise = torch.randn(latent_shape, generator=generator, device=torch_device_obj) # 초기 노이즈 생성 (4채널)


    # 4. Scheduler에 노이즈 제거 단계 설정
    scheduler.set_timesteps(num_inference_steps, device=torch_device_obj) # torch.device 객체 사용
    current_latent = initial_noise

    # 5. 반복적 노이즈 제거 (Sampling Loop)
    print(f"Diffusion 샘플링 시작 ({num_inference_steps} 스텝)...")
    # AMP 추론 사용 (메모리 및 속도 향상) - device가 "cuda"일 때만 autocast 사용
    context_manager = autocast(device_type="cuda") if device == "cuda" else torch.no_grad()


    with context_manager:
        for t in tqdm.tqdm(scheduler.timesteps, desc="Diffusion Sampling"):
            # UNet 입력 준비: 노이즈가 추가된 하이트맵 잠재 표현 + 입력 이미지 잠재 표현
            unet_input = torch.cat([current_latent, input_image_latent], dim=1) # 4 + 4 = 8 채널

            # UNet에 입력하여 노이즈 예측
            timesteps_tensor = t.to(torch_device_obj).unsqueeze(0) # 스칼라 t를 [1] 텐서로 변환 (배치 크기 1 가정)
            with torch.no_grad(): # 추론 시에는 기울기 계산 필요 없음
                # 더미 조건부 텐서 생성 및 전달 (학습 때와 동일하게)
                current_batch_size = unet_input.shape[0]
                cross_attention_dim = unet.config.cross_attention_dim
                sequence_length = 77 # CLIP 기본값

                dummy_encoder_hidden_states = torch.zeros(
                    current_batch_size, sequence_length, cross_attention_dim, device=torch_device_obj, dtype=unet_input.dtype
                )

                # UNet forward 호출 시 더미 조건부 텐서 전달
                model_pred = unet(
                    unet_input,
                    timesteps_tensor,
                    encoder_hidden_states=dummy_encoder_hidden_states
                ).sample # 예측된 노이즈 (4채널)

                # Scheduler를 사용하여 노이즈 제거
                scheduler_output = scheduler.step(model_pred, t, current_latent)
                current_latent = scheduler_output.prev_sample # 다음 스텝의 잠재 표현


    # 6. 최종 하이트맵 이미지 잠재 표현 디코딩 (VAE 디코더)
    with torch.no_grad():
        final_heightmap_latent = current_latent / vae.config.scaling_factor
        # VAE 디코더는 4채널 latent를 받아 3채널 이미지를 출력합니다.
        generated_image_tensor_3channel = vae.decode(final_heightmap_latent).sample # 디코딩 결과 (3채널 픽셀 텐서)
        # print(f"DEBUG: VAE decoder output shape: {generated_image_tensor_3channel.shape}")


    # 7. 후처리 및 저장 (3채널 출력을 흑백 1채널로 변환)
    generated_heightmap_pil = postprocess_heightmap_image(generated_image_tensor_3channel)

    if generated_heightmap_pil:
        # 이미지 저장
        os.makedirs(os.path.dirname(output_path), exist_ok=True) # 출력 디렉토리 생성
        generated_heightmap_pil.save(output_path) # 흑백 이미지로 저장
        print(f"흑백 하이트맵 이미지 생성 완료 및 저장: {output_path}")
    else:
        print("경고: 이미지 후처리 실패 또는 빈 이미지 데이터로 인해 저장을 건너뜁니다.")

    return generated_heightmap_pil # 생성된 PIL Image 객체 반환 (선택 사항)


# 10. 체크포인트 선택 함수 (새로 추가)
def select_checkpoint_interactive(base_dir):
    """
    지정된 기본 디렉토리에서 체크포인트 폴더 목록을 사용자에게 보여주고 선택받습니다.
    선택된 체크포인트 폴더의 전체 경로를 반환합니다.
    """
    if not os.path.isdir(base_dir):
        print(f"오류: 체크포인트 기본 디렉토리 '{base_dir}'를 찾을 수 없습니다.")
        return None

    try:
        sub_items = os.listdir(base_dir)
        # "checkpoint-"로 시작하거나 "final_model"인 디렉토리만 필터링
        checkpoint_dirs_filtered = [
            d for d in sub_items
            if os.path.isdir(os.path.join(base_dir, d)) and \
               (d.startswith("checkpoint-") or d == "final_model")
        ]
    except Exception as e:
        print(f"오류: '{base_dir}' 디렉토리 접근 중 문제 발생: {e}")
        return None

    if not checkpoint_dirs_filtered:
        print(f"'{base_dir}' 디렉토리에 유효한 체크포인트 폴더('checkpoint-*' 또는 'final_model')가 없습니다.")
        return None

    print("\n사용 가능한 체크포인트 폴더:")
    
    def get_sort_key(dir_name):
        if dir_name == "final_model":
            return float('inf') # final_model을 가장 마지막에 정렬
        if dir_name.startswith("checkpoint-"):
            try:
                return int(dir_name.split("-")[-1])
            except ValueError:
                return -1 # 숫자로 변환할 수 없는 경우, 앞으로 정렬
        return -2 # 그 외의 경우 (거의 발생 안 함)

    sorted_checkpoint_dirs = sorted(checkpoint_dirs_filtered, key=get_sort_key)

    for i, dir_name in enumerate(sorted_checkpoint_dirs):
        print(f"{i + 1}: {dir_name}")

    while True:
        try:
            choice_str = input(f"로드할 체크포인트 번호를 입력하세요 (1-{len(sorted_checkpoint_dirs)}), 또는 취소하려면 'c' 입력: ")
            if not choice_str:
                print("입력이 없습니다. 다시 시도하세요.")
                continue
            if choice_str.lower() == 'c':
                print("체크포인트 선택이 취소되었습니다.")
                return None

            choice_idx = int(choice_str) - 1
            if 0 <= choice_idx < len(sorted_checkpoint_dirs):
                selected_dir_name = sorted_checkpoint_dirs[choice_idx]
                selected_path = os.path.join(base_dir, selected_dir_name)
                # 선택된 경로가 실제로 디렉토리인지 한 번 더 확인
                if os.path.isdir(selected_path):
                    return selected_path
                else: 
                    print(f"오류: 선택된 '{selected_dir_name}'은(는) 유효한 디렉토리가 아닙니다. 목록이 변경되었을 수 있습니다. 다시 시도하세요.")
                    # 이 경우 목록을 다시 로드하는 것이 좋을 수 있지만, 여기서는 간단히 오류 메시지 표시
                    return None # 또는 루프 계속
            else:
                print(f"잘못된 번호입니다. 1부터 {len(sorted_checkpoint_dirs)} 사이의 숫자를 입력하거나 'c'를 입력하세요.")
        except ValueError:
            print("숫자를 입력하거나 'c'를 입력해야 합니다.")
        except Exception as e:
            print(f"선택 중 오류 발생: {e}")
            return None
    return None # 이 줄은 도달하지 않아야 함


# 11. 메인 실행 블록
if __name__ == "__main__":
    print("스크립트 실행 시작")

    # --- 커맨드 라인 인자 파서 설정 ---
    parser = argparse.ArgumentParser(description="Train Heightmap Diffusion Model")

    # 데이터셋 및 이미지 관련 인자
    parser.add_argument('--data_root', type=str, default="heightmap_dataset", help="Path to the dataset root directory")
    parser.add_argument('--image_height', type=int, default=256, help="Target image height")
    parser.add_argument('--image_width', type=int, default=256, help="Target image width")
    parser.add_argument('--use_data_augmentation', type=lambda x: (str(x).lower() == 'true'), default=False, help="Enable data augmentation (true/false)")

    # 학습 관련 인자
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training") # 스크립트의 원래 기본값
    parser.add_argument('--num_workers', type=int, default=0, help="Number of data loading workers") # 스크립트의 원래 기본값
    parser.add_argument('--num_epochs', type=int, default=100, help="Total number of training epochs")

    # 옵티마이저 및 스케줄러 관련 인자
    parser.add_argument('--learning_rate', type=float, default=1e-5, help="Initial learning rate")
    parser.add_argument('--adam_beta1', type=float, default=0.9, help="AdamW beta1 parameter")
    parser.add_argument('--adam_beta2', type=float, default=0.999, help="AdamW beta2 parameter")
    parser.add_argument('--adamw_weight_decay', type=float, default=1e-2, help="AdamW weight_decay parameter")
    parser.add_argument('--num_warmup_steps_ratio', type=float, default=0.1, help="Ratio of total training steps for warmup")

    # 체크포인트 관련 인자
    parser.add_argument('--output_dir', type=str, default="heightmap_diffusion_checkpoints", help="Directory to save checkpoints")
    parser.add_argument('--save_interval_steps', type=int, default=250, help="Save checkpoint every N steps")

    # 추론 관련 인자 (필요에 따라 추가)
    parser.add_argument('--num_inference_steps', type=int, default=100, help="Number of steps for inference")
    parser.add_argument('--input_image_for_generation', type=str, default="./IOFiles/Input/input_Carpet3.png", help="Input image path for generation")
    parser.add_argument('--output_heightmap_image_path', type=str, default="./IOFiles/Output/output_heightmap.png", help="Base path to save generated heightmap image")
    
    parser.add_argument('--mode', type=str, choices=['1', '2', '3'], required=True, help="Execution mode: 1 (new train), 2 (resume), 3 (infer)")

    parser.add_argument('--excluded_materials', type=str, nargs='*', default=None, help="List of materials to exclude from training")
    parser.add_argument('--model_tag', type=str, default="", help="A tag to append to the output directory for this specific run (e.g., excluded material name)")

    args = parser.parse_args()

    # --- 파싱된 인자들을 스크립트 변수에 할당 ---
    # 이 변수들은 스크립트 상단에 정의된 전역 변수들을 이 블록 내에서 덮어쓰거나 새로운 지역 변수로 사용됩니다.
    # 스크립트의 다른 부분(함수 등)에서 이 변수들을 사용하려면, 함수 인자로 전달하거나,
    # 정말로 전역적으로 변경해야 한다면 `global` 키워드를 사용해야 할 수 있으나,
    # 현재 구조에서는 __main__ 블록 내의 로직에 주로 사용되므로 이 방식이 괜찮을 수 있습니다.

    # 데이터셋 및 이미지 설정
    data_root_directory = args.data_root
    target_image_size = (args.image_height, args.image_width) # 튜플로 재구성
    use_data_augmentation = args.use_data_augmentation       # 위에서 lambda로 boolean 처리

    # 학습 설정
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_epochs = args.num_epochs

    # 옵티마이저 및 스케줄러 설정
    learning_rate = args.learning_rate
    adamw_betas = (args.adam_beta1, args.adam_beta2)       # 튜플로 재구성
    adamw_weight_decay = args.adamw_weight_decay
    num_warmup_steps_ratio = args.num_warmup_steps_ratio

    # 체크포인트 설정
    output_dir = args.output_dir
    save_interval_steps = args.save_interval_steps

    # 추론 설정 (argparse에서 직접 할당받은 변수 사용)
    num_inference_steps = args.num_inference_steps
    input_image_for_generation = args.input_image_for_generation
    output_heightmap_image_path = args.output_heightmap_image_path # 기본 경로 업데이트+

    # 모드 설정
    mode = args.mode
    # mode = "1"

    # 제외할 재료 목록 (선택적)
    excluded_materials_list = args.excluded_materials
    model_tag = args.model_tag

    base_output_dir = args.output_dir
    if model_tag:
        current_run_output_dir = os.path.join(base_output_dir, model_tag)
    else:
        current_run_output_dir = os.path.join(base_output_dir, "default_run_without_tag")

    os.makedirs(current_run_output_dir, exist_ok=True)
    print(f"  Output Directory for this run: {current_run_output_dir}")
    

    

    print(f"스크립트 실행 파라미터:")
    print(f"  Data Root: {data_root_directory}")
    print(f"  Target Image Size: {target_image_size}")
    print(f"  Use Data Augmentation: {use_data_augmentation}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Num Workers: {num_workers}")
    print(f"  Num Epochs: {num_epochs}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  AdamW Betas: {adamw_betas}")
    print(f"  AdamW Weight Decay: {adamw_weight_decay}")
    print(f"  Warmup Steps Ratio: {num_warmup_steps_ratio}")
    print(f"  Output Directory: {output_dir}")
    print(f"  Save Interval Steps: {save_interval_steps}")
    print(f"  Num Inference Steps: {num_inference_steps}")
    print(f"  Input Image (Gen): {input_image_for_generation}")
    print(f"  Output Heightmap Path (Gen): {output_heightmap_image_path}")


    # 추론 시 실제 사용될 출력 경로 (파싱된 기본 경로로 초기화)
    actual_inference_output_path = output_heightmap_image_path # 이제 args에서 온 값 사용

    # --- 학습 모드 또는 추론 모드 선택 ---
    # mode_arg = args.mode # 만약 mode도 인자로 받았다면 사용
    # 현재는 기존 input 방식 유지
    # print("\n실행 모드를 선택하세요:")
    # print("1: 모델 학습 시작")
    # print("2: 체크포인트 로드 후 학습 재개")
    # print("3: 체크포인트 로드 후 추론만 수행")
    # mode = input("모드 입력 (1, 2, 3): ")


    is_training = False
    is_inference_only = False
    resume_from_checkpoint = None

    if mode == '1': # 학습 모드 (새로운 학습)
        is_training = True
        is_inference_only = False
        resume_from_checkpoint = None # 새롭게 학습 시작이므로 체크포인트 로드 안 함
        print(">> 새로운 학습 모드 시작.")

    elif mode == '2': # 학습 재개 모드
        is_training = True
        is_inference_only = False
        print(f"\n'{output_dir}' 디렉토리에서 학습 재개할 체크포인트를 선택합니다.")
        resume_from_checkpoint = select_checkpoint_interactive(output_dir)
        if resume_from_checkpoint is None:
            print("체크포인트 선택이 이루어지지 않았거나 취소되었습니다. 프로그램을 종료합니다.")
            sys.exit() # 프로그램 종료
        print(f">> 학습 재개 모드 시작. 체크포인트 경로: {resume_from_checkpoint}")

    elif mode == '3': # 추론 모드
        is_training = False
        is_inference_only = True
        print(f"\n'{output_dir}' 디렉토리에서 추론에 사용할 체크포인트를 선택합니다.")
        resume_from_checkpoint = select_checkpoint_interactive(output_dir)
        if resume_from_checkpoint is None:
            print("체크포인트 선택이 이루어지지 않았거나 취소되었습니다. 프로그램을 종료합니다.")
            sys.exit() # 프로그램 종료
        print(f">> 추론 모드 시작. 체크포인트 경로: {resume_from_checkpoint}")

        # 모드 3 선택 및 체크포인트 로드 성공 시, 추론 출력 파일명 수정
        if resume_from_checkpoint:
            checkpoint_name = os.path.basename(resume_from_checkpoint)
            step_identifier = ""

            if checkpoint_name == "final_model":
                step_identifier = "final"
            elif checkpoint_name.startswith("checkpoint-"):
                try:
                    # "checkpoint-123" -> "123"
                    step_identifier = checkpoint_name.split("-")[-1]
                    int(step_identifier) # 숫자형인지 확인
                except (IndexError, ValueError):
                    print(f"경고: 체크포인트 이름 '{checkpoint_name}'에서 유효한 스텝 번호를 추출할 수 없습니다. 기본 출력 파일명을 사용합니다.")
                    step_identifier = "" # 추출 실패 시 식별자 비움

            if step_identifier:
                # 기본 설정된 output_heightmap_image_path를 기반으로 새 경로 생성
                output_folder_for_inference = os.path.dirname(output_heightmap_image_path) # 예: "./IOFiles/Output"
                original_extension = os.path.splitext(output_heightmap_image_path)[1]      # 예: ".png"
                
                new_filename = f"output_heightmap_{step_identifier}{original_extension}" # 예: "output_heightmap_7500.png"
                actual_inference_output_path = os.path.join(output_folder_for_inference, new_filename)
                # print(f"추론 결과는 다음 경로에 저장됩니다: {actual_inference_output_path}") # generate_heightmap_image 함수 시작 시 경로 출력
        
    else:
        print("잘못된 모드 입력입니다. 프로그램을 종료합니다.")
        sys.exit() # 프로그램 종료


    # --- 체크포인트 로드 실행 (선택된 모드에 따라 resume_from_checkpoint 값이 다름) ---
    # 학습 또는 추론 전에 모델 상태를 로드합니다.
    if resume_from_checkpoint is not None:
        print(f"\n체크포인트 로드 중... 경로: {resume_from_checkpoint}")

        # UNet 모델 상태 로드 (필수)
        unet_load_path = os.path.join(resume_from_checkpoint, "unet")
        if os.path.exists(unet_load_path) and os.path.isdir(unet_load_path): # 디렉토리인지 확인
            try:
                unet = UNet2DConditionModel.from_pretrained(
                    unet_load_path,
                    in_channels=new_in_channels, # 수정된 UNet의 입력 채널 수 (8)
                    ignore_mismatched_sizes=True, # conv_in 레이어 크기 불일치 무시
                    low_cpu_mem_usage=False
                ).to(device)
                print("- UNet 로드 완료")
            except Exception as e:
                print(f"- 오류: UNet 체크포인트 로드 중 문제 발생 ('{unet_load_path}'): {e}")
                print("  기존 UNet 모델을 계속 사용합니다 (만약 초기화되었다면). 문제가 지속되면 체크포인트를 확인하세요.")
                # 로드 실패 시, 프로그램 종료 또는 기본 모델 사용 결정 가능
                # 여기서는 일단 진행하되, 학습/추론 시 문제가 될 수 있음을 인지
        else:
            print(f"- 오류: UNet 체크포인트 디렉토리 '{unet_load_path}'를 찾을 수 없거나 유효하지 않습니다. 실행 불가.")
            sys.exit() # 프로그램 종료

        # Optimizer, Scheduler 상태 로드 (학습 재개 시 필수)
        if is_training: # 학습 모드일 때만 옵티마이저/스케줄러 상태 로드
            print("학습 재개를 위해 데이터셋 및 DataLoader를 먼저 초기화합니다 (필요시).")
            if dataset is None or dataloader is None: # 아직 초기화되지 않은 경우
                try:
                    dataset = GelSightDataset(data_root_directory, target_size=target_image_size, augment=use_data_augmentation, excluded_materials=excluded_materials_list)
                    if len(dataset) > 0:
                        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device == 'cuda'))
                        print("DataLoader 생성 완료.")
                    else:
                        print("데이터셋에 유효한 샘플이 없어 학습용 DataLoader를 생성하지 않았습니다.")
                        dataset = None 
                        dataloader = None
                except Exception as e:
                    print(f"데이터셋 또는 DataLoader 구축 중 오류 발생: {e}")
                    traceback.print_exc()
                    dataset = None
                    dataloader = None
            
            if dataloader is not None:
                # DataLoader 생성 후 Optimizer 및 Scheduler 초기화 (만약 아직 안됐다면)
                if optimizer is None: 
                    optimizer = optim.AdamW(
                        unet.parameters(), lr=learning_rate, betas=adamw_betas, weight_decay=adamw_weight_decay
                    )
                    print("학습 재개를 위해 Optimizer 초기화 완료.")

                if lr_scheduler is None: 
                    num_update_steps_per_epoch = len(dataloader)
                    total_train_steps = num_epochs * num_update_steps_per_epoch
                    num_warmup_steps = int(num_warmup_steps_ratio * total_train_steps)
                    lr_scheduler = get_cosine_schedule_with_warmup(
                        optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_train_steps
                    )
                    print("학습 재개를 위해 LR Scheduler 초기화 완료.")
                    print(f"총 학습 스텝: {total_train_steps}, 웜업 스텝: {num_warmup_steps}")
                
                # 이제 초기화된 옵티마이저/스케줄러에 상태 로드
                optimizer_load_path = os.path.join(resume_from_checkpoint, "optimizer.pt")
                if os.path.exists(optimizer_load_path):
                    try:
                        optimizer.load_state_dict(torch.load(optimizer_load_path, map_location=device))
                        print("- Optimizer 상태 로드 완료")
                    except Exception as e:
                         print(f"- 경고: Optimizer 상태 로드 중 오류 발생 ('{optimizer_load_path}'): {e}. Optimizer 상태가 초기화될 수 있습니다.")
                else:
                    print(f"- 경고: Optimizer 체크포인트 '{optimizer_load_path}'를 찾을 수 없습니다. Optimizer가 새로 시작됩니다.")

                lr_scheduler_load_path = os.path.join(resume_from_checkpoint, "lr_scheduler.pt")
                if os.path.exists(lr_scheduler_load_path):
                    try:
                        lr_scheduler.load_state_dict(torch.load(lr_scheduler_load_path, map_location=device))
                        print("- LR Scheduler 상태 로드 완료")
                    except Exception as e:
                        print(f"- 경고: LR Scheduler 상태 로드 중 오류 발생 ('{lr_scheduler_load_path}'): {e}. Scheduler 상태가 초기화될 수 있습니다.")
                else:
                    print(f"- 경고: LR Scheduler 체크포인트 '{lr_scheduler_load_path}'를 찾을 수 없습니다. Scheduler가 새로 시작됩니다.")

                # Global step 로드 (학습 재개 시 필수)
                step_load_path = os.path.join(resume_from_checkpoint, "global_step.txt")
                if os.path.exists(step_load_path):
                    try:
                        with open(step_load_path, "r") as f:
                            loaded_step = int(f.read())
                        global_step = loaded_step # 로드된 스텝으로 업데이트
                        print(f"- Global step 로드 완료. 학습 스텝을 {global_step} 부터 시작합니다.")
                        
                        if dataloader is not None and len(dataloader) > 0:
                            start_epoch = global_step // len(dataloader)
                            print(f"- 대략적인 시작 에폭: {start_epoch}")
                        else:
                            start_epoch = 0 # DataLoader 없으면 에폭 계산 불가
                            print("- 경고: DataLoader가 유효하지 않아 시작 에폭을 0으로 설정합니다.")
                    except Exception as e:
                        print(f"- 경고: Global step 로드 중 오류 발생 ('{step_load_path}'): {e}. 스텝이 0부터 시작됩니다.")
                        global_step = 0
                        start_epoch = 0
                else:
                    print(f"- 경고: Global step 체크포인트 '{step_load_path}'를 찾을 수 없습니다. 학습 스텝이 0부터 시작됩니다.")
                    global_step = 0
                    start_epoch = 0
            else: # dataloader is None
                print("- 학습 재개 모드이나 DataLoader가 유효하지 않아 학습 관련 상태(Optimizer, Scheduler, step) 로드를 건너뜁니다.")
                is_training = False # 학습 모드 취소


        elif is_inference_only:
            # 추론 모드에서는 옵티마이저/스케줄러 상태는 로드하지 않습니다.
            print("- 추론 모드이므로 Optimizer 및 LR Scheduler 상태는 로드하지 않습니다.")
            global_step = 0 
            start_epoch = 0

        print("체크포인트 관련 로드 절차 완료.")
    else: # 체크포인트를 로드하지 않는 경우 (새로운 학습 시작)
        print("체크포인트를 로드하지 않습니다 (새로운 학습 시작).")
        global_step = 0 
        start_epoch = 0

        if is_training:
            # 새로운 학습 시작 시 데이터셋, DataLoader, Optimizer, Scheduler 초기화
            print("새로운 학습을 위해 데이터셋, DataLoader, Optimizer, Scheduler를 초기화합니다.")
            if dataset is None or dataloader is None: # 아직 초기화되지 않은 경우
                try:
                    # !!!! 중요 !!!!
                    # GelSightDataset은 dataset.py에서 input_images를 흑백으로 로드 후 3채널로 변환하도록 수정되어야 합니다.
                    dataset = GelSightDataset(data_root_directory, target_size=target_image_size, augment=use_data_augmentation)
                    if len(dataset) > 0:
                        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device == 'cuda'))
                        print("DataLoader 생성 완료.")
                    else:
                        print("데이터셋에 유효한 샘플이 없어 학습용 DataLoader를 생성하지 않았습니다.")
                        dataset = None
                        dataloader = None # 학습 불가
                except Exception as e:
                    print(f"데이터셋 또는 DataLoader 구축 중 오류 발생: {e}")
                    traceback.print_exc()
                    dataset = None
                    dataloader = None

            if dataloader is not None:
                if optimizer is None: 
                    optimizer = optim.AdamW(
                        unet.parameters(), lr=learning_rate, betas=adamw_betas, weight_decay=adamw_weight_decay
                    )
                    print("Optimizer 초기화 완료.")

                if lr_scheduler is None: 
                    num_update_steps_per_epoch = len(dataloader)
                    total_train_steps = num_epochs * num_update_steps_per_epoch
                    num_warmup_steps = int(num_warmup_steps_ratio * total_train_steps)
                    lr_scheduler = get_cosine_schedule_with_warmup(
                        optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_train_steps
                    )
                    print("LR Scheduler 초기화 완료.")
                    print(f"총 학습 스텝: {total_train_steps}, 웜업 스텝: {num_warmup_steps}")
            else: # dataloader is None
                print("- 새로운 학습 모드이나 DataLoader가 유효하지 않아 학습을 시작할 수 없습니다.")
                is_training = False # 학습 모드 취소


    # --- 선택된 모드에 따라 학습 또는 추론 실행 ---
    if is_training and dataloader is not None and optimizer is not None and lr_scheduler is not None and \
       (device == 'cpu' or (device == 'cuda' and scaler is not None)): # scaler 조건은 CUDA일 때만 확인
        print("\n모델 학습 시작!")

        if total_train_steps == 0 and dataloader is not None : # total_train_steps가 아직 계산 안된 경우 (재개 시 등)
             num_update_steps_per_epoch = len(dataloader)
             total_train_steps = num_epochs * num_update_steps_per_epoch # 남은 에폭 기준이 아닌 총 에폭 기준
             # 만약 재개 시 남은 스텝만 계산하려면 다르게 접근해야 함. 여기서는 전체 스텝으로 가정.
             print(f"총 학습 스텝 재계산 (필요시): {total_train_steps}")


        if total_train_steps == 0 : # DataLoader가 없거나 해서 여전히 0이면
            print("오류: 총 학습 스텝이 0입니다. DataLoader 초기화에 문제가 있었을 수 있습니다.")
            is_training = False 
        else:
            if progress_bar is None:
                progress_bar = tqdm.tqdm(total=total_train_steps, initial=global_step, desc="학습 진행")
            else: # 이미 생성된 경우 업데이트 (재개 시)
                progress_bar.total = total_train_steps
                progress_bar.initial = global_step
                progress_bar.n = global_step
                progress_bar.refresh()


        if is_training: 
            # torch.device 객체를 학습 루프 전에 한 번 생성
            torch_device_obj_train = torch.device(device)
            for epoch in range(start_epoch, num_epochs):
                unet.train() 
                vae.eval() 
                text_encoder.eval() 

                for step, batch_data in enumerate(dataloader):
                    if batch_data is None or batch_data[0] is None: # dataset에서 오류로 None 반환 시
                        print(f"경고: [Epoch {epoch+1}, Step {step+1}] 유효하지 않은 배치 데이터 수신. 건너뜁니다.")
                        if progress_bar: progress_bar.update(0) # 진행바는 업데이트하지 않거나, 오류 카운트
                        continue

                    # 현재 배치의 전역 스텝 번호는 global_step으로 관리
                    if global_step >= total_train_steps and mode != '1': # 재개 시 이미 완료된 경우
                        print(f"로드된 global_step ({global_step})이 total_train_steps ({total_train_steps}) 이상입니다. 학습을 종료합니다.")
                        is_training = False
                        break # 내부 루프 종료

                    try: 
                        # input_images와 gelsight_images는 dataset.py에서 이미 3채널 텐서로 변환되어 제공됨
                        # (input_images도 이제 흑백 원본 -> 3채널 텐서로 가정)
                        input_images, gelsight_images = batch_data
                        input_images = input_images.to(torch_device_obj_train) # torch.device 객체 사용
                        gelsight_images = gelsight_images.to(torch_device_obj_train) # torch.device 객체 사용

                        with torch.no_grad(): 
                            input_images_latent = vae.encode(input_images).latent_dist.sample() * vae.config.scaling_factor
                            gelsight_images_latent = vae.encode(gelsight_images).latent_dist.sample() * vae.config.scaling_factor
                        
                        noise = torch.randn_like(gelsight_images_latent) 
                        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (gelsight_images_latent.shape[0],), device=torch_device_obj_train).long() # torch.device 객체 사용
                        noisy_gelsight_latent = scheduler.add_noise(gelsight_images_latent, noise, timesteps) 
                        
                        unet_input = torch.cat([noisy_gelsight_latent, input_images_latent], dim=1) 
                        
                        # AMP 사용 설정 (CUDA일 때만)
                        amp_context = autocast(device_type="cuda") if device == "cuda" else torch.no_grad()


                        with amp_context:
                            current_batch_size = unet_input.shape[0]
                            cross_attention_dim = unet.config.cross_attention_dim
                            sequence_length = 77 

                            dummy_encoder_hidden_states = torch.zeros(
                                current_batch_size, sequence_length, cross_attention_dim, device=torch_device_obj_train, dtype=unet_input.dtype # torch.device 객체 사용
                            )
                            
                            model_pred = unet(
                                unet_input,
                                timesteps,
                                encoder_hidden_states=dummy_encoder_hidden_states
                            ).sample 
                            
                            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                        if device == "cuda" and scaler is not None:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else: # CPU 또는 AMP 미사용
                            loss.backward()
                            optimizer.step()
                        
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        
                        global_step += 1
                        
                        if progress_bar:
                            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "epoch": epoch +1}
                            progress_bar.update(1)
                            progress_bar.set_postfix(**logs)
                        
                        if global_step % save_interval_steps == 0:
                            checkpoint_dir = os.path.join(current_run_output_dir, f"checkpoint-{global_step}")
                            os.makedirs(checkpoint_dir, exist_ok=True)

                            print(f"\n{global_step} 스텝에서 체크포인트 저장 중... 경로: {checkpoint_dir}")
                            unet_save_path = os.path.join(checkpoint_dir, "unet")
                            unet.save_pretrained(unet_save_path)

                            if optimizer is not None:
                                optimizer_save_path = os.path.join(checkpoint_dir, "optimizer.pt")
                                torch.save(optimizer.state_dict(), optimizer_save_path)
                            if lr_scheduler is not None:
                                lr_scheduler_save_path = os.path.join(checkpoint_dir, "lr_scheduler.pt")
                                torch.save(lr_scheduler.state_dict(), lr_scheduler_save_path)

                            step_save_path = os.path.join(checkpoint_dir, "global_step.txt")
                            with open(step_save_path, "w") as f:
                                f.write(str(global_step))
                            print(f"체크포인트 저장 완료: {checkpoint_dir}")

                    except Exception as e:
                        print(f"ERROR: [Global Step {global_step}, Epoch {epoch+1}] 스텝 처리 중 오류 발생: {e}") 
                        traceback.print_exc()
                        # 심각한 오류 시 학습 중단 결정 가능
                        # raise 
                
                if not is_training: # 내부 루프에서 학습 중단 플래그 설정 시 외부 루프도 종료
                    break

                print(f"\n에폭 {epoch+1}/{num_epochs} 종료. 최종 손실: {loss.detach().item() if 'loss' in locals() and loss is not None else 'N/A'}") 

            if progress_bar:
                progress_bar.close()

            if is_training : # 학습이 정상적으로 완료된 경우
                print("\n모델 학습 완료!")
                final_model_dir = os.path.join(current_run_output_dir, "final_model") # <--- 수정된 경로 사용
                os.makedirs(final_model_dir, exist_ok=True)
                print(f"\n학습 완료 후 최종 모델 저장 중... 경로: {final_model_dir}")
                unet_save_path = os.path.join(final_model_dir, "unet")
                unet.save_pretrained(unet_save_path)
                # global_step 저장
                step_save_path = os.path.join(final_model_dir, "global_step.txt")
                with open(step_save_path, "w") as f:
                    f.write(str(global_step))
                print("최종 모델 저장 완료.")


    elif is_inference_only: # 추론 모드 (mode == '3')
        print("\n모델 추론 시작!")
        if vae is not None and unet is not None and scheduler is not None:
            try:
                # 추론 전 입력 이미지 경로 확인
                if not os.path.exists(input_image_for_generation):
                    print(f"오류: 추론할 입력 이미지 '{input_image_for_generation}'를 찾을 수 없습니다.")
                else:
                    generate_heightmap_image(
                        input_image_path=input_image_for_generation, 
                        vae=vae,
                        unet=unet,
                        scheduler=scheduler,
                        device=device, # generate_heightmap_image 함수 내부에서 torch.device(device)로 처리
                        target_image_size=target_image_size,
                        output_path=actual_inference_output_path, # 수정된 경로 사용
                        num_inference_steps=num_inference_steps
                    )
            except Exception as e:
                print(f"\n추론 중 오류 발생: {e}")
                traceback.print_exc()
        else:
            print("\n오류: 추론에 필요한 모델(vae, unet, scheduler)이 로드되지 않았습니다.")
            
    else:
        print("\n선택된 모드에 따라 실행되지 않았습니다. 설정을 확인하세요.")
        if mode == '1' or mode == '2': # 학습 모드였으나 조건 불충족
             if dataloader is None :
                 print("- 학습 모드가 선택되었으나 DataLoader가 초기화되지 못했습니다 (데이터셋 문제 또는 경로 오류).")
             elif optimizer is None or lr_scheduler is None:
                 print("- 학습 모드가 선택되었으나 Optimizer 또는 LR Scheduler가 초기화되지 못했습니다.")
             elif device == 'cuda' and scaler is None: # scaler 조건은 CUDA일 때만 확인
                 print("- CUDA 환경에서 학습 모드가 선택되었으나 GradScaler가 초기화되지 못했습니다.")


    print("\n스크립트 실행 종료.")
