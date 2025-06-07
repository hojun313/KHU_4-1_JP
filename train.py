import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from transformers import CLIPTextModel, CLIPTokenizer
import tqdm
import numpy as np
import traceback
import sys


from dataset import GelSightDataset

data_root_directory = "heightmap_dataset"

target_image_size = (256, 256)

batch_size = 16
num_workers = 0
num_epochs = 100

use_data_augmentation = False

learning_rate = 1e-5
adamw_betas = (0.9, 0.999)
adamw_weight_decay = 1e-2
num_warmup_steps_ratio = 0.1

output_dir = "완료 모델 창고"
save_interval_steps = 250

num_inference_steps = 100
input_image_for_generation = "./IOFiles/Input/input_Carpet3.png"
output_heightmap_image_path = "./IOFiles/Output/output_heightmap.png"


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용할 디바이스: {device}")

scaler = None
if device == "cuda":
    scaler = GradScaler()
    print("AMP GradScaler 초기화 완료.")
else:
    print("CUDA 디바이스를 찾을 수 없어 AMP GradScaler를 사용하지 않습니다.")


dataset = None
dataloader = None


model_id = "runwayml/stable-diffusion-v1-5"
print(f"\n{model_id} 모델 컴포넌트 로드 중...")

vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
vae = vae.to(device)
vae.eval()

unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
unet = unet.to(device)

scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")

tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
text_encoder = text_encoder.to(device)
text_encoder.eval()

print("모델 컴포넌트 로드 완료.")

original_in_channels = unet.conv_in.in_channels
new_in_channels = 4 + 4

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
        torch.nn.init.zeros_(new_conv_in.weight)

        new_conv_in.weight.data[:, :original_in_channels, :, :].copy_(original_weight.data)
        
        if original_bias is not None:
            new_conv_in.bias.data.copy_(original_bias.data)

    unet.conv_in = new_conv_in.to(device)
    print("UNet 입력 채널 수정 완료.")
else:
    print(f"UNet 입력 채널이 이미 {new_in_channels}개입니다 (수정 불필요).")


optimizer = None
lr_scheduler = None
total_train_steps = 0
num_warmup_steps = 0


global_step = 0
start_epoch = 0
progress_bar = None


def postprocess_heightmap_image(image_tensor):
    image_tensor = (image_tensor / 2) + 0.5
    image_tensor = image_tensor.permute(0, 2, 3, 1).clamp(0, 1)

    image_np = image_tensor.detach().cpu().numpy()

    image_np = (image_np * 255).astype(np.uint8)

    if image_np.shape[0] > 0:
        image_pil = Image.fromarray(image_np[0]).convert("L")
    else:
        print("경고: postprocess_heightmap_image에 빈 배치 텐서가 전달되었습니다.")
        return None
    return image_pil


def generate_heightmap_image(input_image_path, vae, unet, scheduler, device, target_image_size, output_path="generated_heightmap.png", num_inference_steps=50):

    print(f"\n흑백 하이트맵 이미지 생성 시작 (입력: {input_image_path})")
    print(f"생성된 이미지는 다음 경로에 저장될 예정입니다: {output_path}")


    inference_transform = T.Compose([
        T.Resize(target_image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    unet.eval()
    vae.eval()
    text_encoder.eval()

    try:
        input_image_pil = Image.open(input_image_path).convert("RGB")
        input_image_tensor = inference_transform(input_image_pil).unsqueeze(0).to(device)
    except Exception as e:
        print(f"오류: 입력 이미지 로드 또는 전처리 중 문제 발생 - {input_image_path}")
        print(f"오류 내용: {e}")
        return

    with torch.no_grad():
        input_image_latent = vae.encode(input_image_tensor).latent_dist.sample() * vae.config.scaling_factor


    latent_shape = input_image_latent.shape
    torch_device_obj = torch.device(device)
    initial_noise = torch.randn(latent_shape, device=torch_device_obj)


    scheduler.set_timesteps(num_inference_steps, device=torch_device_obj)
    current_latent = initial_noise

    print(f"Diffusion 샘플링 시작 ({num_inference_steps} 스텝)...")
    context_manager = autocast(device_type="cuda") if device == "cuda" else torch.no_grad()


    with context_manager:
        for t in tqdm.tqdm(scheduler.timesteps, desc="Diffusion Sampling"):
            unet_input = torch.cat([current_latent, input_image_latent], dim=1)

            timesteps_tensor = t.to(torch_device_obj).unsqueeze(0)
            with torch.no_grad():
                current_batch_size = unet_input.shape[0]
                cross_attention_dim = unet.config.cross_attention_dim
                sequence_length = 77

                dummy_encoder_hidden_states = torch.zeros(
                    current_batch_size, sequence_length, cross_attention_dim, device=torch_device_obj, dtype=unet_input.dtype
                )

                model_pred = unet(
                    unet_input,
                    timesteps_tensor,
                    encoder_hidden_states=dummy_encoder_hidden_states
                ).sample

                scheduler_output = scheduler.step(model_pred, t, current_latent)
                current_latent = scheduler_output.prev_sample


    with torch.no_grad():
        final_heightmap_latent = current_latent / vae.config.scaling_factor
        generated_image_tensor_3channel = vae.decode(final_heightmap_latent).sample


    generated_heightmap_pil = postprocess_heightmap_image(generated_image_tensor_3channel)

    if generated_heightmap_pil:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        generated_heightmap_pil.save(output_path)
        print(f"흑백 하이트맵 이미지 생성 완료 및 저장: {output_path}")
    else:
        print("경고: 이미지 후처리 실패 또는 빈 이미지 데이터로 인해 저장을 건너뜁니다.")

    return generated_heightmap_pil


def select_checkpoint_interactive(base_dir):
    if not os.path.isdir(base_dir):
        print(f"오류: 체크포인트 기본 디렉토리 '{base_dir}'를 찾을 수 없습니다.")
        return None

    try:
        sub_items = os.listdir(base_dir)
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
            return float('inf')
        if dir_name.startswith("checkpoint-"):
            try:
                return int(dir_name.split("-")[-1])
            except ValueError:
                return -1
        return -2

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
                if os.path.isdir(selected_path):
                    return selected_path
                else: 
                    print(f"오류: 선택된 '{selected_dir_name}'은(는) 유효한 디렉토리가 아닙니다. 목록이 변경되었을 수 있습니다. 다시 시도하세요.")
                    return None
            else:
                print(f"잘못된 번호입니다. 1부터 {len(sorted_checkpoint_dirs)} 사이의 숫자를 입력하거나 'c'를 입력하세요.")
        except ValueError:
            print("숫자를 입력하거나 'c'를 입력해야 합니다.")
        except Exception as e:
            print(f"선택 중 오류 발생: {e}")
            return None
    return None


if __name__ == "__main__":
    print("스크립트 실행 시작")

    actual_inference_output_path = output_heightmap_image_path


    print("\n실행 모드를 선택하세요:")
    print("1: 모델 학습 시작")
    print("2: 체크포인트 로드 후 학습 재개")
    print("3: 체크포인트 로드 후 추론만 수행")
    mode = input("모드 입력 (1, 2, 3): ")

    is_training = False
    is_inference_only = False
    resume_from_checkpoint = None

    if mode == '1':
        is_training = True
        is_inference_only = False
        resume_from_checkpoint = None
        print(">> 새로운 학습 모드 시작.")

    elif mode == '2':
        is_training = True
        is_inference_only = False
        print(f"\n'{output_dir}' 디렉토리에서 학습 재개할 체크포인트를 선택합니다.")
        resume_from_checkpoint = select_checkpoint_interactive(output_dir)
        if resume_from_checkpoint is None:
            print("체크포인트 선택이 이루어지지 않았거나 취소되었습니다. 프로그램을 종료합니다.")
            sys.exit()
        print(f">> 학습 재개 모드 시작. 체크포인트 경로: {resume_from_checkpoint}")

    elif mode == '3':
        is_training = False
        is_inference_only = True
        print(f"\n'{output_dir}' 디렉토리에서 추론에 사용할 체크포인트를 선택합니다.")
        resume_from_checkpoint = select_checkpoint_interactive(output_dir)
        if resume_from_checkpoint is None:
            print("체크포인트 선택이 이루어지지 않았거나 취소되었습니다. 프로그램을 종료합니다.")
            sys.exit()
        print(f">> 추론 모드 시작. 체크포인트 경로: {resume_from_checkpoint}")

        if resume_from_checkpoint:
            checkpoint_name = os.path.basename(resume_from_checkpoint)
            step_identifier = ""

            if checkpoint_name == "final_model":
                step_identifier = "final"
            elif checkpoint_name.startswith("checkpoint-"):
                try:
                    step_identifier = checkpoint_name.split("-")[-1]
                    int(step_identifier)
                except (IndexError, ValueError):
                    print(f"경고: 체크포인트 이름 '{checkpoint_name}'에서 유효한 스텝 번호를 추출할 수 없습니다. 기본 출력 파일명을 사용합니다.")
                    step_identifier = ""

            if step_identifier:
                output_folder_for_inference = os.path.dirname(output_heightmap_image_path)
                original_extension = os.path.splitext(output_heightmap_image_path)[1]
                
                new_filename = f"output_heightmap_{step_identifier}{original_extension}"
                actual_inference_output_path = os.path.join(output_folder_for_inference, new_filename)
        
    else:
        print("잘못된 모드 입력입니다. 프로그램을 종료합니다.")
        sys.exit()


    if resume_from_checkpoint is not None:
        print(f"\n체크포인트 로드 중... 경로: {resume_from_checkpoint}")

        unet_load_path = os.path.join(resume_from_checkpoint, "unet")
        if os.path.exists(unet_load_path) and os.path.isdir(unet_load_path):
            try:
                unet = UNet2DConditionModel.from_pretrained(
                    unet_load_path,
                    in_channels=new_in_channels,
                    ignore_mismatched_sizes=True,
                    low_cpu_mem_usage=False
                ).to(device)
                print("- UNet 로드 완료")
            except Exception as e:
                print(f"- 오류: UNet 체크포인트 로드 중 문제 발생 ('{unet_load_path}'): {e}")
                print("  기존 UNet 모델을 계속 사용합니다 (만약 초기화되었다면). 문제가 지속되면 체크포인트를 확인하세요.")
        else:
            print(f"- 오류: UNet 체크포인트 디렉토리 '{unet_load_path}'를 찾을 수 없거나 유효하지 않습니다. 실행 불가.")
            sys.exit()
            
        if is_training:
            print("학습 재개를 위해 데이터셋 및 DataLoader를 먼저 초기화합니다 (필요시).")
            if dataset is None or dataloader is None:
                try:
                    dataset = GelSightDataset(data_root_directory, target_size=target_image_size, augment=use_data_augmentation)
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

                step_load_path = os.path.join(resume_from_checkpoint, "global_step.txt")
                if os.path.exists(step_load_path):
                    try:
                        with open(step_load_path, "r") as f:
                            loaded_step = int(f.read())
                        global_step = loaded_step
                        print(f"- Global step 로드 완료. 학습 스텝을 {global_step} 부터 시작합니다.")
                        
                        if dataloader is not None and len(dataloader) > 0:
                            start_epoch = global_step // len(dataloader)
                            print(f"- 대략적인 시작 에폭: {start_epoch}")
                        else:
                            start_epoch = 0
                            print("- 경고: DataLoader가 유효하지 않아 시작 에폭을 0으로 설정합니다.")
                    except Exception as e:
                        print(f"- 경고: Global step 로드 중 오류 발생 ('{step_load_path}'): {e}. 스텝이 0부터 시작됩니다.")
                        global_step = 0
                        start_epoch = 0
                else:
                    print(f"- 경고: Global step 체크포인트 '{step_load_path}'를 찾을 수 없습니다. 학습 스텝이 0부터 시작됩니다.")
                    global_step = 0
                    start_epoch = 0
            else:
                print("- 학습 재개 모드이나 DataLoader가 유효하지 않아 학습 관련 상태(Optimizer, Scheduler, step) 로드를 건너뜁니다.")
                is_training = False


        elif is_inference_only:
            print("- 추론 모드이므로 Optimizer 및 LR Scheduler 상태는 로드하지 않습니다.")
            global_step = 0 
            start_epoch = 0

        print("체크포인트 관련 로드 절차 완료.")
    else:
        print("체크포인트를 로드하지 않습니다 (새로운 학습 시작).")
        global_step = 0 
        start_epoch = 0

        if is_training:
            print("새로운 학습을 위해 데이터셋, DataLoader, Optimizer, Scheduler를 초기화합니다.")
            if dataset is None or dataloader is None:
                try:
                    dataset = GelSightDataset(data_root_directory, target_size=target_image_size, augment=use_data_augmentation)
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
            else:
                print("- 새로운 학습 모드이나 DataLoader가 유효하지 않아 학습을 시작할 수 없습니다.")
                is_training = False


    if is_training and dataloader is not None and optimizer is not None and lr_scheduler is not None and \
       (device == 'cpu' or (device == 'cuda' and scaler is not None)):
        print("\n모델 학습 시작!")

        if total_train_steps == 0 and dataloader is not None :
             num_update_steps_per_epoch = len(dataloader)
             total_train_steps = num_epochs * num_update_steps_per_epoch
             print(f"총 학습 스텝 재계산 (필요시): {total_train_steps}")


        if total_train_steps == 0 :
            print("오류: 총 학습 스텝이 0입니다. DataLoader 초기화에 문제가 있었을 수 있습니다.")
            is_training = False 
        else:
            if progress_bar is None:
                progress_bar = tqdm.tqdm(total=total_train_steps, initial=global_step, desc="학습 진행")
            else:
                progress_bar.total = total_train_steps
                progress_bar.initial = global_step
                progress_bar.n = global_step
                progress_bar.refresh()


        if is_training: 
            torch_device_obj_train = torch.device(device)
            for epoch in range(start_epoch, num_epochs):
                unet.train() 
                vae.eval() 
                text_encoder.eval() 

                for step, batch_data in enumerate(dataloader):
                    if batch_data is None or batch_data[0] is None:
                        print(f"경고: [Epoch {epoch+1}, Step {step+1}] 유효하지 않은 배치 데이터 수신. 건너뜁니다.")
                        if progress_bar: progress_bar.update(0)
                        continue

                    if global_step >= total_train_steps and mode != '1':
                        print(f"로드된 global_step ({global_step})이 total_train_steps ({total_train_steps}) 이상입니다. 학습을 종료합니다.")
                        is_training = False
                        break

                    try: 
                        input_images, gelsight_images = batch_data
                        input_images = input_images.to(torch_device_obj_train)
                        gelsight_images = gelsight_images.to(torch_device_obj_train)

                        with torch.no_grad(): 
                            input_images_latent = vae.encode(input_images).latent_dist.sample() * vae.config.scaling_factor
                            gelsight_images_latent = vae.encode(gelsight_images).latent_dist.sample() * vae.config.scaling_factor
                        
                        noise = torch.randn_like(gelsight_images_latent) 
                        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (gelsight_images_latent.shape[0],), device=torch_device_obj_train).long()
                        noisy_gelsight_latent = scheduler.add_noise(gelsight_images_latent, noise, timesteps) 
                        
                        unet_input = torch.cat([noisy_gelsight_latent, input_images_latent], dim=1) 
                        
                        amp_context = autocast(device_type="cuda") if device == "cuda" else torch.no_grad()


                        with amp_context:
                            current_batch_size = unet_input.shape[0]
                            cross_attention_dim = unet.config.cross_attention_dim
                            sequence_length = 77 

                            dummy_encoder_hidden_states = torch.zeros(
                                current_batch_size, sequence_length, cross_attention_dim, device=torch_device_obj_train, dtype=unet_input.dtype
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
                        else:
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
                            checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
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
                
                if not is_training:
                    break

                print(f"\n에폭 {epoch+1}/{num_epochs} 종료. 최종 손실: {loss.detach().item() if 'loss' in locals() and loss is not None else 'N/A'}") 

            if progress_bar:
                progress_bar.close()

            if is_training :
                print("\n모델 학습 완료!")
                final_model_dir = os.path.join(output_dir, "final_model")
                os.makedirs(final_model_dir, exist_ok=True)
                print(f"\n학습 완료 후 최종 모델 저장 중... 경로: {final_model_dir}")
                unet_save_path = os.path.join(final_model_dir, "unet")
                unet.save_pretrained(unet_save_path)
                step_save_path = os.path.join(final_model_dir, "global_step.txt")
                with open(step_save_path, "w") as f:
                    f.write(str(global_step))
                print("최종 모델 저장 완료.")


    elif is_inference_only:
        print("\n모델 추론 시작!")
        if vae is not None and unet is not None and scheduler is not None:
            try:
                if not os.path.exists(input_image_for_generation):
                    print(f"오류: 추론할 입력 이미지 '{input_image_for_generation}'를 찾을 수 없습니다.")
                else:
                    generate_heightmap_image(
                        input_image_path=input_image_for_generation, 
                        vae=vae,
                        unet=unet,
                        scheduler=scheduler,
                        device=device,
                        target_image_size=target_image_size,
                        output_path=actual_inference_output_path,
                        num_inference_steps=num_inference_steps
                    )
            except Exception as e:
                print(f"\n추론 중 오류 발생: {e}")
                traceback.print_exc()
        else:
            print("\n오류: 추론에 필요한 모델(vae, unet, scheduler)이 로드되지 않았습니다.")
            
    else:
        print("\n선택된 모드에 따라 실행되지 않았습니다. 설정을 확인하세요.")
        if mode == '1' or mode == '2':
             if dataloader is None :
                 print("- 학습 모드가 선택되었으나 DataLoader가 초기화되지 못했습니다 (데이터셋 문제 또는 경로 오류).")
             elif optimizer is None or lr_scheduler is None:
                 print("- 학습 모드가 선택되었으나 Optimizer 또는 LR Scheduler가 초기화되지 못했습니다.")
             elif device == 'cuda' and scaler is None:
                 print("- CUDA 환경에서 학습 모드가 선택되었으나 GradScaler가 초기화되지 못했습니다.")


    print("\n스크립트 실행 종료.")
