import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
import torchvision
import argparse
from torchvision import transforms
import lpips # LPIPS 손실을 계속 사용한다고 가정

# 직접 만든 모듈 임포트
from re_dataset import TextureHeightmapDataset 
from re_model import create_model # 생성자(U-Net)를 위해

# ----------------- 판별자(PatchGAN) 네트워크 정의 -----------------
# (사용자님이 이미 추가하신 내용)
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, target_channels=1, ndf=64):
        super(Discriminator, self).__init__()
        # 입력은 (원본 이미지 + 타겟/생성 이미지)를 채널 방향으로 합친 것
        input_c = in_channels + target_channels # 예: 재질(3) + 하이트맵(1) = 4채널

        self.model = nn.Sequential(
            # 입력: (B, input_c, 256, 256)
            nn.Conv2d(input_c, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True), # (B, ndf, 128, 128)

            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 2), # Pix2Pix는 종종 BatchNorm 대신 InstanceNorm 사용
            nn.LeakyReLU(0.2, inplace=True), # (B, ndf*2, 64, 64)

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True), # (B, ndf*4, 32, 32)

            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True), # (B, ndf*8, 31, 31)

            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1) # (B, 1, 30, 30)
            # Sigmoid는 BCEWithLogitsLoss 사용 시 생략
        )

    def forward(self, x, y): # x: input_image, y: target_image (real or fake)
        xy = torch.cat([x, y], dim=1)
        return self.model(xy)

# ----------------- 유틸리티 함수 (GAN에 맞게 수정) -----------------
def save_checkpoint(gen, disc, opt_gen, opt_disc, scheduler_gen, epoch, filename="pix2pix_checkpoint.pth.tar"):
    print("=> 체크포인트 저장")
    checkpoint = {
        "epoch": epoch,
        "gen_state_dict": gen.state_dict(),
        "disc_state_dict": disc.state_dict(),
        "opt_gen_state_dict": opt_gen.state_dict(),
        "opt_disc_state_dict": opt_disc.state_dict(),
        "scheduler_gen_state_dict": scheduler_gen.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, gen, disc, opt_gen, opt_disc, scheduler_gen, lr_g, lr_d, device):
    print(f"=> 체크포인트 불러오기: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    
    gen.load_state_dict(checkpoint["gen_state_dict"])
    disc.load_state_dict(checkpoint["disc_state_dict"])
    opt_gen.load_state_dict(checkpoint["opt_gen_state_dict"])
    opt_disc.load_state_dict(checkpoint["opt_disc_state_dict"])
    
    if "scheduler_gen_state_dict" in checkpoint and checkpoint["scheduler_gen_state_dict"] is not None:
        try:
            scheduler_gen.load_state_dict(checkpoint["scheduler_gen_state_dict"])
            print("생성자 스케줄러 상태를 성공적으로 불러왔습니다.")
        except Exception as e:
            print(f"경고: 생성자 스케줄러 상태 로드 중 오류 ({e}). 초기 상태로 시작합니다.")
    else:
        print("경고: 체크포인트에 생성자 스케줄러 상태 없음. 초기 상태로 시작합니다.")
        
    start_epoch = checkpoint.get("epoch", -1) + 1
    print(f"=> 체크포인트 로드 완료. 에포크 {start_epoch}부터 학습을 재개합니다.")
    
    for param_group in opt_gen.param_groups: param_group["lr"] = lr_g
    for param_group in opt_disc.param_groups: param_group["lr"] = lr_d
        
    return start_epoch

def save_predictions_as_imgs(loader, gen_model, epoch, folder, device, current_batch_size):
    # 이 함수는 생성자(gen_model)만 사용하므로 이전과 거의 동일
    gen_model.eval()
    num_batches_to_save = 1 # GAN 학습 시에는 이미지 저장을 더 적게 할 수도 있음
    saved_count = 0
    for idx, (x, y) in enumerate(loader): # x: input_texture, y: real_heightmap
        if saved_count >= num_batches_to_save: break
        x, y = x.to(device=device), y.to(device=device)
        with torch.no_grad():
            fake_y = gen_model(x) # 생성자로 예측
            # 결과 이미지 정규화 (0~1 범위로)
            fake_y_unnormalized = (fake_y * 0.5) + 0.5 
        y_unnormalized = (y * 0.5) + 0.5
        
        # 원본 입력, 실제 하이트맵, 생성된 하이트맵을 나란히 저장 (선택 사항)
        # 여기서는 실제와 예측만 비교
        grid_tensor = torch.cat((y_unnormalized[:, :1, :, :], fake_y_unnormalized), dim=0)
        
        torchvision.utils.save_image(
            grid_tensor, 
            f"{folder}/comparison_epoch_{epoch+1}_batch_{idx}.png", 
            nrow=x.size(0) 
        )
        saved_count += 1
    gen_model.train()

# ----------------- 메인 학습 함수 (Pix2Pix용으로 대폭 수정) -----------------
def train_fn(loader, gen, disc, opt_gen, opt_disc, l1_loss_fn, bce_loss_fn, lambda_l1, scaler_gen, scaler_disc, device):
    loop = tqdm(loader, desc="Pix2Pix Training", leave=True)
    
    avg_loss_D_epoch = 0.0
    avg_loss_G_epoch = 0.0
    avg_loss_G_l1_epoch = 0.0
    avg_loss_G_adv_epoch = 0.0

    for batch_idx, (input_img, target_img) in enumerate(loop):
        input_img = input_img.to(device)    # 재질 이미지 (real_A)
        target_img = target_img.to(device)  # 실제 하이트맵 (real_B)

        # --- 판별자(D) 학습 ---
        with torch.amp.autocast(device_type=device.split(':')[0] if device != 'cpu' else 'cpu', enabled=(device != 'cpu')):
            fake_img = gen(input_img) # 생성자가 가짜 하이트맵 생성

            # 실제 이미지에 대한 D의 예측
            D_real_pred = disc(input_img, target_img)
            loss_D_real = bce_loss_fn(D_real_pred, torch.ones_like(D_real_pred)) # 진짜는 1로

            # 가짜 이미지에 대한 D의 예측 (생성자 그래디언트 흐름 차단)
            D_fake_pred = disc(input_img, fake_img.detach())
            loss_D_fake = bce_loss_fn(D_fake_pred, torch.zeros_like(D_fake_pred)) # 가짜는 0으로
            
            loss_D = (loss_D_real + loss_D_fake) * 0.5 # 판별자 전체 손실

        opt_disc.zero_grad()
        if device != 'cpu':
            scaler_disc.scale(loss_D).backward()
            scaler_disc.step(opt_disc)
            scaler_disc.update()
        else:
            loss_D.backward()
            opt_disc.step()
        
        # --- 생성자(G) 학습 ---
        with torch.amp.autocast(device_type=device.split(':')[0] if device != 'cpu' else 'cpu', enabled=(device != 'cpu')):
            # fake_img는 위에서 생성된 것을 다시 사용 (detach 안 된 버전)
            # 또는, 명확성을 위해 다시 생성: fake_img_for_G = gen(input_img)
            D_fake_pred_for_G = disc(input_img, fake_img) # 여기서는 fake_img의 그래디언트가 흘러야 함
            
            # 생성자는 판별자가 가짜 이미지를 진짜라고 믿게 만들어야 함 (label을 1로)
            loss_G_adversarial = bce_loss_fn(D_fake_pred_for_G, torch.ones_like(D_fake_pred_for_G))
            
            # 재구성 손실 (L1)
            loss_G_l1 = l1_loss_fn(fake_img, target_img) * lambda_l1
            
            loss_G = loss_G_adversarial + loss_G_l1 # 생성자 전체 손실
            
        opt_gen.zero_grad()
        if device != 'cpu':
            scaler_gen.scale(loss_G).backward()
            scaler_gen.step(opt_gen)
            scaler_gen.update()
        else:
            loss_G.backward()
            opt_gen.step()

        # 손실 값 기록
        avg_loss_D_epoch += loss_D.item()
        avg_loss_G_epoch += loss_G.item()
        avg_loss_G_l1_epoch += loss_G_l1.item()
        avg_loss_G_adv_epoch += loss_G_adversarial.item()
        
        loop.set_postfix(D_loss=loss_D.item(), G_loss=loss_G.item(), G_L1=loss_G_l1.item(), G_adv=loss_G_adversarial.item())

    num_batches = len(loader)
    print(f"에포크 평균 손실 - D: {avg_loss_D_epoch/num_batches:.4f}, G_Total: {avg_loss_G_epoch/num_batches:.4f}, G_L1: {avg_loss_G_l1_epoch/num_batches:.4f}, G_Adv: {avg_loss_G_adv_epoch/num_batches:.4f}")
    return avg_loss_G_epoch/num_batches # 생성자 손실을 스케줄러용으로 반환 (또는 L1 부분만)

# ----------------- 메인 실행 부분 -----------------
def main(args):
    DEVICE = "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    print(f"사용 장치: {DEVICE}")

    # 모델 초기화
    gen_model = create_model(device=DEVICE, encoder_name=args.encoder)
    disc_model = Discriminator(in_channels=3, target_channels=1, ndf=64).to(DEVICE) # 하이트맵은 1채널

    # 옵티마이저 (G와 D 각각)
    opt_gen = optim.Adam(gen_model.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc_model.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    # 손실 함수
    bce_loss = nn.BCEWithLogitsLoss().to(DEVICE) # Adversarial loss
    l1_loss = nn.L1Loss().to(DEVICE)             # Reconstruction loss for Generator

    # 스케줄러 (생성자 옵티마이저용)
    # GAN 학습 시 스케줄러는 좀 더 신중하게 사용하거나, 다른 종류를 사용하기도 함.
    # 여기서는 일단 ReduceLROnPlateau를 생성자 손실에 대해 적용.
    scheduler_gen = ReduceLROnPlateau(opt_gen, mode='min', factor=args.lr_factor, patience=args.lr_patience, verbose=True)

    start_epoch = 0
    if args.load_checkpoint: # 인자 이름 변경 load_model -> load_checkpoint
        if os.path.exists(args.load_checkpoint):
            start_epoch = load_checkpoint(
                args.load_checkpoint, gen_model, disc_model, 
                opt_gen, opt_disc, scheduler_gen, 
                args.lr_g, args.lr_d, DEVICE
            )
        else:
            print(f"경고: 지정된 체크포인트 파일 '{args.load_checkpoint}'를 찾을 수 없습니다.")

    # 데이터셋 및 데이터로더 (이전과 거의 동일)
    transform_texture = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # -1 ~ 1 범위로 정규화
    ])
    transform_heightmap = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) # -1 ~ 1 범위로 정규화 (흑백)
    ])

    # 데이터셋 및 데이터로더 생성
    dataset = TextureHeightmapDataset(
        data_root=args.data_root,
        transform_texture=transform_texture,
        transform_heightmap=transform_heightmap,
        exclude_heightmap_indices_up_to=args.exclude_indices # 인자로부터 받도록 수정
    )
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    # GradScaler (GPU 사용 시)
    scaler_gen_enabled = (DEVICE != 'cpu')
    scaler_disc_enabled = (DEVICE != 'cpu')
    scaler_gen = torch.amp.GradScaler(DEVICE.split(':')[0] if DEVICE != 'cpu' else 'cpu', enabled=scaler_gen_enabled)
    scaler_disc = torch.amp.GradScaler(DEVICE.split(':')[0] if DEVICE != 'cpu' else 'cpu', enabled=scaler_disc_enabled)

    # 결과 저장 디렉토리 (모델 태그 포함)
    CHECKPOINT_DIR = os.path.join(args.output_dir, args.model_tag, "checkpoints")
    SAVE_PREDICTIONS_DIR = os.path.join(args.output_dir, args.model_tag, "saved_images")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SAVE_PREDICTIONS_DIR, exist_ok=True)

    # 학습 루프
    for epoch in range(start_epoch, args.num_epochs):
        current_lr_g = opt_gen.param_groups[0]['lr']
        current_lr_d = opt_disc.param_groups[0]['lr']
        print(f"\n--- 에포크 {epoch+1}/{args.num_epochs} --- LR_G: {current_lr_g:.2e}, LR_D: {current_lr_d:.2e} ---")
        
        avg_g_loss = train_fn(loader, gen_model, disc_model, opt_gen, opt_disc, l1_loss, bce_loss, args.lambda_l1, scaler_gen, scaler_disc, DEVICE)
        
        scheduler_gen.step(avg_g_loss) # 생성자 손실을 기준으로 스케줄러 업데이트
        
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.num_epochs:
            print(f"\n--- 에포크 {epoch+1}, 저장 분기점 도달 ---")
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth.tar")
            save_checkpoint(
                gen_model, disc_model,      # 생성자, 판별자 모델
                opt_gen, opt_disc,          # 각 옵티마이저
                scheduler_gen,              # 생성자용 스케줄러
                epoch, 
                filename=checkpoint_path
            )
            
            # save_predictions_as_imgs는 생성자(gen_model)만 필요
            save_predictions_as_imgs(
                loader,                     # 데이터 로더
                gen_model,                  # 생성자 모델
                epoch, 
                folder=SAVE_PREDICTIONS_DIR, 
                device=DEVICE, 
                current_batch_size=args.batch_size
            )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pix2Pix Heightmap Generation Training")
    # 이전 re_train.py의 인자들을 기반으로 수정/추가
    parser.add_argument("--data_root", type=str, required=True, help="데이터셋 루트 경로")
    parser.add_argument("--output_dir", type=str, default="./pix2pix_results", help="결과 저장 상위 디렉토리")
    parser.add_argument("--model_tag", type=str, default="pix2pix_run", help="실험 태그")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="재개할 체크포인트 파일 경로") # load_model -> load_checkpoint
    
    parser.add_argument("--encoder", type=str, default="resnet34", help="생성자(U-Net)의 인코더")
    parser.add_argument("--lr_g", type=float, default=2e-4, help="생성자 학습률 (Pix2Pix 기본값)")
    parser.add_argument("--lr_d", type=float, default=2e-4, help="판별자 학습률 (Pix2Pix 기본값)")
    parser.add_argument("--batch_size", type=int, default=1, help="배치 크기 (Pix2Pix는 종종 1 사용)") # GAN은 배치 크기에 민감할 수 있음
    parser.add_argument("--num_epochs", type=int, default=200, help="총 학습 에포크")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader 워커 수")
    parser.add_argument("--save_interval", type=int, default=25, help="체크포인트 저장 주기 (에포크)")
    parser.add_argument("--image_size", type=int, default=256, help="이미지 리사이즈 크기")
    parser.add_argument("--exclude_indices", type=int, default=15, help="제외할 하이트맵 인덱스 상한값")
    
    # 스케줄러 관련 인자 (생성자용)
    parser.add_argument("--lr_patience", type=int, default=10, help="ReduceLROnPlateau의 patience (생성자용)")
    parser.add_argument("--lr_factor", type=float, default=0.5, help="ReduceLROnPlateau의 factor (생성자용, 0.2 -> 0.5로 변경 제안)")
    
    # GAN 관련 손실 가중치
    parser.add_argument("--lambda_l1", type=float, default=100.0, help="생성자 L1 재구성 손실 가중치 (Pix2Pix 논문 기본값)")
    
    parser.add_argument("--use_gpu", action='store_true', default=False, help="GPU 사용 여부")

    args, unknown = parser.parse_known_args()
    
    # transform 정의 및 loader 생성 등 main 함수 안으로 이동시키거나, 여기서 정의 후 main에 전달
    # 여기서는 main 함수 안에서 처리하도록 함.
    
    # TextureHeightmapDataset, transforms 등 필요한 클래스/모듈이 이 파일 범위에 있거나 임포트되어야 함.
    # 위에 이미 임포트 되어 있음.
    
    # 예시 실행을 위해 transforms 및 dataset, loader 정의를 main 함수 밖으로 빼거나,
    # main 함수를 직접 호출하기 전에 args를 설정하는 방법이 필요함.
    # 여기서는 main(args)를 직접 호출하는 것으로 가정하고, main 함수 내에서 transforms 등을 정의하도록 함.
    # (이전 코드에서는 main 함수 내에 transforms, dataset, loader 정의가 있었음)
    
    # Dummy transform/dataset/loader for a dry run if needed for testing outside main
    # transform_texture_dummy = transforms.Compose([transforms.ToTensor()])
    # transform_heightmap_dummy = transforms.Compose([transforms.ToTensor()])
    # dataset_dummy = TextureHeightmapDataset(data_root=".", transform_texture=transform_texture_dummy, transform_heightmap=transform_heightmap_dummy, exclude_heightmap_indices_up_to=-1)
    # loader_dummy = DataLoader(dataset_dummy, batch_size=1)

    # save_predictions_as_imgs 에서 사용될 epoch 변수를 위한 임시 조치
    # 실제로는 main 함수 내 epoch 루프의 epoch 변수가 전달되어야 함
    # global epoch 
    # epoch = 0 # save_predictions_as_imgs가 전역 epoch을 참조하고 있다면 필요

    main(args)