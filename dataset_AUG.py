import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF # 데이터 증강 및 잘라내기를 위해 사용
import random # 데이터 증강을 위해 사용
import numpy as np


class GelSightDataset(Dataset):
    def __init__(self, data_root, target_size=(128, 128), augment=False, excluded_materials=None):
        self.data_root = data_root
        self.target_size = target_size # 최종 이미지가 리사이즈될 목표 크기
        self.augment = augment # 증강 사용 여부 저장
        self.center_crop_size = (256, 256) # <--- 중앙 잘라내기 크기 (높이, 너비)
        
        self.excluded_materials = excluded_materials if excluded_materials is not None else [] # 제외할 재질 목록
        if self.excluded_materials: # 제외 목록이 있을 경우에만 출력
            print(f"알림: 다음 재질들이 학습에서 제외되도록 설정되었습니다: {', '.join(self.excluded_materials)}")
            print(f"제외된 재질 개수: {len(self.excluded_materials)}")
        else:
            print("알림: 제외되도록 설정된 재질이 없습니다. 모든 재질을 사용합니다.")

        self.image_pairs = self._build_image_pairs()

        # 입력 이미지(흑백 원본 -> 3채널 RGB 확장 후 정규화)를 위한 변환 정의
        self.transform_input_pil_to_tensor = T.Compose([
            T.Resize(target_size), # PIL 이미지를 최종 target_size로 리사이즈
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # 목표 이미지를 위한 transform (흑백 하이트맵)
        self.transform_output_pil_to_tensor = T.Compose([
            T.Resize(target_size), # PIL 이미지를 최종 target_size로 리사이즈
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])
        
        if self.augment:
            print("데이터 증강(랜덤 4방향 회전, 랜덤 상하/좌우 반전)이 활성화되었습니다.")
        # 중앙 잘라내기가 기본으로 적용됨을 명시
        print(f"이미지 중앙 잘라내기(크기: {self.center_crop_size})가 적용됩니다. 잘라낸 후 {self.target_size}로 리사이즈됩니다.")
        if self.excluded_materials:
            print(f"제외돤 재질: {', '.join(self.excluded_materials)}")

    def _get_material_name_from_path(self, folder_path):
        return os.path.basename(folder_path)

    def _build_image_pairs(self):
        image_pairs = []
        # data_root 바로 아래의 폴더들을 재질 폴더로 간주
        all_material_folder_paths = [d for d in glob.glob(os.path.join(self.data_root, '*')) if os.path.isdir(d)]
        
        # 제외할 재질 폴더 필터링
        material_folders_to_use = []
        for material_path in all_material_folder_paths:
            material_name = self._get_material_name_from_path(material_path) # 재질 이름 추출 함수 필요
            if material_name not in self.excluded_materials:
                material_folders_to_use.append(material_path)
            else:
                print(f"알림: 재질 '{material_name}'은(는) 이번 학습에서 제외됩니다.")
                actually_excluded_count +=1

        if self.excluded_materials: # 제외 목록이 있었을 경우, 실제로 몇 개가 필터링 되었는지 추가 정보 제공
             print(f"확인: 전달된 제외 목록에 따라 총 {actually_excluded_count}개의 재질 폴더가 _build_image_pairs에서 필터링(제외)되었습니다.")

        excluded_numbers_suffix = [f"_{i:05d}" for i in range(0, 16)]

        for material_folder_path in material_folders_to_use: # 필터링된 폴더만 사용
            # ... (기존 input_files, output_folder, condition_folders, heightmaps_folder 찾는 로직은 동일) ...
            # 이하는 기존 _build_image_pairs 로직과 거의 동일하게 유지, material_folder_path만 필터링된 것을 사용
            input_files = glob.glob(os.path.join(material_folder_path, 'input_*.png')) + \
                          glob.glob(os.path.join(material_folder_path, 'input_*.jpg'))

            for input_file_path in input_files:
                input_base_name = os.path.splitext(os.path.basename(input_file_path))[0].replace('input_', '', 1)
                output_folder = os.path.join(material_folder_path, "output")

                if os.path.isdir(output_folder):
                    condition_folders = [d for d in glob.glob(os.path.join(output_folder, '*')) if os.path.isdir(d) and input_base_name.lower() in os.path.basename(d).lower()]

                    for condition_folder_path in condition_folders:
                        heightmaps_folder = os.path.join(condition_folder_path, "heightmaps")
                        if os.path.isdir(heightmaps_folder):
                            found_heightmap_files = glob.glob(os.path.join(heightmaps_folder, "*.*"))
                            valid_heightmap_files_for_this_input = []

                            for heightmap_file_path in found_heightmap_files:
                                heightmap_filename_without_ext = os.path.splitext(os.path.basename(heightmap_file_path))[0]
                                should_exclude_this_heightmap = False
                                for excluded_suffix in excluded_numbers_suffix:
                                    if heightmap_filename_without_ext.endswith(excluded_suffix):
                                        should_exclude_this_heightmap = True
                                        break
                                if not should_exclude_this_heightmap:
                                    valid_heightmap_files_for_this_input.append(heightmap_file_path)
                            
                            for valid_heightmap_file_path in valid_heightmap_files_for_this_input:
                                 image_pairs.append((input_file_path, valid_heightmap_file_path))
        
        if not image_pairs:
             print(f"경고: 경로 '{self.data_root}' 및 하위 폴더 탐색 결과 (제외 항목 적용 후), 유효한 입력-하이트맵 이미지 쌍을 찾을 수 없습니다. 데이터셋이 비어 있습니다.")
        else:
            print(f"\n총 {len(image_pairs)} 개의 입력-하이트맵 쌍을 데이터셋으로 구축했습니다 (제외 항목 적용 후).")
        if not image_pairs:
             print("데이터셋이 비어있으므로 학습 또는 추론을 정상적으로 진행하기 어렵습니다. 데이터 경로와 파일 이름을 확인하세요.")
        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        input_path, output_path = self.image_pairs[idx]
        
        try:
            input_image_pil = Image.open(input_path).convert("L")
            output_image_pil = Image.open(output_path).convert("L")

            # --- 1. 이미지 리사이즈 (잘라내기 전처리, 원본 비율 유지) ---
            crop_h, crop_w = self.center_crop_size
            orig_w, orig_h = input_image_pil.size

            if orig_w < crop_w or orig_h < crop_h:
                scale_w = crop_w / orig_w if orig_w < crop_w else 1.0
                scale_h = crop_h / orig_h if orig_h < crop_h else 1.0
                scale = max(scale_w, scale_h)
                
                new_intermediate_h = int(orig_h * scale)
                new_intermediate_w = int(orig_w * scale)

                input_image_pil = TF.resize(input_image_pil, (new_intermediate_h, new_intermediate_w), antialias=True)
                output_image_pil = TF.resize(output_image_pil, (new_intermediate_h, new_intermediate_w), antialias=True)

            # --- 2. 중앙 256x256 잘라내기 ---
            input_image_pil = TF.center_crop(input_image_pil, self.center_crop_size)
            output_image_pil = TF.center_crop(output_image_pil, self.center_crop_size)
            
            # --- 3. 데이터 증강 (잘라낸 256x256 이미지에 적용) ---
            if self.augment:
                # 3a. 랜덤 좌우 반전 (50% 확률)
                if random.random() > 0.5:
                    input_image_pil = TF.hflip(input_image_pil)
                    output_image_pil = TF.hflip(output_image_pil)
                
                # 3b. 랜덤 상하 반전 (50% 확률)
                if random.random() > 0.5:
                    input_image_pil = TF.vflip(input_image_pil)
                    output_image_pil = TF.vflip(output_image_pil)

                # 3c. 랜덤 4방향 회전 (0, 90, 180, 270도)
                angle = random.choice([0, 90, 180, 270])
                if angle != 0: # 0도 회전은 불필요
                    input_image_pil = TF.rotate(input_image_pil, angle, interpolation=TF.InterpolationMode.NEAREST) # 하이트맵이므로 BILINEAR보다 NEAREST가 적합할 수 있음
                    output_image_pil = TF.rotate(output_image_pil, angle, interpolation=TF.InterpolationMode.NEAREST)

            # --- 4. 최종 변환 (채널 확장, 텐서 변환, 정규화, target_size로 리사이즈) ---
            input_image_rgb_from_gray = input_image_pil.convert("RGB")
            input_tensor = self.transform_input_pil_to_tensor(input_image_rgb_from_gray)

            output_tensor_1channel = self.transform_output_pil_to_tensor(output_image_pil)
            output_tensor_3channel = output_tensor_1channel.repeat(3, 1, 1)

        except Exception as e:
            print(f"오류: 이미지 처리 중 문제 발생 - 입력: {input_path}, 하이트맵: {output_path}")
            print(f"오류 내용: {e}")
            return None, None

        return input_tensor, output_tensor_3channel