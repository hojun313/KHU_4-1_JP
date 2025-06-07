import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np

import logging

logging.basicConfig(filename='dataset_errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


class GelSightDataset(Dataset):
    def __init__(self, data_root, target_size=(256, 256)):
        self.data_root = data_root
        self.target_size = target_size

        self.image_pairs = self._build_image_pairs()

        self.transform_input_pil_to_tensor = T.Compose([
            T.Resize(self.target_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.transform_output_pil_to_tensor = T.Compose([
            T.Resize(self.target_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])
        

    def _get_material_name_from_path(self, folder_path):
        return os.path.basename(folder_path)

    def _build_image_pairs(self):
        image_pairs = []
        all_material_folder_paths = [d for d in glob.glob(os.path.join(self.data_root, '*')) if os.path.isdir(d)]
        
        material_folders_to_use = []
        actually_excluded_count = 0 
        for material_path in all_material_folder_paths:
            material_name = self._get_material_name_from_path(material_path)
            if material_name not in self.excluded_materials:
                material_folders_to_use.append(material_path)
            else:
                actually_excluded_count +=1

        excluded_numbers_suffix = [f"_{i:05d}" for i in range(0, 16)]

        for material_folder_path in material_folders_to_use: 
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
            input_image_pil = Image.open(input_path).convert("RGB")
            output_image_pil = Image.open(output_path).convert("L")

            input_image_rgb_from_gray = input_image_pil.convert("RGB")
            input_tensor = self.transform_input_pil_to_tensor(input_image_pil)

            output_tensor_1channel = self.transform_output_pil_to_tensor(output_image_pil)
            output_tensor_3channel = output_tensor_1channel.repeat(3, 1, 1)

        except Exception as e:
            print(f"오류: 이미지 처리 중 문제 발생 - 입력: {input_path}, 하이트맵: {output_path}")
            print(f"오류 내용: {e}")
            return None, None 

        return input_tensor, output_tensor_3channel