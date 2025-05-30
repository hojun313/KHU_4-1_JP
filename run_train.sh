#!/bin/bash

#SBATCH -J HeightmapTrain
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ce_ugrad
#SBATCH -t 1-00:00:00
#SBATCH -o /data/hojun313/JP/slurm_logs/slurm-%A.out
#SBATCH -e /data/hojun313/JP/slurm_logs/slurm-%A.err

# --- 초기 환경 설정 ---
echo "---------------------------------------------------------------------"
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Project Directory: /data/hojun313/JP/"
mkdir -p /data/hojun313/JP/slurm_logs
echo "Slurm logs will be saved to /data/hojun313/JP/slurm_logs/"
echo "---------------------------------------------------------------------"

# 외부 스크립트에서 전달받은 인자 사용
# EXCLUDED_MATERIAL_ARG_STRING=$1
MODEL_TAG_ARG=$1 # 예: "excluded_group_0_shuffled"

# if [ -z "$EXCLUDED_MATERIAL_ARG_STRING" ] || [ -z "$MODEL_TAG_ARG" ]; then
#     echo "오류: 제외할 재질 이름과 모델 태그가 인자로 전달되지 않았습니다." >&2
#     exit 1
# fi

# --- 사용자 개인 Anaconda 환경 활성화 ---
CONDA_ENV_NAME="JP_3rdTry_GPU"
echo "Activating Anaconda environment: $CONDA_ENV_NAME"
source /data/hojun313/anaconda3/bin/activate "$CONDA_ENV_NAME"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment '$CONDA_ENV_NAME'." >&2
    exit 1
fi
echo "Conda environment '$CONDA_DEFAULT_ENV' activated successfully."
echo "Python executable: $(which python)"
echo "---------------------------------------------------------------------"

# --- 데이터셋 준비 ---
NAS_TARBALL_PATH="/data/hojun313/JP/heightmap_dataset_83.tar"
LOCAL_SCRATCH_PARENT_DIR="/local_datasets/hojun313"
CURRENT_JOB_SCRATCH_DIR_NAME="${MODEL_TAG_ARG}_${SLURM_JOB_ID}"
LOCAL_SCRATCH_BASE="${LOCAL_SCRATCH_PARENT_DIR}/${CURRENT_JOB_SCRATCH_DIR_NAME}"
LOCAL_DATA_FOR_PYTHON_SCRIPT="${LOCAL_SCRATCH_BASE}/heightmap_dataset"

# --- !!! 자동 공간 정리 로직 (CLEANUP_TARGET_PARENT_DIR 아래 모든 내용 삭제) 시작 !!! ---
echo "!!! WARNING: Attempting to remove all contents within $CLEANUP_TARGET_PARENT_DIR !!!"
if [ -d "$CLEANUP_TARGET_PARENT_DIR" ]; then
    echo "Found parent directory for cleanup: $CLEANUP_TARGET_PARENT_DIR"
    echo "Listing contents before cleanup:"
    ls -Alh "$CLEANUP_TARGET_PARENT_DIR" # 삭제 전 내용 확인용 로그 (선택 사항)
    
    # $CLEANUP_TARGET_PARENT_DIR 경로 바로 아래의 모든 파일 및 디렉토리를 삭제합니다.
    # 주의: $CLEANUP_TARGET_PARENT_DIR 자체를 지우는 것이 아니라 그 안의 내용만 지웁니다.
    # (find ... -delete 방식이 rm -rf * 보다 특정 상황에서 더 안전할 수 있습니다)
    find "$CLEANUP_TARGET_PARENT_DIR" -mindepth 1 -maxdepth 1 -exec echo "Removing: {}" \; -exec rm -rf {} \;
    
    # 또는, 더 간단하지만 경로에 매우 주의해야 하는 방식:
    # cd "$CLEANUP_TARGET_PARENT_DIR" && rm -rf ./*
    # cd - # 원래 디렉토리로 돌아오기
    
    # 위의 find 명령 실행 후 성공 여부 확인 (간단한 확인)
    if [ "$(ls -A $CLEANUP_TARGET_PARENT_DIR)" ]; then # 디렉토리가 비어있지 않다면
        echo "WARNING: Cleanup of $CLEANUP_TARGET_PARENT_DIR might not have been complete. Listing remaining contents:"
        ls -Alh "$CLEANUP_TARGET_PARENT_DIR"
    else
        echo "Cleanup of $CLEANUP_TARGET_PARENT_DIR appears successful (directory is empty or only contains new job folder if created early)."
    fi
else
    echo "Parent scratch directory $CLEANUP_TARGET_PARENT_DIR not found. Will attempt to create it and its subdirectory."
    # 이 경우 $CLEANUP_TARGET_PARENT_DIR 자체를 만들어야 할 수도 있습니다.
    # mkdir -p "$CLEANUP_TARGET_PARENT_DIR" (필요시)
fi
echo "--- Cleanup logic finished ---"
# --- 자동 공간 정리 로직 끝 ---

echo "Preparing dataset..."
echo "Attempting to create current job's local scratch directory: $LOCAL_SCRATCH_BASE"
mkdir -p "$LOCAL_SCRATCH_BASE"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create current job's local scratch directory '$LOCAL_SCRATCH_BASE'." >&2
    exit 1
fi
echo "Current job's local scratch directory created successfully."

echo "Copying dataset tarball from $NAS_TARBALL_PATH to $LOCAL_SCRATCH_BASE/ ..."
cp "$NAS_TARBALL_PATH" "${LOCAL_SCRATCH_BASE}/heightmap_dataset_83.tar"
COPY_EXIT_CODE=$? # cp 명령어의 종료 코드 저장

# --- !!! 디버깅 정보 추가 시작 !!! ---
echo "--- Debugging file copy ---"
echo "NAS Tarball Path: $NAS_TARBALL_PATH"
ls -lh "$NAS_TARBALL_PATH" # NAS에 원본 파일이 있는지, 크기는 어떤지 확인
echo "Local Scratch Base for Tarball: ${LOCAL_SCRATCH_BASE}/heightmap_dataset_83.tar"
ls -lh "${LOCAL_SCRATCH_BASE}/heightmap_dataset_83.tar" # 로컬 스크래치에 파일이 복사되었는지, 크기는 어떤지 확인
echo "Exit code of cp command: $COPY_EXIT_CODE"
echo "--- Debugging file copy end ---"
# --- !!! 디버깅 정보 추가 끝 !!! ---

if [ $COPY_EXIT_CODE -ne 0 ]; then # cp 명령어의 종료 코드로 실패 여부 판단
    echo "ERROR: Failed to copy dataset tarball to local scratch. CP Exit Code: $COPY_EXIT_CODE." >&2
    rm -rf "$LOCAL_SCRATCH_BASE"
    exit 1
fi
# cp가 성공했음에도 파일이 없거나 크기가 0이면 추가적인 문제 확인
if [ ! -f "${LOCAL_SCRATCH_BASE}/heightmap_dataset_83.tar" ] || [ ! -s "${LOCAL_SCRATCH_BASE}/heightmap_dataset_83.tar" ]; then
    echo "ERROR: Tarball was not copied correctly or is empty in local scratch AFTER cp reported success." >&2
    echo "Listing contents of $LOCAL_SCRATCH_BASE/ :"
    ls -lh "$LOCAL_SCRATCH_BASE/" # 현재 로컬 스크래치 디렉토리 내용 보여주기
    rm -rf "$LOCAL_SCRATCH_BASE"
    exit 1
fi
echo "Dataset tarball copied successfully and seems valid."

echo "Extracting dataset tarball in $LOCAL_SCRATCH_BASE ..."
# --- !!! 디버깅 정보 추가 시작 !!! ---
echo "--- Debugging before tar extraction ---"
echo "Target tar file for extraction: ${LOCAL_SCRATCH_BASE}/heightmap_dataset_83.tar"
ls -lh "${LOCAL_SCRATCH_BASE}/heightmap_dataset_83.tar" # tar 실행 직전 파일 상태 재확인
echo "Extraction directory: $LOCAL_SCRATCH_BASE"
df -h "$LOCAL_SCRATCH_BASE" # 현재 파일 시스템의 사용 가능한 공간 확인
echo "--- Debugging before tar extraction end ---"
# --- !!! 디버깅 정보 추가 끝 !!! ---

tar -xf "${LOCAL_SCRATCH_BASE}/heightmap_dataset_83.tar" -C "$LOCAL_SCRATCH_BASE"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to extract dataset tarball." >&2
    rm -rf "$LOCAL_SCRATCH_BASE"
    exit 1
fi
echo "Dataset extracted successfully to $LOCAL_DATA_FOR_PYTHON_SCRIPT"

echo "Removing local tarball ${LOCAL_SCRATCH_BASE}/heightmap_dataset_83.tar"
rm "${LOCAL_SCRATCH_BASE}/heightmap_dataset_83.tar"
echo "---------------------------------------------------------------------"

# --- 학습 스크립트 실행 ---
PROJECT_DIR="/data/hojun313/JP"
echo "Changing directory to $PROJECT_DIR"
cd "$PROJECT_DIR"
echo "Current working directory: $(pwd)"

echo "Starting Python training script (trainGPU.py)..."
PYTHON_ARGS=(
    --data_root "$LOCAL_DATA_FOR_PYTHON_SCRIPT" \
    --model_tag "$MODEL_TAG_ARG" \
    --batch_size 32 \
    --num_workers 8 \
    --num_epochs 100 \
    --learning_rate 1e-5 \
    --output_dir "/data/hojun313/JP/heightmap_diffusion_checkpoints" \
    --image_height 256 \
    --image_width 256 \
    --use_data_augmentation False \
    --mode '1' \
    --save_interval_steps 10000
)

python trainGPU.py "${PYTHON_ARGS[@]}"


PYTHON_EXIT_CODE=$?
if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Python script (trainGPU.py) exited with code $PYTHON_EXIT_CODE." >&2
    if [ $PYTHON_EXIT_CODE -eq 2 ]; then
        echo "This might be an argparse error. Check required arguments and their values for trainGPU.py."
    elif [ $PYTHON_EXIT_CODE -eq 127 ]; then
        echo "This might indicate 'command not found' issue related to the python execution line or python itself."
    fi
fi
echo "---------------------------------------------------------------------"

# --- 로컬 데이터셋 정리 (공간 확보를 위해 중요) ---
echo "Final cleanup of local dataset directory: $LOCAL_SCRATCH_BASE"
if [ -d "$LOCAL_SCRATCH_BASE" ]; then
    rm -rf "$LOCAL_SCRATCH_BASE"
    echo "Local dataset cleaned up successfully."
else
    echo "WARNING: Local scratch directory $LOCAL_SCRATCH_BASE not found for final cleanup."
fi
echo "---------------------------------------------------------------------"

echo "Job finished with exit code $PYTHON_EXIT_CODE."
exit $PYTHON_EXIT_CODE