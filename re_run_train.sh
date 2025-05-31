#!/bin/bash

#SBATCH -J HeightmapTrain
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ce_ugrad
#SBATCH -t 1-00:00:00
#SBATCH -o /data/hojun313/JP/slurm_logs/slurm-%A.out
#SBATCH -e /data/hojun313/JP/slurm_logs/slurm-%A.err

# --- 스크립트 디버깅 옵션 ---
set -e # 명령어가 0이 아닌 종료 코드로 끝나면 즉시 스크립트 종료

# --- 초기 환경 설정 ---
echo "---------------------------------------------------------------------"
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
PROJECT_DIR="/data/hojun313/JP" # Python 스크립트들이 있는 프로젝트 최상위 경로
echo "Project Directory: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR/slurm_logs"
echo "Slurm logs will be saved to $PROJECT_DIR/slurm_logs/"
echo "---------------------------------------------------------------------"

# --- 실행 인자 확인 ---
MODEL_TAG_ARG=$1
if [ -z "$MODEL_TAG_ARG" ]; then
    echo "오류: 모델 태그가 첫 번째 인자로 전달되지 않았습니다. (예: sbatch run_training_v2.sh my_experiment_tag)" >&2
    exit 1
fi
echo "Model Tag for this run: $MODEL_TAG_ARG"
echo "---------------------------------------------------------------------"

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

# --- 데이터셋 준비 (사용자 제공 로직 기반) ---
NAS_TARBALL_PATH="/data/hojun313/JP/heightmap_dataset_83.tar"
LOCAL_SCRATCH_PARENT_DIR="/local_datasets/hojun313" # 주의: 이 폴더 자체를 정리하는 로직은 제외함
CURRENT_JOB_SCRATCH_DIR_NAME="${MODEL_TAG_ARG}_${SLURM_JOB_ID}"
LOCAL_SCRATCH_BASE="${LOCAL_SCRATCH_PARENT_DIR}/${CURRENT_JOB_SCRATCH_DIR_NAME}"
# tar 파일 안에 heightmap_dataset 폴더가 있으므로, Python 스크립트가 읽을 최종 경로는 아래와 같음
LOCAL_DATA_FOR_PYTHON_SCRIPT="${LOCAL_SCRATCH_BASE}/heightmap_dataset"

echo "Preparing dataset..."
echo "Attempting to create current job's local scratch directory: $LOCAL_SCRATCH_BASE"
mkdir -p "$LOCAL_SCRATCH_BASE"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create current job's local scratch directory '$LOCAL_SCRATCH_BASE'." >&2
    exit 1
fi
echo "Current job's local scratch directory created successfully: $LOCAL_SCRATCH_BASE"

echo "Copying dataset tarball from $NAS_TARBALL_PATH to ${LOCAL_SCRATCH_BASE}/ ..."
# tarball 이름을 명시적으로 지정 (예: heightmap_dataset_83.tar)
cp "$NAS_TARBALL_PATH" "${LOCAL_SCRATCH_BASE}/heightmap_dataset_83.tar"
COPY_EXIT_CODE=$?

echo "--- Debugging file copy ---"
echo "NAS Tarball Path: $NAS_TARBALL_PATH"; ls -lh "$NAS_TARBALL_PATH"
echo "Local Scratch Tarball: ${LOCAL_SCRATCH_BASE}/heightmap_dataset_83.tar"; ls -lh "${LOCAL_SCRATCH_BASE}/heightmap_dataset_83.tar"
echo "Exit code of cp command: $COPY_EXIT_CODE"
echo "--- Debugging file copy end ---"

if [ $COPY_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Failed to copy dataset tarball to local scratch. CP Exit Code: $COPY_EXIT_CODE." >&2
    rm -rf "$LOCAL_SCRATCH_BASE"
    exit 1
fi
if [ ! -f "${LOCAL_SCRATCH_BASE}/heightmap_dataset_83.tar" ] || [ ! -s "${LOCAL_SCRATCH_BASE}/heightmap_dataset_83.tar" ]; then
    echo "ERROR: Tarball was not copied correctly or is empty in local scratch." >&2
    ls -lh "$LOCAL_SCRATCH_BASE/"
    rm -rf "$LOCAL_SCRATCH_BASE"
    exit 1
fi
echo "Dataset tarball copied successfully and seems valid."

echo "Extracting dataset tarball in $LOCAL_SCRATCH_BASE ..."
echo "--- Debugging before tar extraction ---"
echo "Target tar file for extraction: ${LOCAL_SCRATCH_BASE}/heightmap_dataset_83.tar"
ls -lh "${LOCAL_SCRATCH_BASE}/heightmap_dataset_83.tar"
echo "Extraction directory: $LOCAL_SCRATCH_BASE"
df -h "$LOCAL_SCRATCH_BASE" # 사용 가능한 공간 확인
echo "--- Debugging before tar extraction end ---"

# -C 옵션으로 $LOCAL_SCRATCH_BASE 디렉토리 안에 내용물을 푼다.
# tar 파일 안의 내용물 경로가 heightmap_dataset/MaterialA/... 이므로,
# $LOCAL_SCRATCH_BASE 안에는 heightmap_dataset 폴더가 생성될 것임.
tar -vxf "${LOCAL_SCRATCH_BASE}/heightmap_dataset_83.tar" -C "$LOCAL_SCRATCH_BASE" # -v 옵션 추가로 상세 출력
TAR_EXIT_CODE=$?
if [ $TAR_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Failed to extract dataset tarball. Exit code: $TAR_EXIT_CODE" >&2
    echo "Listing contents of $LOCAL_SCRATCH_BASE after failed tar extraction:"
    ls -Alh "$LOCAL_SCRATCH_BASE"
    rm -rf "$LOCAL_SCRATCH_BASE"
    exit 1
fi
echo "Dataset extracted successfully."

# 압축 해제 후, 파이썬 스크립트가 사용할 경로의 내용 확인 (매우 중요)
echo "--- DEBUG: Verifying Python data root: $LOCAL_DATA_FOR_PYTHON_SCRIPT ---"
if [ -d "$LOCAL_DATA_FOR_PYTHON_SCRIPT" ]; then
    echo "Contents of $LOCAL_DATA_FOR_PYTHON_SCRIPT (first level):"
    ls -Alh "$LOCAL_DATA_FOR_PYTHON_SCRIPT"
    echo "Number of items in $LOCAL_DATA_FOR_PYTHON_SCRIPT:"
    ls -A "$LOCAL_DATA_FOR_PYTHON_SCRIPT" | wc -l
else
    echo "ERROR: Python data root $LOCAL_DATA_FOR_PYTHON_SCRIPT NOT FOUND after extraction!"
    echo "Listing contents of $LOCAL_SCRATCH_BASE for review:"
    ls -Alh "$LOCAL_SCRATCH_BASE"
    rm -rf "$LOCAL_SCRATCH_BASE"
    exit 1
fi

if [ -z "$(ls -A "$LOCAL_DATA_FOR_PYTHON_SCRIPT")" ]; then
    echo "ERROR: Python data root $LOCAL_DATA_FOR_PYTHON_SCRIPT IS EMPTY after extraction!"
    rm -rf "$LOCAL_SCRATCH_BASE"
    exit 1
fi
echo "Python data root $LOCAL_DATA_FOR_PYTHON_SCRIPT is populated."

echo "Removing local tarball ${LOCAL_SCRATCH_BASE}/heightmap_dataset_83.tar"
rm "${LOCAL_SCRATCH_BASE}/heightmap_dataset_83.tar"
echo "---------------------------------------------------------------------"

# --- 학습 스크립트 실행 ---
echo "Changing directory to $PROJECT_DIR" # Python 스크립트가 있는 프로젝트 경로로 이동
cd "$PROJECT_DIR"
echo "Current working directory: $(pwd)"

echo "Starting Python training script (re_train.py)..."
# re_train.py가 받는 인자에 맞게 수정
# (이전 trainGPU.py와 인자 이름이 다를 수 있으므로 re_train.py의 argparse 부분을 참고하여 정확히 맞춰야 함)
python re_train.py \
    --data_root "$LOCAL_DATA_FOR_PYTHON_SCRIPT" \
    --output_dir "/data/hojun313/JP/training_outputs" \
    --model_tag "$MODEL_TAG_ARG" \
    --batch_size 32 \
    --num_workers 8 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --encoder "resnet34" \
    --save_interval 5 \
    --image_size 256 \
    --lr_patience 10 \
    --lr_factor 0.2 \

PYTHON_EXIT_CODE=$?
if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Python script (re_train.py) exited with code $PYTHON_EXIT_CODE." >&2
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

echo "Job finished at $(date) with exit code $PYTHON_EXIT_CODE."
exit $PYTHON_EXIT_CODE