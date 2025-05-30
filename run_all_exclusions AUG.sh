#!/bin/bash

PROJECT_DIR="/data/hojun313/JP"
MATERIALS_FILE="${PROJECT_DIR}/materials.txt" # 재질 목록 파일 경로
SBATCH_SCRIPT="${PROJECT_DIR}/run_train.sh"   # 실행할 sbatch 스크립트
EXCLUSION_LOG_DIR="${PROJECT_DIR}/slurm_logs"
EXCLUSION_LOG_FILE="${EXCLUSION_LOG_DIR}/excluded_materials_per_group.txt" 

# Conda 환경 활성화
CONDA_ENV_NAME="JP_3rdTry_GPU"
echo "Activating Anaconda environment: $CONDA_ENV_NAME before submitting jobs..."
source /data/hojun313/anaconda3/bin/activate "$CONDA_ENV_NAME"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment '$CONDA_ENV_NAME' in submission script." >&2
    exit 1
fi

if [ ! -f "$MATERIALS_FILE" ]; then
    echo "오류: 재질 목록 파일 '$MATERIALS_FILE'을 찾을 수 없습니다." >&2
    exit 1
fi

# 제외 목록 로그 파일을 저장할 디렉토리 생성 (없으면)
mkdir -p "$EXCLUSION_LOG_DIR"

if [ ! -f "$EXCLUSION_LOG_FILE" ] || ! grep -q "MODEL_TAG,EXCLUDED_MATERIALS_COUNT,EXCLUDED_MATERIALS_LIST" "$EXCLUSION_LOG_FILE"; then
    echo "MODEL_TAG,EXCLUDED_MATERIALS_COUNT,EXCLUDED_MATERIALS_LIST" > "$EXCLUSION_LOG_FILE"
fi


mapfile -t ALL_MATERIALS_FROM_FILE < "$MATERIALS_FILE"
ALL_MATERIALS_CLEANED=()
for mat in "${ALL_MATERIALS_FROM_FILE[@]}"; do
    if [[ -n "$mat" ]]; then 
        ALL_MATERIALS_CLEANED+=("$mat")
    fi
done

TEMP_SHUFFLED_MATERIALS_FILE=$(mktemp)
printf "%s\n" "${ALL_MATERIALS_CLEANED[@]}" | if command -v shuf >/dev/null 2>&1; then shuf; else cat; fi > "$TEMP_SHUFFLED_MATERIALS_FILE"
mapfile -t ALL_MATERIALS < <(cat "$TEMP_SHUFFLED_MATERIALS_FILE")
rm "$TEMP_SHUFFLED_MATERIALS_FILE"

NUM_TOTAL_MATERIALS=${#ALL_MATERIALS[@]}
GROUP_SIZE=10 # 테스트용, 나중에 5로 변경 예정
GROUP_COUNT=0

echo "전체 재질 목록 (${NUM_TOTAL_MATERIALS}개, 무작위로 섞임):"
echo "각 작업마다 ${GROUP_SIZE}개 재질을 제외하여 학습을 제출합니다."

# 섞인 재질 목록을 GROUP_SIZE 만큼씩 순회
for (( i=0; i<NUM_TOTAL_MATERIALS; i+=GROUP_SIZE )); do
    CURRENT_GROUP_MATERIALS=("${ALL_MATERIALS[@]:i:GROUP_SIZE}")
    
    EXCLUDED_MATERIALS_STRING=""
    for material in "${CURRENT_GROUP_MATERIALS[@]}"; do
        EXCLUDED_MATERIALS_STRING+="${material} "
    done
    EXCLUDED_MATERIALS_STRING=$(echo "$EXCLUDED_MATERIALS_STRING" | sed 's/ *$//')

    # 모델 태그는 그룹 번호와 "shuffled"만 포함하여 간결하게 유지
    MODEL_TAG="excluded_group_${GROUP_COUNT}_shuffled"

    if [ -z "$EXCLUDED_MATERIALS_STRING" ]; then
        echo "Skipping empty exclusion group."
        continue
    fi

    echo "-----------------------------------------------------"
    echo "제출 작업 그룹 ${GROUP_COUNT}: 다음 ${#CURRENT_GROUP_MATERIALS[@]}개 재질 제외"
    # 제외 재질 목록은 너무 길 수 있으므로, 로그 파일 참조하도록 안내
    echo "  (제외 재질 상세 목록은 ${EXCLUSION_LOG_FILE} 파일 참조)" 
    echo "  모델 태그: ${MODEL_TAG}"
    echo "-----------------------------------------------------"
    
    # 현재 그룹의 제외 재질 목록을 로그 파일에 기록
    # 각 재질 이름에 공백이 없다고 가정하고, 공백으로 구분된 문자열로 저장
    echo "${MODEL_TAG},${#CURRENT_GROUP_MATERIALS[@]},\"${EXCLUDED_MATERIALS_STRING}\"" >> "$EXCLUSION_LOG_FILE"
    
    # sbatch 스크립트에 "제외할 재질들 문자열"과 "모델 태그"를 인자로 전달하여 제출
    sbatch "$SBATCH_SCRIPT" "${EXCLUDED_MATERIALS_STRING}" "${MODEL_TAG}"
    
    sleep 60 # Slurm에 너무 빠르게 연속적인 요청을 보내는 것을 방지
    GROUP_COUNT=$((GROUP_COUNT + 1))
done

echo "모든 작업 그룹 제출 완료 (${GROUP_COUNT}개 그룹)."
echo "제외된 재질 상세 목록은 ${EXCLUSION_LOG_FILE} 파일을 확인하세요."