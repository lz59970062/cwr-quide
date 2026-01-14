source .venv/bin/activate 
export CUDA_VISIBLE_DEVICES=1
set -e # 如果任何命令失败，立即退出

# --- 1. 设置路径 ---
# 如果提供了第一个参数($1)，则使用它作为输入路径，否则使用默认值
INPUT_PATH=${1:-"../Augmented_Datasets/_EUVP/test_samples/Inp"}
# 如果提供了第二个参数($2)，则使用它作为输出路径，否则使用默认值
OUTPUT_PATH=${2:-"./infer_results/"}

# 其他固定路径
GT_PATH="../Augmented_Datasets/_EUVP/test_samples/GTr" # 数据准备需要用到的GT路径
PREPARED_DATA_PATH="./val_data" # 存放预处理后数据的目录

echo "================================================="
echo "推理脚本开始"
echo "输入目录: ${INPUT_PATH}"
echo "输出目录: ${OUTPUT_PATH}"
echo "================================================="

# --- 2. 环境与权重准备 ---
echo "[步骤 1/3] 准备运行环境..."
# 如果 checkpoints 不存在，则创建并放置权重
mkdir -p checkpoints/cwr_finetune
if [ -d "cwr_finetune" ]; then
    cp cwr_finetune/*.pth checkpoints/cwr_finetune/ 2>/dev/null || true
fi

# --- 3. 执行推理 ---
echo "[步骤 2/3] 开始推理..."

# 检查输入路径是否存在
if [ ! -d "${INPUT_PATH}" ]; then
    echo "错误: 输入目录 ${INPUT_PATH} 不存在。"
    exit 1
fi

mkdir -p "${OUTPUT_PATH}"

# 使用 cwr 模型进行推理
# --dataset_mode single: 允许只输入一个文件夹的图片，无需 trainA/trainB 结构
# --model cwr: 使用 CWR 模型
# --name cwr_finetune: 对应 checkpoints/cwr_finetune 文件夹
# --preprocess none: 不进行缩放，保持原图分辨率
# --results_dir: 结果根目录
# --gpu_ids: 如果没有 GPU，代码已修复可以自动切到 CPU

python test.py --dataroot "${INPUT_PATH}" \
               --name cwr_test_all \
               --model cwr \
               --dataset_mode single \
               --preprocess none \
               --results_dir "${OUTPUT_PATH}" \
               --no_dropout \
               --eval

# --- 4. 整理结果 ---
echo "[步骤 3/3] 整理输出文件..."
if [ -d "${OUTPUT_PATH}/good" ]; then
    echo "推理成功！"
    echo "原图（坏图）存放于: ${OUTPUT_PATH}/bad"
    echo "增强图（好图）存放于: ${OUTPUT_PATH}/good"
else
    echo "警告: 推理完成，但未找到预期的结果目录: ${OUTPUT_PATH}/good"
fi

echo "================================================="
echo "推理过程结束"
echo "================================================="