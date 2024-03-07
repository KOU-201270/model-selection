export  CUDA_VISIBLE_DEVICES=0
DATASET=bioasq9b
POOLER=mean
BATCH_SIZE=32
NUM_EPOCHS=10
CACHE_DIR=/home/workspace/models
for PLM in bert-base-cased
do
    for LR in 3e-5
    do
        SAVE_DIR=./output/${DATASET}/fine_tuning/${PLM#*/}_${POOLER}/bs${BATCH_SIZE}_e${NUM_EPOCHS}_lr${LR}/
        mkdir -p ${SAVE_DIR}
        nohup python3 train_reqa.py \
            --seeds 0 42 512 2023 20246 \
            --main_metric mrr \
            --dataset ${DATASET} \
            --epoch ${NUM_EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            --model_name_or_path ${PLM} \
            --pooler ${POOLER} \
            --matching_func dot \
            --temperature 1 \
            --learning_rate ${LR} \
            --save_dir ${SAVE_DIR} \
            --cache_dir ${CACHE_DIR} \
            --rm_saved_model True \
            --save_results True \
            > ${SAVE_DIR}/run.log 2>&1
    done
done

# PLM=emilyalsentzer/Bio_ClinicalBERT
# LR=2e-5
# SAVE_DIR=./output/test
# python3 train_reqa.py \
#         --seed 42 \
#         --main_metric mrr \
#         --dataset ${DATASET} \
#         --max_question_len 24 \
#         --max_answer_len 168 \
#         --epoch ${NUM_EPOCHS} \
#         --batch_size ${BATCH_SIZE} \
#         --model_name_or_path ${PLM} \
#         --pooler ${POOLER} \
#         --matching_func dot \
#         --temperature 1 \
#         --learning_rate ${LR} \
#         --save_dir ${SAVE_DIR} \
#         --cache_dir ${CACHE_DIR} \
#         --rm_saved_model True \
#         --save_results False \