#!/bin/bash
CUDA_VISIBLE_DEVICES=1

DATASETS=( "bioasq9b" "scifact" "mutual" )
# DATASETS=( "bioasq9b" "mutual" )
LM_NAMES="bert-base-uncased bert-base-cased roberta-base dmis-lab/biobert-base-cased-v1.1 google/electra-base-discriminator \
           princeton-nlp/unsup-simcse-bert-base-uncased princeton-nlp/sup-simcse-bert-base-uncased facebook/bart-base \
           allenai/scibert_scivocab_cased allenai/scibert_scivocab_uncased distilbert-base-cased distilbert-base-uncased \
           nghuyong/ernie-2.0-base-en distilroberta-base distilbert-base-multilingual-cased albert-base-v2 \
           microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext michiyasunaga/BioLinkBERT-base \
           openai-gpt distilgpt2"
# LM_NAMES="bert-base-uncased bert-base-cased roberta-base dmis-lab/biobert-base-cased-v1.1 google/electra-base-discriminator \
#            princeton-nlp/unsup-simcse-bert-base-uncased princeton-nlp/sup-simcse-bert-base-uncased openai-gpt facebook/bart-base \
#            allenai/scibert_scivocab_cased allenai/scibert_scivocab_uncased distilbert-base-cased"
CACHE_DIR=/home/workspace/models
# SEEDS="1027"
SEEDS="1117 1114 1027 820 905"
ALL_CANDIDATE_SIZES="2 3 4 5 6 7 8 9 10"
# ALL_CANDIDATE_SIZES="6 7 8 9 10"
METHODS="Logistic GBC TransRate HScore SFDA PACTran-0.1-10 LogME"

# METHODS="PACTran-0.1-10 PACTran-0.1-1 PACTran-1-100 PACTran-1-10 PACTran-10-1000 PACTran-10-100"
SAMPLING_METHOD="BM25"
# METHODS="NLEEP"
# METHODS="TMI-3"
SAVE_RESULTS="True"
OVERWRITE_EMBEDDING_CACHE="False"
OVERWRITE_RESULTS="False"


# iterate over datasets
for dt_idx in "${!DATASETS[@]}"; do
  dataset=${DATASETS[$dt_idx]}
  python model_selection.py \
    --methods $METHODS \
    --all_candidate_sizes $ALL_CANDIDATE_SIZES \
    --candidate_sampling_method $SAMPLING_METHOD \
    --aggregate \
    --agg_rate 0.8 \
    --dataset $dataset \
    --batch_size 16 \
    --model_name_or_paths ${LM_NAMES} \
    --matching_func dot \
    --pooler mean \
    --cache_dir ${CACHE_DIR} \
    --seeds ${SEEDS} \
    --save_results ${SAVE_RESULTS} \
    --overwrite_embedding_cache ${OVERWRITE_EMBEDDING_CACHE} \
    --overwrite_results ${OVERWRITE_RESULTS}
done

