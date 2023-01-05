#!/bin/bash
conda init bash
conda activate clms_project


fairseq-eval-lm /home2/awlapas/project/preprocessed_data \
--path /home2/awlapas/project/checkpoints/npi_transformer/checkpoint_best.pt \
--results-path eval_results --batch-size 2 \
--tokens-per-sample 512 --context-window 400 \
--user-dir /home2/awlapas/project/src