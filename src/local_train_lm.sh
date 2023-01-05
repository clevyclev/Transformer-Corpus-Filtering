#!/bin/bash


for i in {1..5}
do
	fairseq-train --task language_modeling \
	preprocessed_data \
	--save-dir checkpoints/npi_transformer_$i \
	--arch npi_transformer --user-dir src \
	--log-file logs/npi_transformer_$i \
	--tokens-per-sample 512 --sample-break-mode none \
	--optimizer adam --lr 0.0005 \
	--max-tokens 2048 --update-freq 16 \
	--fp16 \
	--no-epoch-checkpoints \
	--decoder-layers 8 \
	--decoder-input-dim 512 \
	--decoder-ffn-embed-dim 2048 \
	--decoder-embed-dim 512 \
	--decoder-attention-heads 8 \
	--max-target-positions 512 \
	--patience 2 \
	--seed $i

	fairseq-eval-lm preprocessed_data \
	--path npi_transformer_$i/checkpoint_best.pt \
	--results-path eval_results/npi_transformer_$i --batch-size 8 \
	--tokens-per-sample 512 --context-window 400 \
	--user-dir src
done