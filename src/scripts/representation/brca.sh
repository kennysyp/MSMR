#!/bin/bash

gpuid=$1

declare -a dataroots_combine=(
	'path/to/tcga_brca_combined_feature'
)
declare -a dataroots_x20=(
	'path/to/tcga_brca_x20_feature'
)
declare -a dataroots_x40=(
	'path/to/tcga_brca_x40_feature'
)



# Loop through different folds
for k in 0 1 2 3 4; do
	split_dir="survival/TCGA_BRCA_overall_survival_k=${k}"
	split_names="train"
    bash "./scripts/representation/clustering.sh" $gpuid $split_dir $split_names "${dataroots_combine[@]}" "${dataroots_x20[@]}" "${dataroots_x40[@]}"
done