cd /path/to/EnDora
conda activate EnDora


declare -a test_names=(
    # test names    
)

declare -a source_paths=(
    # data frames
)


gpu_id=GPU_ID

for i in "${!test_names[@]}"; do
    cd /path/to/EnDora
    test_name=/path/to/vidoes/generated/during/test/${test_names[i]}
    source_path=${source_paths[i]}
    test_frame=/path/to/frames/${test_names[i]}
    CUDA_VISIBLE_DEVICES=$gpu_id python process_data.py -s $source_path -t $test_frame

    cd /path/to/stylegan-v
    CUDA_VISIBLE_DEVICES=$gpu_id python ./src/scripts/calc_metrics_for_dataset.py --fake_data_path $test_frame --real_data_path $source_path 

done
