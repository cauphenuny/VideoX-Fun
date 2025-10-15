set -e
# conda activate /mnt/jydai111/envs/ycp_video_x_fun_cp310
if [ -z $visible ]; then unset ASCEND_RT_VISIBLE_DEVICES; unset NPU_VISIBLE_DEVICES; unset CUDA_VISIBLE_DEVICES; else export ASCEND_RT_VISIBLE_DEVICES=$visible; export NPU_VISIBLE_DEVICES=$visible; export CUDA_VISIBLE_DEVICES=$visible; fi
python examples/wan2.2_fun/predict_v2v_control_camera_5b.py $@

