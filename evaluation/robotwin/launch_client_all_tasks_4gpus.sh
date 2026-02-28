#!/bin/bash
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH


# save_root=${1:-'./results'}
# save_root=${1:-'/cpfs04/user/caizetao/projects/IDM_ws/lingbot-va/results'}
save_root=${1:-'/cpfs04/user/caizetao/projects/IDM_ws/lingbot-va/results_w_pred'}

# General parameters
policy_name=ACT
task_config=demo_clean
train_config_name=0
model_name=0
seed=${3:-0}
test_num=${4:-100}
start_port=29556
num_gpus=4

# task_groups=(
#   "stack_bowls_three handover_block hanging_mug scan_object"
#   "lift_pot put_object_cabinet stack_blocks_three place_shoe"
#   "adjust_bottle place_mouse_pad dump_bin_bigbin move_pillbottle_pad" 
#   "pick_dual_bottles shake_bottle place_fan turn_switch"
#   "shake_bottle_horizontally place_container_plate rotate_qrcode place_object_stand" 
#   "put_bottles_dustbin move_stapler_pad place_burger_fries place_bread_basket"
#   "pick_diverse_bottles open_microwave beat_block_hammer press_stapler" 
#   "click_bell move_playingcard_away open_laptop move_can_pot"
#   "stack_bowls_two place_a2b_right stamp_seal place_object_basket" 
#   "handover_mic place_bread_skillet stack_blocks_two place_cans_plasticbox"
#   "click_alarmclock blocks_ranking_size place_phone_stand place_can_basket" 
#   "place_object_scale place_a2b_left grab_roller place_dual_shoes"
#   "place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb"
# )

task_groups=(
  "stack_bowls_three handover_block hanging_mug scan_object"
)

start_group_id=${2:-0}
# end_group_id=${5:-$(( ${#task_groups[@]} - 1 ))}
end_group_id=${5:-0}

if (( start_group_id < 0 || start_group_id >= ${#task_groups[@]} )); then
  echo "start_group_id out of range: $start_group_id (0..$(( ${#task_groups[@]} - 1 )))" >&2
  exit 1
fi

if (( end_group_id < start_group_id || end_group_id >= ${#task_groups[@]} )); then
  echo "end_group_id out of range: $end_group_id (start_group_id..$(( ${#task_groups[@]} - 1 )))" >&2
  exit 1
fi

task_names=()
for (( group_id=start_group_id; group_id<=end_group_id; group_id++ )); do
  read -r -a group_tasks <<< "${task_groups[$group_id]}"
  task_names+=( "${group_tasks[@]}" )
done

echo "start_group_id=$start_group_id"
echo "end_group_id=$end_group_id"
printf 'task_names (%d): %s\n' "${#task_names[@]}" "${task_names[*]}"

log_dir="./logs"
mkdir -p "$log_dir"

echo -e "\033[32mLaunching ${#task_names[@]} tasks with max parallel ${num_gpus}. Ports ${start_port}-$(( start_port + num_gpus - 1 )).\033[0m"

pid_file="pids.txt"
> "$pid_file"

batch_time=$(date +%Y%m%d_%H%M%S)

ports=()
gpus=()
for (( i=0; i<num_gpus; i++ )); do
  ports+=( $(( start_port + i )) )
  gpus+=( $i )
done

free_ports=( "${ports[@]}" )
free_gpus=( "${gpus[@]}" )
running_pids=()
declare -A pid_gpu pid_port pid_task pid_log

pop_front() {
  local arr_name="$1"
  local out_var="$2"
  local val
  eval "val=\"\${${arr_name}[0]}\""
  eval "${arr_name}=(\"\${${arr_name}[@]:1}\")"
  printf -v "$out_var" '%s' "$val"
}

launch_task() {
  local task_name="$1"
  local gpu_id="$2"
  local port="$3"

  export CUDA_VISIBLE_DEVICES=${gpu_id}
  local log_file="${log_dir}/${task_name}_${batch_time}.log"

  echo -e "\033[33m[Launch] Task: ${task_name}, GPU: ${gpu_id}, PORT: ${port}, Log: ${log_file}\033[0m"

  PYTHONWARNINGS=ignore::UserWarning \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python -u -m evaluation.robotwin.eval_polict_client_openpi --config policy/$policy_name/deploy_policy.yml \
      --overrides \
      --task_name ${task_name} \
      --task_config ${task_config} \
      --train_config_name ${train_config_name} \
      --model_name ${model_name} \
      --ckpt_setting ${model_name} \
      --seed ${seed} \
      --policy_name ${policy_name} \
      --save_root ${save_root} \
      --video_guidance_scale 5 \
      --action_guidance_scale 1 \
      --test_num ${test_num} \
      --port ${port} > "$log_file" 2>&1 &

  local pid=$!
  running_pids+=( "$pid" )
  pid_gpu["$pid"]="$gpu_id"
  pid_port["$pid"]="$port"
  pid_task["$pid"]="$task_name"
  pid_log["$pid"]="$log_file"
  echo "${pid}" | tee -a "$pid_file"
}

task_index=0
task_total=${#task_names[@]}

while (( task_index < task_total || ${#running_pids[@]} > 0 )); do
  while (( task_index < task_total && ${#free_gpus[@]} > 0 )); do
    task_name="${task_names[$task_index]}"
    pop_front free_gpus gpu_id
    pop_front free_ports port
    launch_task "$task_name" "$gpu_id" "$port"
    task_index=$(( task_index + 1 ))
  done

  if (( ${#running_pids[@]} > 0 )); then
    wait -n
    alive_pids=()
    for pid in "${running_pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        alive_pids+=( "$pid" )
      else
        finished_task="${pid_task[$pid]}"
        finished_gpu="${pid_gpu[$pid]}"
        finished_port="${pid_port[$pid]}"
        finished_log="${pid_log[$pid]}"
        echo -e "\033[32m[Done] Task: ${finished_task}, GPU: ${finished_gpu}, PORT: ${finished_port}, Log: ${finished_log}\033[0m"
        free_gpus+=( "$finished_gpu" )
        free_ports+=( "$finished_port" )
        unset pid_gpu["$pid"] pid_port["$pid"] pid_task["$pid"] pid_log["$pid"]
      fi
    done
    running_pids=( "${alive_pids[@]}" )
  fi
done

echo -e "\033[32mAll tasks launched. PIDs saved to ${pid_file}\033[0m"
echo -e "\033[36mTo terminate all processes, run: kill \$(cat ${pid_file})\033[0m"
