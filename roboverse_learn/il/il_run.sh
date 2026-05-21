#!/bin/bash
# Usage: bash roboverse_learn/il/il_run.sh --task_name_set close_box --policy_name ddpm_dit --dr_level_eval 2 --train_enable False
export PYTHONPATH=$(pwd):$PYTHONPATH

# Force IsaacSim to exit cleanly after close (avoid shutdown hang)
export METASIM_FORCE_EXIT_ON_CLOSE=1
export METASIM_CLOSE_TIMEOUT_SEC=${METASIM_CLOSE_TIMEOUT_SEC:-8}

task_name_set="pick_cube" # Tasks, e.g., close_box, stack_cube, pick_cube
policy_name="a2a"    # IL policy, opts: ddpm_unet, ddpm_dit, ddim_unet, fm_unet, fm_dit, vita, a2a, a2a_dit, mos_flow, act, score
sim_set="isaacsim"          # Simulator, e.g., mujoco, isaacsim
demo_num=100              # Number of demonstrations to collect, train, and eval

# Training/eval control
train_enable=True   # True for training, False for evaluation
eval_enable=True

# Training parameters
num_epochs=100
seed=42
eval_seed=42
gpu=0
obs_space=joint_pos
act_space=joint_pos
delta_ee=0
eval_num_envs=1
eval_max_step=300
exp_name=""

# Domain Randomization Level
dr_level_collect=0
dr_level_eval=0

# Extra hydra overrides (collected from unknown args)
hydra_overrides=""

# Parse parameters
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task_name_set)
            task_name_set="$2"
            shift 2
            ;;
        --policy_name)
            policy_name="$2"
            shift 2
            ;;
        --sim_set)
            sim_set="$2"
            shift 2
            ;;
        --demo_num)
            demo_num="$2"
            shift 2
            ;;
        --train_enable)
            train_enable="$2"
            shift 2
            ;;
        --eval_enable)
            eval_enable="$2"
            shift 2
            ;;
        --dr_level_collect)
            dr_level_collect="$2"
            shift 2
            ;;
        --dr_level_eval)
            dr_level_eval="$2"
            shift 2
            ;;
        --num_epochs)
            num_epochs="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        --eval_seed)
            eval_seed="$2"
            shift 2
            ;;
        --gpu)
            gpu="$2"
            shift 2
            ;;
        --exp_name)
            exp_name="$2"
            shift 2
            ;;
        --obs_space)
            obs_space="$2"
            shift 2
            ;;
        --act_space)
            act_space="$2"
            shift 2
            ;;
        --delta_ee)
            delta_ee="$2"
            shift 2
            ;;
        --eval_num_envs)
            eval_num_envs="$2"
            shift 2
            ;;
        --eval_max_step)
            eval_max_step="$2"
            shift 2
            ;;
        *)
            # Collect unknown args as hydra overrides (e.g., policy_config.dropout=0.1)
            hydra_overrides="${hydra_overrides} $1"
            shift
            ;;
    esac
done

# # Collect demo
# echo "=== Running collect_demo.sh ==="
# sed -i "s/^task_name_set=.*/task_name_set=$task_name_set/" ./roboverse_learn/il/collect_demo.sh
# sed -i "s/^sim_set=.*/sim_set=$sim_set/" ./roboverse_learn/il/collect_demo.sh
# sed -i "s/^num_demo_success=.*/num_demo_success=$demo_num/" ./roboverse_learn/il/collect_demo.sh
# sed -i "s/^expert_data_num=.*/expert_data_num=$demo_num/" ./roboverse_learn/il/collect_demo.sh
# sed -i "s/^random_level=.*/random_level=$dr_level_collect/" ./roboverse_learn/il/collect_demo.sh
# bash ./roboverse_learn/il/collect_demo.sh

if [[ "${gpu}" == cuda:* || "${gpu}" == "cpu" ]]; then
    device_arg="${gpu}"
else
    device_arg="cuda:${gpu}"
fi

# Map policy_name to model config
config_name="default_runner"
main_script="./roboverse_learn/il/train.py"

# if policy_name is ACT
if [ "${policy_name}" = "act" ]; then
    echo "=== Running ACT training and evaluation==="
    sed -i "s/^task_name_set=.*/task_name_set=$task_name_set/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^sim_set=.*/sim_set=$sim_set/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^expert_data_num=.*/expert_data_num=$demo_num/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^train_enable=.*/train_enable=$train_enable/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^eval_enable=.*/eval_enable=$eval_enable/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^collect_level=.*/collect_level=$dr_level_collect/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^eval_level=.*/eval_level=$dr_level_eval/" ./roboverse_learn/il/policies/act/act_run.sh
    bash ./roboverse_learn/il/policies/act/act_run.sh
    echo "=== Completed all data collection, training, and evaluation ==="
    exit 0
fi

# Run training/evaluation for DP/FM/VITA policies
echo "=== Running ${policy_name} ==="

eval_ckpt_name=$num_epochs
if [ -z "${exp_name}" ]; then
    exp_name="default"
fi
output_dir="./il_outputs/${policy_name}/${exp_name}"
eval_path="${output_dir}/${task_name_set}/checkpoints/${eval_ckpt_name}.ckpt"

echo "Checkpoint path: $eval_path"

extra="obs:${obs_space}_act:${act_space}"
if [ "${delta_ee}" = 1 ]; then
  extra="${extra}_delta"
fi

export policy_name="${policy_name}"
python ${main_script} --config-name=${config_name}.yaml \
task_name=${task_name_set} \
"dataset_config.zarr_path=./data_policy/${task_name_set}FrankaL${dr_level_collect}_${extra}_${demo_num}.zarr" \
train_config.training_params.seed=${seed} \
train_config.training_params.num_epochs=${num_epochs} \
train_config.training_params.device=${device_arg} \
eval_config.policy_runner.obs.obs_type=${obs_space} \
eval_config.policy_runner.action.action_type=${act_space} \
eval_config.policy_runner.action.delta=${delta_ee} \
eval_config.eval_args.task=${task_name_set} \
eval_config.eval_args.max_step=${eval_max_step} \
eval_config.eval_args.num_envs=${eval_num_envs} \
eval_config.eval_args.sim=${sim_set} \
eval_config.eval_args.level=${dr_level_eval} \
eval_config.eval_args.randomization_seed=${eval_seed} \
+eval_config.eval_args.max_demo=50 \
exp_name=${exp_name:-default} \
train_enable=${train_enable} \
eval_enable=${eval_enable} \
eval_path=${eval_path} \
${hydra_overrides}

echo "=== Completed all data collection, training, and evaluation ==="
