#!/usr/bin/env bash
set -euo pipefail

## run nvidia-smi to check available GPUs
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

# Force IsaacSim to exit cleanly after close (avoid shutdown hang)
export METASIM_FORCE_EXIT_ON_CLOSE=1
export METASIM_CLOSE_TIMEOUT_SEC=${METASIM_CLOSE_TIMEOUT_SEC:-8}

python - <<'PY_CHECK'
try:
    import metasim  # noqa: F401
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: metasim is not installed in the active environment.\n"
        "Install it from the repo root with e.g.:\n"
        "  python -m pip install -e '.[learn,isaacsim]'\n"
        "or, for MuJoCo collection:\n"
        "  python -m pip install -e '.[learn,mujoco]'"
    ) from exc
PY_CHECK

## Parameters
task_name_set=close_box
random_level=0          # Randomization level: 0=None, 1=Scene+Material, 2=+Light, 3=+Camera
num_envs=1              # Number of parallel environments
demo_start_idx=0        # Index of the first demo to collect
sim_set=isaacsim
cust_name=test
num_demo_success=100

expert_data_num=100

obs_space=joint_pos
act_space=joint_pos
delta_ee=0              # Delta control
extra="obs:${obs_space}_act:${act_space}"
if [ "${delta_ee}" = 1 ]; then
  extra="${extra}_delta"
fi

success_dir="./roboverse_demo/demo_${sim_set}/${task_name_set}-${cust_name}/robot-franka/success"

## Collecting demonstration data
set +e
python ./scripts/advanced/collect_demo.py \
--sim=${sim_set} \
--task=${task_name_set} \
--num_envs=${num_envs} \
--run_unfinished \
--headless \
--demo_start_idx=${demo_start_idx} \
--num_demo_success ${num_demo_success} \
--cust_name=${cust_name} \
--level=${random_level}
collect_status=$?
set -e

if [ "${collect_status}" -ne 0 ]; then
  if [ -d "${success_dir}" ]; then
    valid_demo_count=$(find "${success_dir}" -mindepth 2 -maxdepth 2 -name metadata.json | wc -l | tr -d ' ')
  else
    valid_demo_count=0
  fi

  if [ "${valid_demo_count}" -lt "${num_demo_success}" ]; then
    echo "ERROR: collect_demo.py exited with ${collect_status}, and only ${valid_demo_count}/${num_demo_success} valid demos were found."
    exit "${collect_status}"
  fi

  echo "WARNING: collect_demo.py exited with ${collect_status}, but ${valid_demo_count}/${num_demo_success} demos exist. Continuing to zarr conversion."
fi

## Convert demonstration data
python ./roboverse_learn/il/data2zarr_dp.py \
--task_name ${task_name_set}FrankaL${random_level}_${extra} \
--expert_data_num ${expert_data_num} \
--metadata_dir ./roboverse_demo/demo_${sim_set}/${task_name_set}-${cust_name}/robot-franka/success \
--action_space ${act_space} \
--observation_space ${obs_space}
