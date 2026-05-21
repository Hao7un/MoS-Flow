# A2A Policy (IL)

A2A (Action-to-Action) is a flow matching policy that directly transforms history action distributions to future action distributions, conditioned on visual observations.

## Variants

- **A2A**: Base Action-to-Action flow matching policy
- **A2A-Noise**: A variant that adds Gaussian noise to history states before encoding for improved robustness

## Architecture

```
History States [s_{t-n+1}, ..., s_t] --encode--> history_latents (x_0)
Visual Obs [img_{t-n+1}, ..., img_t] --encode--> obs_latents (condition)

Flow Matching: x_0 --flow(condition)--> x_1 (future_action_latents)

x_1 --decode--> Future Actions [a_t, a_{t+1}, ..., a_{t+k}]
```

## Install

```bash
cd roboverse_learn/il/policies/a2a
pip install -r requirements.txt
```

Create a Weights & Biases account to obtain an API key for logging.

## Collect and process data

```bash
./roboverse_learn/il/collect_demo.sh
```

## Train and eval

### A2A
```bash
bash roboverse_learn/il/il_run.sh --task_name_set close_box --policy_name a2a
```

### A2A-Noise
```bash
bash roboverse_learn/il/il_run.sh --task_name_set close_box --policy_name a2a_noise
```

Inside `il_run.sh` you can toggle `train_enable` / `eval_enable`, set task names, seeds, GPU id, and checkpoint paths for evaluation.

