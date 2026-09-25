# LatentSpec

We use the speculative decoding training framework from [DeepSpec](https://github.com/deepseek-ai/DeepSpec) to implement the training and evaluation code for LatentSpec. We also integrate the speculative decoding algorithms into [SGLang](https://github.com/sgl-project/sglang) to benchmark the end-to-end performance of different speculative decoding methods.

We thank the DeepSeek and SGLang teams for their outstanding work!

## Installation

We recommend using `uv` to manage the project environment.

```
cd LatentSpec
uv venv --python 3.12 --seed
source .venv/bin/activate
cd DeepSpec && uv pip install -r requirements.txt
cd ../sglang && uv pip install -e python/
```



## Training

Follow `DeepSpec/scripts/data/README.md` to generate the data required for training.

Before running the training script, set the following environment variables:

```bash
# Path to the cached training data
export target_cache_dir=...
# Directory for saving training checkpoints
export TRAIN_LOG_CHECKPOINTS=...
# Speculative decoding algorithm. Options: eagle3, dflash, dspark, latentspec, dspark_latentspec
export spec_mode=...

bash DeepSpec/scripts/train/train.sh
```



## Acceptance Length Benchmark

```bash
# Set the checkpoint directory and the checkpoint step to evaluate
export CHECKPOINT_DIR=...
export STRIDE=...
bash DeepSpec/scripts/eval/eval.sh
```



## Speed Benchmark

```
# Use the SGLang service launch script for each speculative decoding method in
# sglang/latentspec_scripts, then benchmark all methods with the same
# sglang/latentspec_scripts/run_tps.sh script.

# Start the service
export DRAFT_MODEL=...
bash sglang/latentspec_scripts/run_sglang_latentspec.sh 
# Run the benchmark
bash sglang/latentspec_scripts/run_tps.sh

```
