# Finetune-Qwen2.5-VL

A comprehensive toolkit for finetuning the Qwen2.5-VL (Visual Language) model using LoRA. This project provides easy-to-use scripts for Supervised Fine-Tuning (SFT), LoRA merging, and inference.

## Requirements

```bash
# Core dependencies
torch>=2.0.0
transformers>=4.37.0
peft>=0.7.0
accelerate>=0.21.0
```

## Installation

```bash
git clone https://github.com/prstyaDev/Finetune-Qwen2.5-VL.git
cd Finetune-Qwen2.5-VL
pip install -r requirements.txt
```

## Usage

### 1.1 Supervised Fine-Tuning (SFT) - Single GPU

Run the following command to start the fine-tuning process:

```bash
python src/train.py config/vlm_config.yaml

or

python main.py config/vlm_config.yaml
```

The `vlm_config.yaml` should contain your training configurations such as:
- Model parameters
- Training hyperparameters
- Dataset configurations
- LoRA settings

Check the configurations `distributed_type: "NO"` in `accelerate.yaml`.

### 1.2 Supervised Fine-Tuning (SFT) - Multiple GPU

Run the following command to start the fine-tuning process:

```bash
python main.py config/vlm_config.yaml config/accelerate.yaml
```

The `vlm_config.yaml` should contain your training configurations and  `accelerate.yaml` should contain `accelerate` configurations. If you want to use multiple GPUs, set `use_accelerate: true` in `vlm_config.yaml` and `distributed_type: "MULTI_GPU", gpu_ids: all` in `accelerate.yaml`

A better approach would be to use Ray, but I am currently facing some issues that have yet to be resolved. I plan to work on this further in the future.

### 2. Merge LoRA Weights

After training, merge the LoRA weights with the base model:

```bash
python src/merge_model.py config/vlm_merge_adapter_config.yaml
```

### 3. Inference

Run inference with your fine-tuned model:

```bash
python src/inference.py config/vlm_inference_config.yaml
```

### 4. Evaluation
```bash
python evaluation/evaluation.py config/vlm_inference_config.yaml
```


### 5. Dockerfile
```bash
docker build --no-cache -t vlm_finetune:latest .
docker run -it --name CONATINER_NAME -v LOCAL_PATH:/VLM vlm_finetune:latest
```

## TODO
- [ ] Resolve issues with Ray for multi-GPU training
- [ ] Implement evaluation pipeline for fine-tuned models  
- [ ] Add test cases for training, merging, and inference  
- [ ] Load Data may be more flexible
- [ ] Load Data support Image_url in Trainig stage
- [ ] Finetune JSON Dataset


## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledge
This project is based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main). Special thanks to the original authors and contributors!
