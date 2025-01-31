import torch

if torch.cuda.is_available():
    cache_dir = ".cache"
else:
    cache_dir = "cache"



finetuned_model_backbone_mapping_dict = {
    "WizardLM-7B-V1.0": "llama-7b-hf",
    "WizardLM-7B-V1.0-recovered": "llama-7b-hf",
    "WizardLM-13B-V1.2": "Llama-2-13b-hf",
    "WizardLM-70B-V1.0": "Llama-2-70b-hf",
    "WizardMath-7B-V1.0": "Llama-2-7b-hf",
    "WizardMath-13B-V1.0": "Llama-2-13b-hf",
    "WizardMath-70B-V1.0": "Llama-2-70b-hf",
    "WizardCoder-Python-7B-V1.0": "CodeLlama-7b-Python-hf",
    "WizardCoder-Python-13B-V1.0": "CodeLlama-13b-Python-hf",
    "WizardCoder-Python-34B-V1.0": "CodeLlama-34b-Python-hf",
    "llama-2-13b-code-alpaca": "Llama-2-13b-hf",
    "Llama-2-7b-hf": "Llama-2-7b-hf",
    "deepseek-math-7b-rl": "deepseek-math-7b-base",
    "deepseek-math-7b-base": "deepseek-math-7b-base",
    'Math-LLaVA': 'llava-v1.5-13b',
    'base_llava-v1.5-13b': 'base_llava-v1.5-13b',
    'CodeLlama-7b-Python-hf': 'Llama-2-7b-hf',
    'llava-v1.5-13b': 'llava-v1.5-13b',
    'llava-v1.5-7b': 'llava-v1.5-7b',
    'table-llava-v1.5-13b': 'llava-v1.5-13b',
    'llava-v1.6-vicuna-13b': 'llava-v1.6-vicuna-13b',
    'MuggleMath_13B': "Llama-2-13b-hf",
    'WizardLM-13B-V1.0': 'Llama-2-13b-hf',
    'MetaMath-13B-V1.0': 'Llama-2-13b-hf',
    'Llama-2-7b-chat-hf': "Llama-2-7b-hf",
    "MetaMath-7B-V1.0": "Llama-2-7b-hf",
    'MetaMath-Llemma-7B': "Llama-2-7b-hf",
    'MAmmoTH-13B': 'Llama-2-13b-hf',
    'MAmmoTH-7B': 'Llama-2-7b-hf',
    'tora-7b-v1.0': 'Llama-2-7b-hf',
    'CodeLlama-7b-hf': 'Llama-2-7b-hf',
    'CodeLlama-7b-Instruct-hf': 'Llama-2-7b-hf',
    'tora-code-7b-v1.0': 'Llama-2-7b-hf',
    'vicuna-7b-v1.5': 'Llama-2-7b-hf',
    'tora-13b-v1.0': 'Llama-2-13b-hf',
    'TableLLM-7b': 'Llama-2-7b-hf',
    'TableLLM-13b': 'Llama-2-13b-hf',
    'llemma_7b': 'Llama-2-7b-hf',
    'MathCoder2-CodeLlama-7B': 'Llama-2-7b-hf',
    'llava-next-8b-hf': 'Llama-3-8B-Instruct',

}