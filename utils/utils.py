import re
import os
from typing import Dict
import random
import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import Trainer, TrainerState
import sys
import io
import json


def verify_response(response):
    if isinstance(response, str):
        response = response.strip()
    if response == "" or response == None:
        return False
    if "Response Error" in response:
        return False
    return True


def evaluate_code(code_string):
    # execute_code_and_capture_output
    # Backup the original stdout
    old_stdout = sys.stdout

    # Redirect stdout to capture the output
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    # Try executing the code and capture any exception
    error = None
    try:
        exec(code_string)
    except Exception as e:
        error = e

    # Restore the original stdout
    sys.stdout = old_stdout

    # Get the captured output
    captured_output = new_stdout.getvalue()
    if isinstance(captured_output, str):
        captured_output = captured_output.strip()

    # Return the captured output or error
    return captured_output, error
def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def read_pandas_csv(csv_path):
    # read a pandas csv sheet
    import pandas as pd
    df = pd.read_csv(csv_path)
    return df


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def set_random_seed(seed: int = 0):
    """
    set random seed
    :param seed: int, random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_state_and_model_for_hf_trainer(trainer: Trainer):
    """
    save the state and model for trainer
    :param trainer: transformers.Trainer to be saved
    :return:
    """
    # save trainer state at trainer.args.output_dir path
    trainer.save_state()
    # save model at output_dir
    if trainer.args.should_save:
        # convert state_dict to cpu
        cpu_state_dict = {key: value.cpu() for key, value in trainer.model.state_dict().items()}
        trainer._save(trainer.args.output_dir, state_dict=cpu_state_dict)


def load_state_and_model_for_hf_trainer(model: nn.Module, load_model_dir: str, map_location: str = None):
    """
    load the state and model for trainer
    :param model: nn.Module, the model to be loaded
    :param load_model_dir: str, the path where the state and model to be loaded
    :param map_location: str, how to remap the storage locations
    :return:
    """
    # load model and trainer state from load_model_dir
    model.load_state_dict(torch.load(os.path.join(load_model_dir, "pytorch_model.bin"), map_location=map_location))
    # model = model.from_pretrained(load_model_dir)
    trainer_state = TrainerState.load_from_json(os.path.join(load_model_dir, "trainer_state.json"))
    return model, trainer_state


def get_param_names_to_merge(input_param_names: list, exclude_param_names_regex: list):
    """
    get the names of parameters that need to be merged
    :param input_param_names: list, names of input parameters
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :return:
    """
    param_names_to_merge = []
    for param_name in input_param_names:
        exclude = any([re.match(exclude_pattern, param_name) for exclude_pattern in exclude_param_names_regex])
        if not exclude:
            param_names_to_merge.append(param_name)
    return param_names_to_merge


def get_modules_to_merge(model: nn.Module, include_module_types: list):
    """
    get the model modules that need to be merged, whose type is in include_module_types
    :param model: nn.Module, input model
    :param include_module_types: list, module types that want to include
    :return:
    """
    modules_to_merge = {}
    for module_name, module in model.named_modules():
        is_valid_type = not include_module_types or any([isinstance(module, include_module_type) for include_module_type in include_module_types])
        if is_valid_type:
            modules_to_merge[module_name] = module
    return modules_to_merge


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    print(tokenizer.vocab_size)
    assert tokenizer.vocab_size == 32000
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    if num_new_tokens > 0:
        model.resize_token_embeddings(tokenizer.vocab_size + num_new_tokens)

        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


# def smart_tokenizer_and_embedding_resize(
#         special_tokens_dict: Dict,
#         tokenizer: transformers.PreTrainedTokenizer,
#         model: transformers.PreTrainedModel,
# ):
#     """Resize tokenizer and embedding.
#
#     Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
#     """
#     current_vocab_size = tokenizer.vocab_size
#     print(f"Current tokenizer vocab size: {current_vocab_size}")
#
#     # Assert only if you know the vocab size should be exactly 32000
#     # assert current_vocab_size == 32000
#
#     num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
#     print(f"Number of new tokens added: {num_new_tokens}")
#
#     if num_new_tokens > 0:
#         new_vocab_size = tokenizer.vocab_size
#         print(f"New tokenizer vocab size: {new_vocab_size}")
#
#         model.resize_token_embeddings(new_vocab_size)
#
#         input_embeddings = model.get_input_embeddings().weight.data
#         output_embeddings = model.get_output_embeddings().weight.data
#
#         input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
#         output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
#
#         input_embeddings[-num_new_tokens:] = input_embeddings_avg
#         output_embeddings[-num_new_tokens:] = output_embeddings_avg
#
#     print("Tokenizer and embeddings resized successfully.")