import argparse
import gc
import logging
import os.path
import torch
from model_merging_methods.mask_weights_utils import merge_llms_and_vlms_by_task_vector_V4, \
     merge_llms_and_vlms_by_projection_svd, ties_merging, emr_merging
from utils.utils import set_random_seed
from utils.load_config import cache_dir, finetuned_model_backbone_mapping_dict
from utils.extract_ans import *
from utils.prompt_template import *
import torch.multiprocessing as mp
from evaluation_math_reasoning.evaluate_math_vision import test_mathvision, evaluate, math_level_subject_acc
from evaluation_math_reasoning.evaluate_mathvista import test_mathvista
from evaluation_math_reasoning.evaluate_mmmu import test_mmmu, eval_mmmu_results
from Llava.model.builder import load_pretrained_model
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

finetuned_llm_mapping_dict = {
    "math": "tora-7b-v1.0",
    "code": "tora-code-7b-v1.0",
    'table': 'MetaMath-7B-V1.0'
}


# MathCoder2-CodeLlama-7B
# MetaMath-13B-V1.0
#"tora-13b-v1.0",
#MetaMath-Llemma-7B
def create_merged_mllm(args, pretrained_vlm_name, logger: logging.Logger, save_model_path=None):
    ## set random seed
    set_random_seed(seed=0)

    ## load pretrained vlm
    pretrained_tokenizer, pretrained_model, pretrained_vis_processors, pretrained_context_len = load_pretrained_model(
        model_path=os.path.join(cache_dir, pretrained_vlm_name),
        model_base=None,
        model_name=pretrained_vlm_name, device="cpu")

    ## select & load model to merge
    finetuned_llm_model_names = []
    merge_task_names = []

    for merge_flag, task_name in zip([args.merge_math, args.merge_code, args.merge_table],
                                     ["math", "code", "table"]):
        if merge_flag:
            finetuned_llm_model_names.append(finetuned_llm_mapping_dict[task_name])
            merge_task_names.append(task_name)

    llm_models_to_merge = []
    finetuned_tokenizers = []

    for finetuned_model_name in finetuned_llm_model_names:
        finetuned_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=os.path.join(cache_dir, finetuned_model_name), low_cpu_mem_usage=True,
            device_map="cpu")
        finetuned_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(cache_dir, finetuned_model_name), )
        llm_models_to_merge.append(finetuned_model)
        finetuned_tokenizers.append(finetuned_tokenizer)



    pretrained_base_llm_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=os.path.join(cache_dir,
                                                   finetuned_model_backbone_mapping_dict[finetuned_llm_model_names[0]]),
        low_cpu_mem_usage=True,
        device_map="cpu")

    if args.merge_method == 'task_vector':
        # merged_param_dict = merge_llms_and_vlms_by_task_vector(args, vlm=pretrained_model,
        #                                                        finetuned_llm=llm_models_to_merge,
        #                                                          exclude_param_names_regex=['model.embed_tokens.weight',
        #                                                                                   'lm_head.weight'])

        merged_param_dict = merge_llms_and_vlms_by_task_vector_V4(args, vlm=pretrained_model,
                                                                  pretrained_lm=pretrained_base_llm_model,
                                                                  finetuned_llm=llm_models_to_merge,
                                                                  exclude_param_names_regex=[
                                                                      'model.embed_tokens.weight',
                                                                      'lm_head.weight'])


        torch.cuda.empty_cache()

    elif args.merge_method == 'svd':
        print("You are using SVD Merging to merge", pretrained_vlm_name, 'with', finetuned_llm_model_names, "!!!!")
        merged_param_dict = merge_llms_and_vlms_by_projection_svd(args, vlm=pretrained_model,
                                                                  pretrained_lm=pretrained_base_llm_model,
                                                                  finetuned_llm=llm_models_to_merge,
                                                                  exclude_param_names_regex=[
                                                                      'model.embed_tokens.weight',
                                                                      'lm_head.weight'])

        torch.cuda.empty_cache()


    elif args.merge_method == 'emr_merging':
        print("You are using emr_merging to merge", pretrained_vlm_name, 'with', finetuned_llm_model_names, "!!!!")

        merged_param_dict = emr_merging(vlm=pretrained_model,
                                        pretrained_lm=pretrained_base_llm_model,
                                        models_to_merge=llm_models_to_merge,
                                        exclude_param_names_regex=['model.embed_tokens.weight', 'lm_head.weight'], )

        torch.cuda.empty_cache()


    elif args.merge_method == 'ties_merging':
        print("You are using ties_merging to merge", pretrained_vlm_name, 'with', finetuned_llm_model_names, "!!!!")

        merged_param_dict = ties_merging(vlm=pretrained_model,
                                         pretrained_lm=pretrained_base_llm_model,
                                         models_to_merge=llm_models_to_merge,
                                         exclude_param_names_regex=['model.embed_tokens.weight', 'lm_head.weight'], )

        torch.cuda.empty_cache()


    else:
        merged_param_dict = None
        assert ('No LLM Model to Merge.')

    for param_name, param_value in pretrained_model.named_parameters():
        if param_name in merged_param_dict:
            param_value.data.copy_(merged_param_dict[param_name])

    logger.info(f"saving model at {save_model_path}...")
    os.makedirs(save_model_path, exist_ok=True)
    pretrained_model.save_pretrained(save_directory=save_model_path)
    pretrained_tokenizer.save_pretrained(save_directory=save_model_path)
    logger.info(f"model is saved")

    tokenizer, merged_model, vis_processors, context_len = load_pretrained_model(
        model_path=save_model_path,
        model_base=None,
        model_name=pretrained_vlm_name)

    return merged_model, tokenizer, vis_processors, context_len


if __name__ == '__main__':
    # os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser("Interface for inference LLMs")
    parser.add_argument("--finetuned_model_name", type=str, default="Math-LLaVA",
                        help="name of the finetuned language model",
                        choices=["WizardLM-7B-V1.0", "WizardLM-13B-V1.2", "WizardLM-70B-V1.0",
                                 "WizardMath-7B-V1.0", "WizardMath-13B-V1.0", "WizardMath-70B-V1.0",
                                 "WizardCoder-Python-7B-V1.0", "WizardCoder-Python-13B-V1.0", 'G-LLaVA-7B',
                                 'G-LLaVA-13B',"WizardCoder-Python-34B-V1.0", "Llama-2-7b-hf", "deepseek-math-7b-base",
                                 "deepseek-math-7b-rl", 'Math-LLaVA', 'base_llava-v1.5-13b', 'CodeLlama-7b-Python-hf',
                                 'llava-v1.6-vicuna-7b','llava-v1.5-13b', 'llava-v1.6-vicuna-13b', 'llava-v1.5-7b', 'table-llava-v1.5-13b',
                                 'table-llava-v1.5-7b', 'TableLLM-7b', 'llava-next-8b-hf',
                                 "llama-2-13b-code-alpaca"])

    parser.add_argument("--llm_to_add", type=str, default=None,
                        help="name of the finetuned language model")

    parser.add_argument("--dataset_name", type=str, default="gsm8k", help="dataset to be used",
                        choices=["alpaca_eval", "gsm8k", "MATH", "human_eval", "mbpp", "math_check", 'MMMU', "MMTab",
                                 'MathVista', "MathVision"])
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=sys.maxsize)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--weight_format", type=str, help="the format of weights to be masked", default="delta_weight",
                        choices=["finetuned_weight", "delta_weight"])
    parser.add_argument("--weight_mask_rate", type=float, default=0.0, help="weight mask rate")
    parser.add_argument("--use_weight_rescale", action="store_true", default=False,
                        help="whether to rescale the weight by 1 / (1 - weight_mask_rate)")
    parser.add_argument("--mask_strategy", type=str, help="mask strategy", default="random",
                        choices=["random", "magnitude"])
    parser.add_argument("--wizardcoder_use_llama2_as_backbone", action="store_true", default=False,
                        help="whether to use llama-2 as the backbone for WizardCoder")

    parser.add_argument('--output_path', type=str, default='mathvista_add_muggle.json',
                        help='name of saved json')
    parser.add_argument("--scaling_coefficient", type=float, default=0.5)

    parser.add_argument('--query_file', type=str, default='query.json')
    parser.add_argument('--caption_file', type=str, default=None)
    parser.add_argument('--ocr_file', type=str, default=None)
    parser.add_argument('--shot_type', type=str, default='solution', help='shot type',
                        choices=['solution', 'code'])
    parser.add_argument('--shot_num', type=int, default=0, help='number of shot examples')
    parser.add_argument('--use_caption', default=False, help='use caption data')
    parser.add_argument('--use_ocr', default=False, help='use ocr data')
    # other settings
    parser.add_argument('--rerun', default=False, help='rerun answer extraction for all problems')
    parser.add_argument('--debug', default=False, help='debug mode')
    parser.add_argument("--merge_instruct", action="store_true", default=False, help="whether to merge instruct model")
    parser.add_argument("--merge_math", action="store_true", default=False, help="whether to merge math model")
    parser.add_argument("--merge_code", action="store_true", default=False, help="whether to merge code model")
    parser.add_argument("--merge_table", action="store_true", default=False, help="whether to merge table model")
    parser.add_argument("--merge_method", type=str, help="the format of weights to be masked", default="task_vector",
                        choices=["task_vector", "mario", 'svd', "ties_merging", "emr_merging"])
    parser.add_argument("--merge_mode", type=str, default="online",
                        choices=["online", "offline", "load_finetuned_model"])

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit()

    if args.weight_mask_rate == 0.0:
        save_model_name = f"{args.finetuned_model_name}_inference_mask_{args.weight_mask_rate}"
        save_model_path = None
        just_inference = True
    else:
        save_model_name = f"{args.finetuned_model_name}_inference_mask_{args.weight_mask_rate}_rescale_{args.use_weight_rescale}"
        if args.mask_strategy == "magnitude":
            save_model_name = f"{save_model_name}_strategy_{args.mask_strategy}"
        if args.weight_format == "finetuned_weight":
            save_model_name = f"{save_model_name}_weight_format_{args.weight_format}"
        if args.wizardcoder_use_llama2_as_backbone:
            assert args.finetuned_model_name in ["WizardCoder-Python-7B-V1.0", "WizardCoder-Python-13B-V1.0"]
            if args.finetuned_model_name == "WizardCoder-Python-7B-V1.0":
                finetuned_model_backbone_mapping_dict["WizardCoder-Python-7B-V1.0"] = "Llama-2-7b-hf"
            else:
                finetuned_model_backbone_mapping_dict["WizardCoder-Python-13B-V1.0"] = "Llama-2-13b-hf"
            save_model_name = f"{save_model_name}_llama_2_as_backbone"
        save_model_path = f"./save_models/{args.dataset_name}/{save_model_name}"
        just_inference = False
    if args.dataset_name == "alpaca_eval":
        save_gen_results_folder = f"./save_gen_instruct_responses_results/{args.dataset_name}/{save_model_name}"
    elif args.dataset_name in ["human_eval", "mbpp"]:
        save_gen_results_folder = f"./save_gen_codes_results/{args.dataset_name}/{save_model_name}"
    else:
        save_gen_results_folder = None

    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    os.makedirs(f"./save_logs/{args.dataset_name}", exist_ok=True)
    os.makedirs(f"./save_logs/{args.dataset_name}/{save_model_name}", exist_ok=True)


    os.makedirs(f"./saved_results", exist_ok=True)
    os.makedirs(f"./saved_results/{args.dataset_name}", exist_ok=True)
    # os.makedirs(f"./saved_results/{args.dataset_name}/{args.merge_method}", exist_ok=True)

    if args.merge_mode == 'load_finetuned_model':
        os.makedirs(f"./saved_results/{args.dataset_name}/load_finetuned_model", exist_ok=True)
        # args.output_path = os.path.join(f"./saved_results/{args.dataset_name}/load_finetuned_model/", output_file_name)
    else:

        os.makedirs(f"./saved_results/{args.dataset_name}/{args.merge_method}", exist_ok=True)


    fh = logging.FileHandler(f"./save_logs/{args.dataset_name}/{save_model_name}/{str(time.time())}.log")
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    run_start_time = time.time()
    logger.info(f"********** Run starts. **********")
    logger.info(f"configuration is {args}")

    merge_task_names = []  # Initialize an empty list to store task names
    merge_llm_names = []
    for merge_flag, task_name in zip([args.merge_instruct, args.merge_math, args.merge_code, args.merge_table],
                                     ["instruct", "math", "code", "table"]):
        if merge_flag:
            merge_task_names.append(task_name)  # Append the task name if the flag is True
            merge_llm_names.append(finetuned_llm_mapping_dict[task_name])
    # Join the list of task names with underscores and format the output file name
    output_file_name = f"{args.dataset_name}_{args.finetuned_model_name}_{args.merge_method}_{'_'.join(merge_task_names)}_{'_'.join(merge_llm_names)}_{args.merge_mode}_{args.scaling_coefficient}.json"

    sim_file_name = f"Space_Sim_Score_{args.finetuned_model_name}_{args.merge_method}_{'_'.join(merge_task_names)}_{'_'.join(merge_llm_names)}.txt"

    if args.merge_mode == 'load_finetuned_model':

        args.output_path = os.path.join(f"./saved_results/{args.dataset_name}/load_finetuned_model/", output_file_name)
    else:

        args.output_path = os.path.join(f"./saved_results/{args.dataset_name}/{args.merge_method}/", output_file_name)

    args.param_sim_path = os.path.join(f"./saved_results/{args.dataset_name}/{args.merge_method}/", sim_file_name)

    if args.merge_mode == 'online':

        model, tokenizer, vis_processors, context_len = create_merged_mllm(args,
                                                                           args.finetuned_model_name,
                                                                           logger=logger,
                                                                           save_model_path=save_model_path)

    else:
        tokenizer, model, vis_processors, context_len = load_pretrained_model(
            model_path=os.path.join(cache_dir, args.finetuned_model_name),
            model_base=None,
            model_name=args.finetuned_model_name, device="cuda")

    if args.dataset_name == "MMMU":
        args.test_data_path = "Your data path"

        test_mmmu(args, tokenizer, model, vis_processors, test_data_path=args.test_data_path)
        eval_mmmu_results(output_path=args.output_path,
                          answer_path=os.path.join(args.test_data_path, 'answer_dict_val.json'))


    elif args.dataset_name == "MathVista":
        args.test_data_path = "Your data path"

        args.input_file = 'testmini.json'

        test_mathvista(tokenizer,
                       model,
                       vis_processors,
                       context_len,
                       test_data_path=args.test_data_path,
                       args=args,
                       logger=logger,
                       start_index=args.start_index,
                       end_index=args.end_index,
                       save_model_path=save_model_path)



    logger.info(f"inference of {args.finetuned_model_name} is completed")
    sys.exit()
