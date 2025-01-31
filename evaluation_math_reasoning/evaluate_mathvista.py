
import logging

import torch

from utils.utils import set_random_seed, smart_tokenizer_and_embedding_resize, read_json, disable_torch_init, \
    verify_response, evaluate_code
from utils.load_config import cache_dir
from utils.extract_ans import *
from utils.prompt_template import *
from utils.build_query import create_query_data

from llava_utils.data_utils import load_yaml, construct_prompt, save_json, process_single_sample, CAT_SHORT2LONG

from Llava.eval.run_llava import evalmodel
from Llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria



def test_mathvista(tokenizer, model, vis_processors, context_len, test_data_path, args, logger: logging.Logger,
                   start_index=0, end_index=sys.maxsize,
                   save_model_path=None):
    set_random_seed(0)
    input_file = os.path.join(test_data_path, args.input_file)
    print(f"Reading {input_file}...")
    data = read_json(input_file)

    if args.query_file:
        query_file = os.path.join(test_data_path, args.query_file)
        if os.path.exists(query_file):
            print(f"Loading existing {query_file}...")
            query_data = read_json(query_file)
    else:
        print("\nCreating new query...")
        # load caption
        caption_data = {}
        if args.use_caption:
            caption_file = args.caption_file
            if os.path.exists(caption_file):
                print(f"Reading {caption_file}...")
                try:
                    caption_data = read_json(caption_file)["texts"]
                    print("Caption data loaded.")
                except:
                    print("Caption data not found!! Please Check.")
                    # load ocr
        ocr_data = {}
        if args.use_ocr:
            ocr_file = args.ocr_file
            if os.path.exists(ocr_file):
                print(f"Reading {ocr_file}...")
                try:
                    ocr_data = read_json(ocr_file)["texts"]
                    print("OCR data loaded.")
                except:
                    print("OCR data not found!! Please Check.")
        # create query
        query_data = create_query_data(data, caption_data, ocr_data, args)

    if os.path.exists(args.output_path):
        print("\nResults already exist.")
        print(f"Reading {args.output_path}...")
        results = read_json(args.output_path)
    else:
        results = {}

    disable_torch_init()

    if save_model_path == None:
        model_path = os.path.join(cache_dir, args.finetuned_model_name)
    else:
        model_path = save_model_path
    model_name = get_model_name_from_path(model_path)
    ##

    # build final test pid list
    test_pids = list(data.keys())
    print("\nNumber of test problems in total:", len(test_pids))

    skip_pids = []
    if not args.rerun:
        print("\nRemoving problems with existing valid response...")
        for pid in test_pids:
            # print(f"Checking {pid}...")
            if pid in results and 'response' in results[pid]:
                response = results[pid]['response']
                if verify_response(response):
                    # print(f"Valid response found for {pid}.")
                    skip_pids.append(pid)
    else:
        print("\nRerun answer extraction for all problems...")

    test_pids = [pid for pid in test_pids if pid not in skip_pids]
    print("Number of test problems to run:", len(test_pids))

    for _, pid in enumerate(tqdm(test_pids)):
        problem = data[pid]
        query = query_data[pid]
        image = problem['image']
        image_path = os.path.join(test_data_path, image)

        if args.debug:
            print("--------------------------------------------------------------")
        print(f"\nGenerating response for {pid}...")
        # tqdm.write(f"Generating response for {pid}...")
        try:

            args_llava = type('Args', (), {
                "model_path": model_path,
                "model_base": None,
                "model_name": get_model_name_from_path(model_path),
                "query": query,
                "conv_mode": None,
                "image_file": image_path,
                "sep": ",",
                "temperature": 0.2,
                "top_p": None,
                "num_beams": 1,
                "max_new_tokens": 512
            })()
            response = evalmodel(args_llava, model_name, tokenizer, model, vis_processors, context_len)
            results[pid] = problem
            results[pid]['query'] = query
            if args.shot_type == 'solution':
                results[pid]['response'] = response
            else:
                output, error = evaluate_code(response)
                results[pid]['response'] = response
                results[pid]['execution'] = output
                results[pid]['error'] = str(error)
            if args.debug:
                print(f"\n#Query: \n{query}")
                print(f"\n#Response: \n{response}")
        except Exception as e:
            print(e)
            print(f"Error in extracting answer for {pid}")
            results[pid]['error'] = e

        try:
            print(f"Saving results to {args.output_path}...")
            # tqdm.write(f"Saving results to {args.output_path}...")
            save_json(args.output_path, results)
            print(f"Results saved.")
        except Exception as e:
            print(e)
            print(f"Error in saving {args.output_path}")

    del model
    torch.cuda.empty_cache()