from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import get_param_names_to_merge
from model_merging_methods.task_vector import TaskVector
from collections import defaultdict, OrderedDict
import cop
def merge_matrices_with_space_similarity(vlm_task_vector, matrices, param_name, device='cuda',
                                         similarity_threshold=0.4):
    # Move matrix A to device and compute its SVDZZ
    A = vlm_task_vector[param_name].to(device)
    U_A, S_A, V_A = torch.svd_lowrank(A, q=1024)
    V_A = V_A.T

    # Compute cosine similarities and merge matrices
    def compute_similarity_and_merge(mat, param_name):
        matrix = mat[param_name].to(device)
        U_matrix, S, V_matrix = torch.svd_lowrank(matrix, q=1024)
        V_matrix = V_matrix.T

        cosine_sim = nn.CosineSimilarity(dim=1)(V_A, V_matrix)
        softmax_sim = F.softmax(cosine_sim*32.0)
        sorted_sim, _ = torch.sort(cosine_sim,descending=True)

        if cosine_sim.max() < similarity_threshold:
            return torch.tensor(0.0, device='cpu')

        else:

            matrix = U_matrix @ torch.diag(S) @ V_matrix
            score = torch.sum(S_A) / torch.sum(S)
            Q = softmax_sim.unsqueeze(1)* V_A
            merged_mat = (score) * matrix @ Q.T @ Q

            return merged_mat

    # Merge all matrices in the list
    merged_matrices = [compute_similarity_and_merge(matrix, param_name) for matrix in matrices]
    return sum(merged_matrices).to('cpu')







def mask_input_with_mask_rate(input_tensor: torch.Tensor, mask_rate: float, use_rescale: bool, mask_strategy: str):
    """
    mask the input with mask rate
    :param input_tensor: Tensor, input tensor
    :param mask_rate: float, mask rate
    :param use_rescale: boolean, whether to rescale the input by 1 / (1 - mask_rate)
    :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
    :return:
    else:
    """
    assert 0.0 <= mask_rate <= 1.0, f"wrong range of mask_rate {mask_rate}, should be [0.0, 1.0]!"
    if mask_strategy == "random":
        mask = torch.bernoulli(torch.full_like(input=input_tensor.float(), fill_value=mask_rate)).to(
            input_tensor.device)
        # mask = create_mask(input_tensor=input_tensor, mask_rate=mask_rate)
        masked_input_tensor = input_tensor * (1 - mask)
        assert mask_strategy == "magnitude", f"wrong setting for mask_strategy {mask_strategy}!"
        original_shape = input_tensor.shape
        input_tensor = input_tensor.flatten()
        num_mask_params = int(len(input_tensor) * mask_rate)
        # Tensor, shape (1, ), find the num_mask_params-th smallest magnitude element of all the parameters in the model
        kth_values, _ = input_tensor.abs().kthvalue(k=num_mask_params, dim=0, keepdim=True)
        # Tensor, shape (num_total_params, ), where True is for parameters that we want to perform mask
        mask = input_tensor.abs() <= kth_values
        masked_input_tensor = input_tensor * (~mask)
        masked_input_tensor = masked_input_tensor.reshape(original_shape)

    if use_rescale and mask_rate != 1.0:
        masked_input_tensor = torch.div(input=masked_input_tensor, other=1 - mask_rate)
    return masked_input_tensor


def mask_model_weights(finetuned_model: nn.Module, pretrained_model: nn.Module, exclude_param_names_regex: list, weight_format: str,
                       weight_mask_rate: float, use_weight_rescale: bool, mask_strategy: str):
    """
    mask model weights
    :param finetuned_model: nn.Module, the finetuned model
    :param pretrained_model: nn.Module, the pretrained model
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :param weight_format: str, the format of weights to be masked, can be "finetuned_weight" and "delta_weight"
    :param weight_mask_rate: float, weight mask rate
    :param use_weight_rescale: boolean, whether to rescale the weight by 1 / (1 - weight_mask_rate)
    :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
    :return:
    """
    # get weights that need to be masked
    if weight_format == "finetuned_weight":
        param_dict = {param_name: param_value for param_name, param_value in finetuned_model.named_parameters()}
        # exclude parameter whose name matches element in exclude_param_names_regex
        param_names_to_merge = get_param_names_to_merge(input_param_names=list(param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)
        model_param_dict = {param_name: param_dict[param_name] for param_name in param_names_to_merge}
    else:
        assert weight_format == "delta_weight", f"wrong setting for weight_format {weight_format}!"
        task_vector = TaskVector(pretrained_model=pretrained_model, finetuned_model=finetuned_model, exclude_param_names_regex=exclude_param_names_regex)
        model_param_dict = task_vector.task_vector_param_dict

    with torch.no_grad():
        masked_param_dict = {}
        for param_name, param_value in tqdm(model_param_dict.items()):
            masked_param_dict[param_name] = mask_input_with_mask_rate(input_tensor=param_value, mask_rate=weight_mask_rate,
                                                                      use_rescale=use_weight_rescale, mask_strategy=mask_strategy)

        if weight_format == "delta_weight":
            new_task_vector = TaskVector(task_vector_param_dict=masked_param_dict)
            # combine with parameters of the merged model based on scaling coefficient
            masked_param_dict = new_task_vector.combine_with_pretrained_model(pretrained_model=pretrained_model, scaling_coefficient=1.0)

    return masked_param_dict




def merge_llms_and_vlms_by_task_vector(args, vlm: nn.Module, finetuned_llm: list, exclude_param_names_regex: list):
    """
    mask model weights
    :param finetuned_model: nn.Module, the finetuned model
    :param pretrained_model: nn.Module, the pretrained model
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :param weight_format: str, the format of weights to be masked, can be "finetuned_weight" and "delta_weight"
    :param weight_mask_rate: float, weight mask rate
    :param use_weight_rescale: boolean, whether to rescale the weight by 1 / (1 - weight_mask_rate)
    :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
    :return:
    """
    # get weights that need to be masked

    # pretrained_model_dict = {param_name: param_value for param_name, param_value in vlm.named_parameters()}

    models_to_merge_task_vectors_param_dict = [TaskVector(pretrained_model=vlm,
                                                          finetuned_model=model_to_merge,
                                                          exclude_param_names_regex=exclude_param_names_regex).task_vector_param_dict
                                               for model_to_merge in finetuned_llm]

    param_names_to_merge = get_param_names_to_merge(
        input_param_names=list(models_to_merge_task_vectors_param_dict[0].keys()),
        exclude_param_names_regex=exclude_param_names_regex)

    with torch.no_grad():
        merged_task_vector = {}
        for param_name in tqdm(param_names_to_merge):
            merged_task_vector[param_name] = args.scaling_coefficient * models_to_merge_task_vectors_param_dict[0][
                param_name] + args.scaling_coefficient * models_to_merge_task_vectors_param_dict[1][param_name]

    new_task_vector = TaskVector(task_vector_param_dict=merged_task_vector)

    merged_param_dict = new_task_vector.combine_with_pretrained_model(
        pretrained_model=vlm)

    return merged_param_dict


def merge_llms_and_vlms_by_task_vector_V4(args, vlm: nn.Module,
                                          pretrained_lm: nn.Module,
                                          finetuned_llm: list,
                                          exclude_param_names_regex: list):

    models_to_merge_task_vectors_param_dict = [TaskVector(pretrained_model=pretrained_lm,
                                                          finetuned_model=model_to_merge,
                                                          exclude_param_names_regex=exclude_param_names_regex).task_vector_param_dict
                                               for model_to_merge in finetuned_llm]

    models_to_merge_task_vectors_param_dict.append(TaskVector(pretrained_model=pretrained_lm,
                                                              finetuned_model=vlm,
                                                              exclude_param_names_regex=exclude_param_names_regex).task_vector_param_dict)

    param_names_to_merge = get_param_names_to_merge(
        input_param_names=list(models_to_merge_task_vectors_param_dict[0].keys()),
        exclude_param_names_regex=exclude_param_names_regex)


    with torch.no_grad():
        merged_task_vector = {}
        for param_name in tqdm(param_names_to_merge):
            merged_value = 0
            for model_idx, task_vector in enumerate(models_to_merge_task_vectors_param_dict):
                merged_value += (1 / len(models_to_merge_task_vectors_param_dict)) * task_vector[param_name]
                # if 'self_attn' in param_name or 'mlp' in param_name:
                #     merged_value += (1 / len(models_to_merge_task_vectors_param_dict)) * task_vector[param_name]

            merged_task_vector[param_name] = merged_value

    new_task_vector = TaskVector(task_vector_param_dict=merged_task_vector)

    merged_param_dict = new_task_vector.combine_with_pretrained_model_v6(pretrained_model=vlm,
                                                                         pretrained_llm=pretrained_lm,
                                                                         scaling_coefficient=args.scaling_coefficient)

    return merged_param_dict






def merge_llms_and_vlms_by_projection_svd(args, vlm: nn.Module, pretrained_lm: nn.Module, finetuned_llm: list,
                                          exclude_param_names_regex: list):
    """.
    merge by task vector
    """
    # get weights that need to be masked

    # pretrained_model_dict = {param_name: param_value for param_name, param_value in pretrained_lm.named_parameters()}
    # vlm_model_dict = {param_name: param_value for param_name, param_value in vlm.named_parameters()}
    vlm_task_vector = TaskVector(pretrained_model=pretrained_lm,
                                 finetuned_model=vlm,
                                 exclude_param_names_regex=exclude_param_names_regex).task_vector_param_dict

    models_to_merge_task_vectors_param_dict = [TaskVector(pretrained_model=pretrained_lm,
                                                          finetuned_model=model_to_merge,
                                                          exclude_param_names_regex=exclude_param_names_regex).task_vector_param_dict
                                               for model_to_merge in finetuned_llm]

    param_names_to_merge = get_param_names_to_merge(
        input_param_names=list(models_to_merge_task_vectors_param_dict[0].keys()),
        exclude_param_names_regex=exclude_param_names_regex)

    with torch.no_grad():
        merged_task_vector = {}
        for param_name in tqdm(param_names_to_merge):
            # if 'mlp' in param_name:
            if 'self_attn' in param_name or 'mlp' in param_name:
                merged_task_vector[param_name] = merge_matrices_with_space_similarity(
                    vlm_task_vector=vlm_task_vector,
                    matrices=models_to_merge_task_vectors_param_dict,
                    param_name=param_name)

            else:
                merged_task_vector[param_name] = 0

    new_task_vector = TaskVector(task_vector_param_dict=merged_task_vector)

    merged_param_dict = new_task_vector.combine_with_pretrained_model_v5(pretrained_model=vlm,
                                                                         pretrained_llm=pretrained_lm,
                                                                         scaling_coefficient=args.scaling_coefficient)

    return merged_param_dict



def ties_merging(vlm: nn.Module, pretrained_lm: nn.Module, models_to_merge: list, exclude_param_names_regex: list,
                 param_value_mask_rate: float = 0.8, scaling_coefficient: float = 1.0):
    """
        ties merging method
        :param merged_model: nn.Module, the merged model
        :param models_to_merge: list, individual models that need to be merged
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
        :param scaling_coefficient: float, scaling coefficient to merge the task vectors
        :return:
        """

    def task_vector_param_dict_to_single_vector(task_vector: TaskVector):
        """
            convert parameter dictionary in task vector to a single vector
            :param task_vector: TaskVector, task vector
            :return:
            """
        task_vector_param_dict = copy.deepcopy(task_vector.task_vector_param_dict)
        sorted_task_vector_param_dict = OrderedDict(sorted(task_vector_param_dict.items()))

        # Tensor, shape (num_total_params, )
        return nn.utils.parameters_to_vector([param.flatten() for param in sorted_task_vector_param_dict.values()])

    def single_vector_to_task_vector_param_dict(single_vector: torch.Tensor, task_vector: TaskVector):
        """
            convert a single vector to parameter dictionary in task vector
            :param single_vector: Tensor, single vector that contain all parameters in task_vector.task_vector_param_dict
            :param task_vector: TaskVector, task vector
            :return:
            """
        task_vector_param_dict = copy.deepcopy(task_vector.task_vector_param_dict)
        sorted_task_vector_param_dict = OrderedDict(sorted(task_vector_param_dict.items()))

        nn.utils.vector_to_parameters(single_vector, sorted_task_vector_param_dict.values())

        return sorted_task_vector_param_dict

    def mask_smallest_magnitude_param_values(flattened_models_to_merge_param: torch.Tensor,
                                             param_value_mask_rate: float = 0.8):
        """
            mask the smallest-magnitude parameter values (set to zeros) based on parameter value mask rate
            :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
            :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
            :return:
            """
        # num_models_to_merge, num_total_params = flattened_models_to_merge_param.shape
        num_mask_params = int(flattened_models_to_merge_param.shape[1] * param_value_mask_rate)

        # Tensor, shape (num_models_to_merge, 1), find the num_mask_params-th smallest magnitude element of all the parameters in each individual model
        kth_values, _ = flattened_models_to_merge_param.abs().kthvalue(k=num_mask_params, dim=1, keepdim=True)
        # Tensor, shape (num_models_to_merge, num_total_params), where True is for parameters that we want to preserve
        mask = flattened_models_to_merge_param.abs() >= kth_values

        return flattened_models_to_merge_param * mask

    def get_param_signs(flattened_models_to_merge_param: torch.Tensor):
        """
            get the signs for each parameter in flattened_models_to_merge_param, computed over individual models that need to be merged
            :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
            :return:
            """
        # Tensor, shape (num_total_params, ), the signs of parameters aggregated across individual models that need to be merged
        param_signs = torch.sign(flattened_models_to_merge_param.sum(dim=0))
        # Tensor, shape (, ), a scalar, replace 0 in param_signs to the major sign in param_signs
        majority_sign = torch.sign(param_signs.sum(dim=0))
        param_signs[param_signs == 0] = majority_sign
        return param_signs

    def disjoint_merge(flattened_models_to_merge_param: torch.Tensor, param_signs: torch.Tensor):
        """
            disjoint merge that only keeps the parameter values in individual models whose signs are the same as the param_signs, and calculates the averaged parameters.
            :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
            :param param_signs: Tensor, shape (num_total_params, ), the signs of parameters aggregated across individual models that need to be merged
            :return:
            """
        # Tensor, shape (num_models_to_merge, num_total_params), where True is for parameters that we want to preserve
        param_to_preserve_mask = ((param_signs.unsqueeze(dim=0) > 0) & (flattened_models_to_merge_param > 0)) | (
                    (param_signs.unsqueeze(dim=0) < 0) & (flattened_models_to_merge_param < 0))
        # Tensor, shape (num_models_to_merge, num_total_params), the preserved parameters
        param_to_preserve = flattened_models_to_merge_param * param_to_preserve_mask

        # Tensor, shape (num_total_params, ), the number of models whose parameters can be preserved
        num_models_param_preserved = (param_to_preserve != 0).sum(dim=0).float()
        # Tensor, shape (num_total_params, ), the averaged flattened parameters
        merged_flattened_param = torch.sum(param_to_preserve, dim=0) / torch.clamp(num_models_param_preserved, min=1.0)

        return merged_flattened_param

    assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"

    models_to_merge_task_vectors = [TaskVector(pretrained_model=pretrained_lm, finetuned_model=model_to_merge,
                                               exclude_param_names_regex=exclude_param_names_regex) for model_to_merge
                                    in models_to_merge]

    models_to_merge_task_vectors.append(TaskVector(pretrained_model=pretrained_lm, finetuned_model=vlm,
                                                   exclude_param_names_regex=exclude_param_names_regex))

    flattened_models_to_merge_param = [task_vector_param_dict_to_single_vector(task_vector=task_vector) for task_vector
                                       in models_to_merge_task_vectors]
    # Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
    flattened_models_to_merge_param = torch.vstack(flattened_models_to_merge_param)

    with torch.no_grad():
        # Tensor, shape (num_models_to_merge, num_total_params), mask the smallest-magnitude parameter values using param_value_mask_rate
        flattened_models_to_merge_param = mask_smallest_magnitude_param_values(
            flattened_models_to_merge_param=flattened_models_to_merge_param,
            param_value_mask_rate=param_value_mask_rate)

        # Tensor, shape (num_total_params, ), get the signs for each parameter in flattened_models_to_merge_param
        param_signs = get_param_signs(flattened_models_to_merge_param=flattened_models_to_merge_param)

        # Tensor, shape (num_total_params, ), disjoint merge
        merged_flattened_param = disjoint_merge(flattened_models_to_merge_param=flattened_models_to_merge_param,
                                                param_signs=param_signs)

        # merged parameter dictionary
        merged_task_vector_param_dict = single_vector_to_task_vector_param_dict(single_vector=merged_flattened_param,
                                                                                task_vector=
                                                                                models_to_merge_task_vectors[0])

        merged_task_vector = TaskVector(task_vector_param_dict=merged_task_vector_param_dict)
        # combine with parameters of the merged model based on scaling coefficient
        # merged_params = merged_task_vector.combine_with_pretrained_model(pretrained_model=merged_model, scaling_coefficient=scaling_coefficient)
        merged_params = merged_task_vector.combine_with_pretrained_model_v6(pretrained_model=vlm,
                                                                            pretrained_llm=pretrained_lm,
                                                                            scaling_coefficient=scaling_coefficient)
    return merged_params


def emr_merging(vlm: nn.Module,pretrained_lm: nn.Module, models_to_merge: list, exclude_param_names_regex: list):
    sum_param = {}
    n2p = []
    # merged_model.to('cpu')
    # for model_to_merge in models_to_merge:
    #     model_to_merge.to('cpu')

    task_vectors = [TaskVector(pretrained_model=pretrained_lm,
                               finetuned_model=model_to_merge,
                               exclude_param_names_regex=exclude_param_names_regex) for model_to_merge in models_to_merge]

    task_vectors.append(TaskVector(pretrained_model=pretrained_lm, finetuned_model=vlm,
                                                   exclude_param_names_regex=exclude_param_names_regex))


    for m in range(len(task_vectors)):
        n2p_temp = task_vectors[m].task_vector_param_dict
        n2p.append(n2p_temp)
        for n in n2p_temp:
            if n not in sum_param:
                sum_param[n] = []
            sum_param[n].append(n2p_temp[n])
    sum_param = {k: torch.stack(v, 0).mean(0) for k, v in sum_param.items()}
    sum_param = sum_param  # .to('cpu')
    Vector_unified = {}
    masks = {}
    scales = torch.zeros(len(task_vectors))
    with torch.no_grad():
        for n in sum_param:
            masks[n] = []
            flag = (sum_param[n] > 0) * 2 - 1
            param_max = torch.zeros_like(n2p[0][n])
            for m in range(len(task_vectors)):
                param = task_vectors[m].task_vector_param_dict[n]
                mask = (param * flag) > 0
                masks[n].append(mask)
                param_abs = torch.abs(mask * param)
                param_max = torch.where(param_abs > param_max, param_abs, param_max)
                # scales[m] += torch.mean(torch.abs(param.cpu()))
                # if n not in scales:
                #
                #     scales[n] = []#torch.mean(torch.abs(param[n].cpu()))
                # else:
                scales[m] += torch.mean(torch.abs(param.cpu()))
                # pass
            Vector_unified[n] = param_max * flag

        new_scales = torch.zeros(len(task_vectors))
        for m in range(len(task_vectors)):
            for n in Vector_unified:
                p = Vector_unified[n].cpu() * masks[n][m].cpu()
                new_scales[m] += torch.mean(torch.abs(p))
        rescales = scales / new_scales

    task_vector_recon = {}

    for idx in range(len(task_vectors)):
        for nt in Vector_unified:
            task_vector_recon[nt] = Vector_unified[nt] * masks[nt][idx] * rescales[idx]

    merged_task_vector = TaskVector(task_vector_param_dict=task_vector_recon)
    # combine with parameters of the merged model based on scaling coefficient
    # merged_params = merged_task_vector.combine_with_pretrained_model(pretrained_model=merged_model, scaling_coefficient=scaling_coefficient)
    merged_params = merged_task_vector.combine_with_pretrained_model_v6(pretrained_model=vlm,
                                                                        pretrained_llm=pretrained_lm)
    return merged_params