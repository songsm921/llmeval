import sys
from lm_eval import evaluator, tasks, utils
from utils_eval import LMEvalAdaptor
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    LlamaTokenizer
)
import torch
import sys

sys.path.append("../")
sys.path.append("../")
sys.path.append("../../quantization")
sys.path.append('/ceph/echoi/codes/BitDistiller')
sys.path.append('/mnt/cephfs/echoi/codes/BitDistiller')
from test_utils import pseudo_quantize_model_weight
from qlinear import QLinear, convertModelToQuant
import quarot.rotation_utils as rotation_utils
import quarot.model_utils as model_utils
import quarot.eval_utils as eval_utils

from test_utils import pseudo_quantize_model_weight

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    def str2bool(v):
        #susendberg's function
        return v.lower() in ("yes", "true", "t", "1")
    parser.add_argument(
        '--model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument('--eval_tasks', type=str, help='evaluation tasks') # hendrycksTest-*; arc_challenge,winogrande,hellaswag,piqa
    parser.add_argument('--test_set', action="store_true", help='evaluation tasks')
    parser.add_argument('--batch_size', type=int, default=2, help='evaluation tasks')
    parser.add_argument('--bits', type=int, default=2, help='evaluation tasks')
    parser.add_argument('--group_size', type=int, default=128, help='evaluation tasks')
    parser.add_argument('--quant_type', type=str, default="int", help='evaluation tasks')
    parser.add_argument('--num_fewshot', type=int, default=0, help='evaluation tasks')
    parser.add_argument('--apply_R2', type=str2bool, default=False)
    parser.add_argument('--R2_type', type=str, default='hadamard')
    parser.add_argument('--apply_R3', type=str2bool, default=False)
    parser.add_argument('--apply_R4', type=str2bool, default=False)
    
    parser.add_argument('--lwc_checkpoint', type=str, default=None)
    
    parser.add_argument('--nu_act', type=str, default='sigmoid')
    parser.add_argument('--nu_dequant', type=str, default='uniform')
    parser.add_argument('--a_quant_type', type=str, default="int")
    parser.add_argument('--a_bit', type=int, default=16)
    parser.add_argument('--a_group', type=int, default=-1)
    parser.add_argument('--a_asym', type=str2bool, default=False)#action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--a_clip', type=float, default=0.9)
    parser.add_argument('--kv_quant_type', type=str, default="int")
    parser.add_argument('--k_bit', type=int, default=16)
    parser.add_argument('--k_group', type=int, default=128)
    parser.add_argument('--k_asym', type=str2bool, default=True)
    parser.add_argument('--k_clip', type=float, default=0.95)
    parser.add_argument('--v_bit', type=int, default=16)
    parser.add_argument('--v_group', type=int, default=128)
    parser.add_argument('--v_asym', type=str2bool, default=True)
    parser.add_argument('--v_clip', type=float, default=0.95)
    parser.add_argument('--limit', type=float, default=None)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--eager', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--nf_dtype', type=str, default='float32')
    args = parser.parse_args()
    print(args)
    print(args.batch_size)
    if "hendrycksTest" not in args.eval_tasks:
        args.test_set = True
    
    if args.eager or '3.2' in args.model:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.model)
        config._attn_implementation = 'eager'
        config.tie_word_embeddings = False
        print('config tie word embeddings', config.tie_word_embeddings)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, use_safetensors=True, low_cpu_mem_usage=True, config = config)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, 
                                                torch_dtype=torch.bfloat16, 
                                                use_safetensors=True,
                                                device_map='auto'
                                                )
    
    if args.lwc_checkpoint is not None:
        nf_dtype = torch.float32 if args.nf_dtype == 'float32' else torch.bfloat16
        if 'opt' in args.lwc_checkpoint or 'r1' in args.lwc_checkpoint or args.apply_R4: # for checkpoint name not r1234...
            rotation_utils.fuse_layer_norms(model)
        model, _ = convertModelToQuant(
                model,
                compute_dtype=torch.bfloat16,
                quant_type=args.quant_type,
                q_group_size=args.group_size,
                nu_act=args.nu_act,
                nu_dequant=args.nu_dequant,
                a_quant_type=args.a_quant_type,
                a_bit=args.a_bit,
                a_asym=args.a_asym,
                a_group=args.a_group,
                a_clip=args.a_clip,
                kv_quant_type=args.kv_quant_type,
                k_bit=args.k_bit,
                k_asym=args.k_asym,
                k_group=args.k_group,
                k_clip=args.k_clip,
                v_bit=args.v_bit,
                v_asym=args.v_asym,
                v_group=args.v_group,
                v_clip=args.v_clip,
                configure_now=True,
                apply_R2=args.apply_R2,
                apply_R4=args.apply_R4,
                num_heads=model.config.num_attention_heads,
                R2_type=args.R2_type,
                intermediate_size=model.config.intermediate_size,
                Q=None,
                nf_dtype=nf_dtype)# SHOULD IMPLEMENT R4 SYLVESTER WALSH WHEN WE DECIDED TO USE IT
        lwc_ckpt = torch.load(args.lwc_checkpoint, map_location='cpu')['module']
        for n, p in model.named_parameters():
            print(f'Loading {n}')
            p.data.copy_(lwc_ckpt[n].data)

        if args.apply_R3:
            from quarot.model_utils import get_layers, get_rope_function_name
            from quarot.rotation_utils import add_qk_rotation_wrapper_after_function_call_in_forward
                
            rope_function_name = get_rope_function_name(model)
            layers = get_layers(model)
            k_quant_config = {'k_bits': 16, "k_groupsize": -1,
                                                "k_sym": False, "k_clip_ratio": 0.95}
                
            for layer in layers:
                add_qk_rotation_wrapper_after_function_call_in_forward(
                    layer.self_attn,
                    rope_function_name,
                    config=model.config,
                    **k_quant_config
                )   
    if args.bits < 16 and args.lwc_checkpoint is None:
        q_config = {
            "zero_point": True,  # by default True
            "q_group_size": args.group_size,  # whether to use group quantization
        }
        pseudo_quantize_model_weight(
            model, w_bit=args.bits, q_config=q_config, quant_type=args.quant_type
        )
    print(model)
    # model = model.to(dtype=torch.bfloat16)
    model = model.to('cuda')

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model.eval()

    task_names = utils.pattern_match(args.eval_tasks.split(","), tasks.ALL_TASKS)

    lm_eval_model = LMEvalAdaptor(args.model, model, tokenizer, args.batch_size)
    
    results = evaluator.simple_evaluate(
                    model=lm_eval_model,
                    tasks=task_names,
                    batch_size=args.batch_size,
                    no_cache=True,
                    num_fewshot=args.num_fewshot,
                    test_set=args.test_set,
                    limit = args.limit if args.limit is not None else None,
                    gpu_id = args.gpu_id if args.limit is not None else None
                )
    print(results)
    # 初始化总 acc 和计数器
    acc_sum = 0
    count = 0
    # 遍历所有 hendrycksTest 相关的数据
    if "hendrycksTest" in args.eval_tasks:
        for key in results['results']:
            if 'hendrycksTest' in key:
                # 累加 acc 值并增加计数器
                acc_sum += results['results'][key]['acc']
                count += 1

        # 计算平均值
        if count > 0:
            avg_acc = acc_sum / count

            # print("mmlu-acc:", avg_acc)
            mmlu_results = {}
            mmlu_results['mmlu-acc'] = avg_acc
            isRotate = 'Y' if args.apply_R2 else 'N'
            mode = str(args.quant_type) + "_" + str(args.a_bit) + '-' + str(args.k_bit) + '-' + isRotate 
            model_name = None
            if '3.2' in args.model:
                model_name = 'llama-3.2'
            else:
                model_name = 'llama1'
            model_size = '7b'
            if '3b' in args.model:
                model_size = '3b'
            elif '1b' in args.model:
                model_size = '1b'
            dir_name = 'MMLU' +'-' + model_name + model_size
            # make dir
            import os
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            file_name = 'MMLU'+ model_size +'-'+ mode + '-#' + str(args.gpu_id) + '.txt'
            with open(os.path.join(dir_name, file_name), 'a') as f:
                f.write(str(args.model))
                f.write('\n')
                f.write(str(mmlu_results))
                f.write('\n')
            print(mmlu_results)
    else:
        for key in results['results']:
            acc_sum += results['results'][key]['acc']
            count += 1
        # 计算平均值
        if count > 0:
            avg_acc = acc_sum / count
            print("QA Avg:", avg_acc)
            file_name = str(args.eval_tasks) + "_results.txt"
            with open(file_name, 'a') as f:
                isRotate = 'Y' if args.apply_R2 else 'N'
                mode = str(args.quant_type) + "_" + str(args.a_bit) + '-' + str(args.k_bit) + '-' + isRotate 
                f.write(mode + '\n')
                f.write(str(avg_acc))
                f.write('\n')
            