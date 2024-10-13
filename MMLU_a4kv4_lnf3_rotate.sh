model=$1
lwc_checkpoint=$2
eager=$3
batch=${4:-4}
# --eager ${eager}
index=0
gpu_num=8
for ((i = 0; i < $gpu_num; i++)); do

  gpu=$((i))
  ((index++))
  (
    CUDA_VISIBLE_DEVICES=$gpu python llm_eval.py --model ${model} \
      --quant_type nf3-nu --bits 3 --group_size 128 --eval_tasks hendrycksTest-* --test_set --num_fewshot 5 --batch_size ${batch} --limit 0.125 --gpu_id $gpu  \
      --apply_R2 True \
      --apply_R3 True \
      --apply_R4 True \
      --lwc_checkpoint ${lwc_checkpoint} \
      --a_bit 4 --k_bit 4 --v_bit 4 \
      --a_quant_type nf --kv_quant_type nf

  ) &
  if (($index % $gpu_num == 0)); then wait; fi
done

