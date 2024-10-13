bash MMLU_a4kv4_int2nu_rotate.sh /mnt/cephfs/echoi/codes/BitDistiller/train/ckpts/hf-llama-3.2-3b/int2-a4kv4-ours-rev/checkpoint-1600 /mnt/cephfs/echoi/codes/BitDistiller/train/ckpts/hf-llama-3.2-3b/int2-a4kv4-ours-rev/checkpoint-1600/global_step1600/mp_rank_00_model_states.pt True 
echo "MMLU_a4kv4_int2nu_rotate.sh done"
bash MMLU_a4_int2nu_rotate.sh /mnt/cephfs/echoi/codes/BitDistiller/train/ckpts/hf-llama-3.2-3b/int2-a4-ours-rev/checkpoint-1600/ /mnt/cephfs/echoi/codes/BitDistiller/train/ckpts/hf-llama-3.2-3b/int2-a4-ours-rev/checkpoint-1600/global_step1600/mp_rank_00_model_states.pt True 
echo "MMLU_a4_int2nu_rotate.sh done"
bash MMLU_a4kv4_nf3_rotate.sh /mnt/cephfs/echoi/codes/BitDistiller/train/ckpts/hf-llama-3.2-3b/nf3-all-a4kv4-r1234/checkpoint-1600 /mnt/cephfs/echoi/codes/BitDistiller/train/ckpts/hf-llama-3.2-3b/nf3-all-a4kv4-r1234/checkpoint-1600/global_step1600/mp_rank_00_model_states.pt True
echo "MMLU_a4kv4_nf3_rotate.sh done"
# bash MMLU_a4_nf3_rotate.sh /mnt/nas3/0.Personal/sumin/ACL/hf-llama-3.2-3b/nf3-all-r1234-a4/checkpoint-1600/ /mnt/nas3/0.Personal/sumin/ACL/hf-llama-3.2-3b/nf3-all-r1234-a4/checkpoint-1600/global_step1600/mp_rank_00_model_states.pt True 
# echo "MMLU_a4_nf3_rotate.sh done"
#Above Llama-3.2-3b
#Below Llama-3.2-1b
bash MMLU_a4kv4_int2nu_rotate.sh /mnt/nas3/0.Personal/sumin/ACL/hf-llama-3.2-1b/int2-ours-a4kv4/checkpoint-3200 /mnt/nas3/0.Personal/sumin/ACL/hf-llama-3.2-1b/int2-ours-a4kv4/checkpoint-3200/global_step3200/mp_rank_00_model_states.pt True 
echo "MMLU_a4kv4_int2nu_rotate.sh done"
bash MMLU_a4kv4_int2_rotate.sh /mnt/nas3/0.Personal/sumin/ACL/hf-llama-3.2-1b/int2-r1234-a4kv4/checkpoint-3200/ /mnt/nas3/0.Personal/sumin/ACL/hf-llama-3.2-1b/int2-r1234-a4kv4/checkpoint-3200/global_step3200/mp_rank_00_model_states.pt True 
echo "MMLU_a4kv4_int2_rotate.sh done"
bash MMLU_a4kv4_int2.sh /mnt/nas3/0.Personal/sumin/ACL/hf-llama-3.2-1b/int2-g128-a4kv4/checkpoint-1600/ /mnt/nas3/0.Personal/sumin/ACL/hf-llama-3.2-1b/int2-g128-a4kv4/checkpoint-1600/global_step1600/mp_rank_00_model_states.pt True 
echo "MMLU_a4kv4_int2.sh done"
bash MMLU_a4kv4_nf3.sh /mnt/nas3/0.Personal/sumin/ACL/hf-llama-3.2-1b/nf3-all-vanila-a4kv4/checkpoint-1600/ /mnt/nas3/0.Personal/sumin/ACL/hf-llama-3.2-1b/nf3-all-vanila-a4kv4/checkpoint-1600/global_step1600/mp_rank_00_model_states.pt True 
echo "MMLU_a4kv4_nf3.sh done"
bash MMLU_a4kv4_nf3_rotate.sh /mnt/cephfs/echoi/codes/BitDistiller/train/ckpts/hf-llama-3.2-1b/nf3-g128-r1234-a4kv4-all-zero/checkpoint-1600/ /mnt/cephfs/echoi/codes/BitDistiller/train/ckpts/hf-llama-3.2-1b/nf3-g128-r1234-a4kv4-all-zero/checkpoint-1600/global_step1600/mp_rank_00_model_states.pt True 
echo "MMLU_a4kv4_nf3_rotate.sh done"
bash MMLU_a4kv4_lnf3_rotate.sh /mnt/nas3/0.Personal/sumin/ACL/hf-llama-3.2-1b/nf3-all-ours-a4kv4-zero/checkpoint-1600/ /mnt/nas3/0.Personal/sumin/ACL/hf-llama-3.2-1b/nf3-all-ours-a4kv4-zero/checkpoint-1600/global_step1600/mp_rank_00_model_states.pt True 
echo "MMLU_a4kv4_lnf3_rotate.sh done"
bash MMLU_a4_int2nu_rotate.sh /mnt/nas3/0.Personal/sumin/ACL/hf-llama-3.2-1b/int2-ours-a4/checkpoint-3200/ /mnt/nas3/0.Personal/sumin/ACL/hf-llama-3.2-1b/int2-ours-a4/checkpoint-3200/global_step3200/mp_rank_00_model_states.pt True 
echo "MMLU_a4_int2nu_rotate.sh done"
bash MMLU_a4_int2_rotate.sh /mnt/nas3/0.Personal/sumin/ACL/hf-llama-3.2-1b/int2-g128-r1234-a4/checkpoint-3200/ /mnt/nas3/0.Personal/sumin/ACL/hf-llama-3.2-1b/int2-g128-r1234-a4/checkpoint-3200/global_step3200/mp_rank_00_model_states.pt True 
echo "MMLU_a4_int2_rotate.sh done"
bash MMLU_a4_int2.sh /mnt/nas3/0.Personal/sumin/ACL/hf-llama-3.2-1b/int2-g128-a4/checkpoint-1600/ /mnt/nas3/0.Personal/sumin/ACL/hf-llama-3.2-1b/int2-g128-a4/checkpoint-1600/global_step1600/mp_rank_00_model_states.pt True 
echo "MMLU_a4_int2.sh done"
bash MMLU_a4_nf3.sh /mnt/nas3/0.Personal/sumin/ACL/hf-llama-3.2-1b/nf3-all-vanila-a4/checkpoint-1600/ /mnt/nas3/0.Personal/sumin/ACL/hf-llama-3.2-1b/nf3-all-vanila-a4/checkpoint-1600/global_step1600/mp_rank_00_model_states.pt True 
echo "MMLU_a4_nf3.sh done"
# bash MMLU_a4_nf3_rotate.sh /mnt/nas3/0.Personal/sumin/ACL/hf-llama-3.2-3b/nf3-all-r1234-a4/checkpoint-1600/ /mnt/nas3/0.Personal/sumin/ACL/hf-llama-3.2-3b/nf3-all-r1234-a4/checkpoint-1600/global_step1600/mp_rank_00_model_states.pt True 
# echo "MMLU_a4_nf3_rotate.sh done"
bash MMLU_a4_lnf3_rotate.sh /mnt/cephfs/echoi/codes/BitDistiller/train/ckpts/hf-llama-3.2-1b/nf3-all-ours-a4-zero-8e-7/checkpoint-1600/ /mnt/cephfs/echoi/codes/BitDistiller/train/ckpts/hf-llama-3.2-1b/nf3-all-ours-a4-zero-8e-7/checkpoint-1600/global_step1600/mp_rank_00_model_states.pt True 
echo "MMLU_a4_lnf3_rotate.sh done"



