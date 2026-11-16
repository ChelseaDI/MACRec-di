# MACRec Instruments 对比学习修正后复现运行指令（权重 1.0）

本次保留修正后的对比学习实现，并补齐 `main.py` 创建 `CrossRQVAE` 时的权重参数传递。文本、图片对比权重显式设为 **1.0**，与首次复现实际生效的默认值一致；重建对齐权重为 **0.001**。RQ-VAE seed=2023，T5 seed=42，其余设置沿用首次 Instruments 复现。

复用已有交互数据、文本/图片特征及 KMeans 标签，从 RQ-VAE 开始重新训练。本文件整理从头复现的完整流程，省略服务器重启后的恢复过程；本次实际 T5 在第 78 轮早停，恢复阶段日志为 `logs/Instruments/confix-w1/t5_resume_3956.log`。

**每一步正常结束后再进入下一步。** 以下统一使用物理 GPU 1；启动前查看占用，需要换卡时同步调整 `--device cuda:1`、`--gpu_ids 1` 和 `--gpu_id 1`。`tail -f` 中按 `Ctrl+C` 只退出日志查看。输出使用独立的 `confix-w1` 路径；重复执行会覆盖该实验的同名输出，不覆盖首次实验和 `confix-v2` 实验。

## 1. 检查 GPU 并创建目录

```bash
nvidia-smi

mkdir -p /home/liuxuedi/projects/MACRec-di/cross_index/log/Instruments/MACRec-confix-w1-seed2023 /home/liuxuedi/projects/MACRec-di/log/Instruments-MACRec-confix-w1-rq2023-t542-single /home/liuxuedi/projects/MACRec-di/logs/Instruments/confix-w1
```

## 2. 训练 RQ-VAE

```bash
cd /home/liuxuedi/projects/MACRec-di/cross_index

setsid nohup /home/liuxuedi/anaconda3/envs/macrec/bin/python -u main.py \
  --text_data_path /home/liuxuedi/projects/MACRec-di/data/Instruments/Instruments.emb-llama-td.npy \
  --image_data_path /home/liuxuedi/projects/MACRec-di/data/Instruments/Instruments.emb-ViT-L-14.npy \
  --text_class_info /home/liuxuedi/projects/MACRec-di/data/Instruments/Instruments.index_lemb_kmeans512.json \
  --image_class_info /home/liuxuedi/projects/MACRec-di/data/Instruments/Instruments.index_vitemb_kmeans512.json \
  --ckpt_dir /home/liuxuedi/projects/MACRec-di/cross_index/log/Instruments/MACRec-confix-w1-seed2023 \
  --device cuda:1 \
  --learner AdamW \
  --lr 0.001 \
  --batch_size 1024 \
  --num_emb_list 256 256 256 256 \
  --sk_epsilons 0.0 0.0 0.0 0.0 \
  --use_cross_rq True \
  --begin_cross_layer 2 \
  --text_contrast_weight 1.0 \
  --image_contrast_weight 1.0 \
  --recon_contrast_weight 0.001 \
  --eval_step 2 \
  --epochs 1000 \
  > /home/liuxuedi/projects/MACRec-di/logs/Instruments/confix-w1/rqvae.log 2>&1 < /dev/null &

tail -n 30 -f /home/liuxuedi/projects/MACRec-di/logs/Instruments/confix-w1/rqvae.log
```

结束标记：`Best Loss`、`Best Text Collision Rate`、`Best Image Collision Rate`。模型目录应保存两路 `best_*_collision_model.pth`。

## 3. 生成文本语义 ID

```bash
cd /home/liuxuedi/projects/MACRec-di/cross_index

setsid nohup /home/liuxuedi/anaconda3/envs/macrec/bin/python -u generate_indices_distance.py \
  --dataset Instruments \
  --text_data_path /home/liuxuedi/projects/MACRec-di/data/Instruments/Instruments.emb-llama-td.npy \
  --image_data_path /home/liuxuedi/projects/MACRec-di/data/Instruments/Instruments.emb-ViT-L-14.npy \
  --device cuda:1 \
  --ckpt_path /home/liuxuedi/projects/MACRec-di/cross_index/log/Instruments/MACRec-confix-w1-seed2023/best_text_collision_model.pth \
  --output_dir /home/liuxuedi/projects/MACRec-di/data/Instruments \
  --output_file Instruments.index_lemb_MACRec-confix-w1-seed2023.json \
  --content text \
  > /home/liuxuedi/projects/MACRec-di/logs/Instruments/confix-w1/codes_text.log 2>&1 < /dev/null &

tail -n 30 -f /home/liuxuedi/projects/MACRec-di/logs/Instruments/confix-w1/codes_text.log
```

最终应显示 `Collision Rate 0`、`check: True`，并成功保存 JSON。

## 4. 生成图片语义 ID

```bash
cd /home/liuxuedi/projects/MACRec-di/cross_index

setsid nohup /home/liuxuedi/anaconda3/envs/macrec/bin/python -u generate_indices_distance.py \
  --dataset Instruments \
  --text_data_path /home/liuxuedi/projects/MACRec-di/data/Instruments/Instruments.emb-llama-td.npy \
  --image_data_path /home/liuxuedi/projects/MACRec-di/data/Instruments/Instruments.emb-ViT-L-14.npy \
  --device cuda:1 \
  --ckpt_path /home/liuxuedi/projects/MACRec-di/cross_index/log/Instruments/MACRec-confix-w1-seed2023/best_image_collision_model.pth \
  --output_dir /home/liuxuedi/projects/MACRec-di/data/Instruments \
  --output_file Instruments.index_vitemb_MACRec-confix-w1-seed2023.json \
  --content image \
  > /home/liuxuedi/projects/MACRec-di/logs/Instruments/confix-w1/codes_image.log 2>&1 < /dev/null &

tail -n 30 -f /home/liuxuedi/projects/MACRec-di/logs/Instruments/confix-w1/codes_image.log
```

最终应显示 `Collision Rate 0`、`check: True`，并成功保存 JSON。

## 5. 创建单卡启动配置

```bash
cat > /home/liuxuedi/projects/MACRec-di/logs/Instruments/confix-w1/accelerate-single.yaml <<'YAML'
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
mixed_precision: 'no'
use_cpu: false
num_processes: 1
num_machines: 1
machine_rank: 0
gpu_ids: '1'
main_training_function: main
YAML
```

## 6. 从头训练 T5

```bash
cd /home/liuxuedi/projects/MACRec-di

setsid nohup /home/liuxuedi/anaconda3/envs/macrec/bin/python -u -m accelerate.commands.launch \
  --config_file /home/liuxuedi/projects/MACRec-di/logs/Instruments/confix-w1/accelerate-single.yaml \
  --gpu_ids 1 \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision no \
  --dynamo_backend no \
  finetune_contrastive.py  \
  --seed 42 \
  --data_path /home/liuxuedi/projects/MACRec-di/data \
  --dataset Instruments \
  --index_file .index_lemb_MACRec-confix-w1-seed2023.json \
  --image_index_file .index_vitemb_MACRec-confix-w1-seed2023.json \
  --output_dir /home/liuxuedi/projects/MACRec-di/log/Instruments-MACRec-confix-w1-rq2023-t542-single \
  --base_model /home/liuxuedi/projects/MACRec-di/config/ckpt \
  --per_device_batch_size 1024 \
  --gradient_accumulation_steps 4 \
  --optim adamw_torch \
  --learning_rate 0.001 \
  --epochs 200 \
  --weight_decay 0.01 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type cosine \
  --save_and_eval_strategy epoch \
  --logging_step 50 \
  --max_his_len 20 \
  --prompt_num 4 \
  --patient 10 \
  --tasks seqrec,seqimage,item2image,image2item,seqimage2item,seqitem2image \
  --valid_task seqrec \
  > /home/liuxuedi/projects/MACRec-di/logs/Instruments/confix-w1/t5.log 2>&1 < /dev/null &

tail -n 30 -f /home/liuxuedi/projects/MACRec-di/logs/Instruments/confix-w1/t5.log
```

有效 batch 为 `1024×1×4=4096`。出现最终 `train_runtime`、`train_loss` 且无异常退出后完成；早停属于正常结束。最终输出目录保存最佳模型，测试时直接使用该目录。

## 7. 测试 seqrec

```bash
cd /home/liuxuedi/projects/MACRec-di

setsid nohup /home/liuxuedi/anaconda3/envs/macrec/bin/python -u test_ddp_save.py \
  --seed 42 \
  --gpu_id 1 \
  --ckpt_path /home/liuxuedi/projects/MACRec-di/log/Instruments-MACRec-confix-w1-rq2023-t542-single \
  --data_path /home/liuxuedi/projects/MACRec-di/data \
  --dataset Instruments \
  --index_file .index_lemb_MACRec-confix-w1-seed2023.json \
  --image_index_file .index_vitemb_MACRec-confix-w1-seed2023.json \
  --test_task seqrec \
  --test_batch_size 64 \
  --num_beams 20 \
  --filter_items  \
  --results_file /home/liuxuedi/projects/MACRec-di/log/Instruments-MACRec-confix-w1-rq2023-t542-single/results_seqrec_20.json \
  --save_file /home/liuxuedi/projects/MACRec-di/log/Instruments-MACRec-confix-w1-rq2023-t542-single/save_seqrec_20.json \
  > /home/liuxuedi/projects/MACRec-di/logs/Instruments/confix-w1/test_seqrec.log 2>&1 < /dev/null &

tail -n 30 -f /home/liuxuedi/projects/MACRec-di/logs/Instruments/confix-w1/test_seqrec.log
```

应以 `Save file:` 正常结束，且没有 `Out of memory!` 或降低 beam 的 `Beam:` 提示。

## 8. 测试 seqimage

```bash
cd /home/liuxuedi/projects/MACRec-di

setsid nohup /home/liuxuedi/anaconda3/envs/macrec/bin/python -u test_ddp_save.py \
  --seed 42 \
  --gpu_id 1 \
  --ckpt_path /home/liuxuedi/projects/MACRec-di/log/Instruments-MACRec-confix-w1-rq2023-t542-single \
  --data_path /home/liuxuedi/projects/MACRec-di/data \
  --dataset Instruments \
  --index_file .index_lemb_MACRec-confix-w1-seed2023.json \
  --image_index_file .index_vitemb_MACRec-confix-w1-seed2023.json \
  --test_task seqimage \
  --test_batch_size 64 \
  --num_beams 20 \
  --filter_items  \
  --results_file /home/liuxuedi/projects/MACRec-di/log/Instruments-MACRec-confix-w1-rq2023-t542-single/results_seqimage_20.json \
  --save_file /home/liuxuedi/projects/MACRec-di/log/Instruments-MACRec-confix-w1-rq2023-t542-single/save_seqimage_20.json \
  > /home/liuxuedi/projects/MACRec-di/logs/Instruments/confix-w1/test_seqimage.log 2>&1 < /dev/null &

tail -n 30 -f /home/liuxuedi/projects/MACRec-di/logs/Instruments/confix-w1/test_seqimage.log
```

应以 `Save file:` 正常结束，且没有 `Out of memory!` 或降低 beam 的 `Beam:` 提示。

## 9. 融合结果（CPU）

```bash
cd /home/liuxuedi/projects/MACRec-di

setsid nohup /home/liuxuedi/anaconda3/envs/macrec/bin/python -u ensemble.py \
  --output_dir /home/liuxuedi/projects/MACRec-di/log/Instruments-MACRec-confix-w1-rq2023-t542-single \
  --data_path /home/liuxuedi/projects/MACRec-di/data \
  --dataset Instruments \
  --index_file .index_lemb_MACRec-confix-w1-seed2023.json \
  --image_index_file .index_vitemb_MACRec-confix-w1-seed2023.json \
  --num_beams 20 \
  > /home/liuxuedi/projects/MACRec-di/logs/Instruments/confix-w1/ensemble.log 2>&1 < /dev/null &

tail -n 30 -f /home/liuxuedi/projects/MACRec-di/logs/Instruments/confix-w1/ensemble.log
```

两路测试均正常结束后执行。融合完成后查看：

```bash
cat /home/liuxuedi/projects/MACRec-di/log/Instruments-MACRec-confix-w1-rq2023-t542-single/results_ensemble_20.json
```

本次融合结果（单次运行）：

| HR@1 | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|
| 0.080762 | 0.109748 | 0.135460 | 0.095593 | 0.103832 |
