# MACRec Arts / Games 复现运行指令

完成日期：2026-11-15。采用仓库已有的交互、商品信息和 ID 映射，不重新预处理。

**使用方式：先设 `D=Arts`，按步骤执行；再设 `D=Games`，重复相同步骤。每个后台任务正常结束后再执行下一步。** 同一终端保留下列变量；重连后重新执行第 1 步。`nohup` 防 SSH 断连，不能防服务器重启；`Ctrl+C` 只退出日志查看。重跑会覆盖对应日志和输出。

## 1. 设置数据集、路径与 GPU

```bash
export R=/home/liuxuedi/projects/MACRec-di
export PY=/home/liuxuedi/anaconda3/envs/macrec/bin/python
export D=Arts                         # 完成 Arts 后改为 Games
export GPU=GPU-80a9c4c6-fb90-85e1-923b-942a1c78b1e0
nvidia-smi \
  --query-gpu=index,uuid,memory.used,utilization.gpu \
  --format=csv
mkdir -p "$R/logs/$D" "$R/data/Images/$D"   "$R/cross_index/log/$D/MACRec-paper-seed2023"   "$R/log/$D-MACRec-rq2023-t542-single"
```

GPU UUID 按启动时的空闲情况选择；下文 `cuda:0` / `--gpu_id 0` 指选中的那张卡。

输入文件：

| 数据集 | Arts | Games |
|---|---|---|
| 商品数 | 9,416 | 13,839 |
| metadata（`data/raw/Metadata/`） | `meta_Arts_Crafts_and_Sewing.json.gz` | `meta_Video_Games.json.gz` |
| 5-core（`data/raw/Review/`） | `Arts_Crafts_and_Sewing_5.json.gz` | `Video_Games_5.json.gz` |

`data/$D/` 已有 `$D.inter.json`、`$D.item.json`、`$D.item2id`。模型已下载到 `/home/liuxuedi/projects/models/llama-7b/` 和 `/home/liuxuedi/projects/models/clip/ViT-L-14.pt`。

## 2. 准备图片

本次图片下载、补齐已完成，现有文件可直接复用：

```text
data/Images/Arts/                 # Arts 商品图片
data/Images/Arts_images_info.json
data/Images/Games/                # Games 商品图片
data/Images/Games_images_info.json
```

从原始数据重新下载时，先按 [Instruments 记录第 2 步](MACRec_Instruments_复现运行指令.md#2-下载图片) 创建 `/tmp/macrec_tls12.py` 并保持代理可用，再执行：

```bash
cd "$R/data_process"
nohup "$PY" -u -c 'import sys, runpy; sys.path.insert(0, "/tmp"); import macrec_tls12; runpy.run_path("/home/liuxuedi/projects/MACRec-di/data_process/load_all_figures.py", run_name="__main__"); print("DOWNLOAD_IMAGES_DONE", flush=True)' \
  --dataset "$D" \
  --meta_data_path "$R/data/raw/Metadata" \
  --rating_data_path "$R/data/raw/Ratings" \
  --review_data_path "$R/data/raw/Review" \
  --save_path "$R/data/Images"  \
  > "$R/logs/$D/download_images.log" 2>&1 < /dev/null &
tail -n 30 -f "$R/logs/$D/download_images.log"
```

结束标记：`DOWNLOAD_IMAGES_DONE`。生成特征前须确保实验商品均有有效图片；本次已补齐，网络诊断及补图重试命令不再列出。

## 3. 生成文本特征（CPU）

沿用默认 FP32，输入商品标题和描述。

```bash
cd "$R/data_process"
nohup "$PY" -u amazon_text_emb.py \
  --dataset "$D" \
  --root "$R/data" \
  --gpu_id -1 \
  --plm_name llama \
  --model_name_or_path /home/liuxuedi/projects/models/llama-7b \
  --model_cache_dir /home/liuxuedi/projects/models/llama-7b \
  --max_sent_len 2048 \
  --word_drop_ratio -1  \
  > "$R/logs/$D/text_features.log" 2>&1 < /dev/null &
tail -n 30 -f "$R/logs/$D/text_features.log"
```

输出 `data/$D/$D.emb-llama-td.npy`，形状为 `(商品数, 4096)`；整个数据集处理完才保存。

## 4. 生成图片特征（CPU）

本次因 GPU 忙碌使用 CPU。

```bash
cd "$R/data_process"
CUDA_VISIBLE_DEVICES="" nohup "$PY" -u clip_feature.py \
  --dataset "$D" \
  --image_root "$R/data/Images" \
  --save_root "$R/data" \
  --model_cache_dir /home/liuxuedi/projects/models/clip \
  --backbone ViT-L/14  \
  > "$R/logs/$D/image_features.log" 2>&1 < /dev/null &
tail -n 30 -f "$R/logs/$D/image_features.log"
```

输出 `data/$D/$D.emb-ViT-L-14.npy`，形状为 `(商品数, 768)`。

## 5. KMeans 伪标签（CPU）

**手动将 `data/kmeans.py` 中的 `dataset` 改为当前数据集：`"Arts"` 或 `"Games"`。** 保持 `n_clusters=512`、`random_state=42`、`n_init=10`。只有这一步需要修改代码中的数据集名称。

```bash
cd "$R/data"
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 nohup "$PY" -u kmeans.py > "$R/logs/$D/kmeans.log" 2>&1 < /dev/null &
tail -n 30 -f "$R/logs/$D/kmeans.log"
```

两条 `Saved to` 出现后完成。输出 `$D.index_lemb_kmeans512.json`、`$D.index_vitemb_kmeans512.json`，各含全部商品和 512 个簇。

## 6. 训练跨模态 RQ-VAE（单卡）

种子 2023 在 `cross_index/main.py` 中固定。

```bash
cd "$R/cross_index"
CUDA_VISIBLE_DEVICES="$GPU" nohup "$PY" -u main.py \
  --text_data_path "$R/data/$D/$D.emb-llama-td.npy" \
  --image_data_path "$R/data/$D/$D.emb-ViT-L-14.npy" \
  --text_class_info "$R/data/$D/$D.index_lemb_kmeans512.json" \
  --image_class_info "$R/data/$D/$D.index_vitemb_kmeans512.json" \
  --ckpt_dir "$R/cross_index/log/$D/MACRec-paper-seed2023" \
  --device cuda:0 \
  --learner AdamW \
  --lr 0.001 \
  --batch_size 1024 \
  --num_emb_list 256 256 256 256 \
  --sk_epsilons 0.0 0.0 0.0 0.0 \
  --use_cross_rq True \
  --begin_cross_layer 2 \
  --text_contrast_weight 0.1 \
  --image_contrast_weight 0.1 \
  --recon_contrast_weight 0.001 \
  --eval_step 2 \
  --epochs 1000  \
  > "$R/logs/$D/rqvae_seed2023.log" 2>&1 < /dev/null &
tail -n 30 -f "$R/logs/$D/rqvae_seed2023.log"
```

结束标记：`Best Loss`、`Best Text Collision Rate`、`Best Image Collision Rate`。模型目录中应有非空的 `best_text_collision_model.pth` 和 `best_image_collision_model.pth`。

## 7. 生成文本语义 ID

```bash
cd "$R/cross_index"
CUDA_VISIBLE_DEVICES="$GPU" nohup "$PY" -u generate_indices_distance.py \
  --dataset "$D" \
  --text_data_path "$R/data/$D/$D.emb-llama-td.npy" \
  --image_data_path "$R/data/$D/$D.emb-ViT-L-14.npy" \
  --device cuda:0 \
  --ckpt_path "$R/cross_index/log/$D/MACRec-paper-seed2023/best_text_collision_model.pth" \
  --output_dir "$R/data/$D" \
  --output_file "$D.index_lemb_MACRec-paper-seed2023.json" \
  --content text  \
  > "$R/logs/$D/codes_text_seed2023.log" 2>&1 < /dev/null &
tail -n 30 -f "$R/logs/$D/codes_text_seed2023.log"
```

## 8. 生成图片语义 ID

```bash
cd "$R/cross_index"
CUDA_VISIBLE_DEVICES="$GPU" nohup "$PY" -u generate_indices_distance.py \
  --dataset "$D" \
  --text_data_path "$R/data/$D/$D.emb-llama-td.npy" \
  --image_data_path "$R/data/$D/$D.emb-ViT-L-14.npy" \
  --device cuda:0 \
  --ckpt_path "$R/cross_index/log/$D/MACRec-paper-seed2023/best_image_collision_model.pth" \
  --output_dir "$R/data/$D" \
  --output_file "$D.index_vitemb_MACRec-paper-seed2023.json" \
  --content image  \
  > "$R/logs/$D/codes_image_seed2023.log" 2>&1 < /dev/null &
tail -n 30 -f "$R/logs/$D/codes_image_seed2023.log"
```

两路均须最终 `Collision Rate 0`、`check: True`，并成功保存 JSON。本次两组数据均为全部商品唯一编码，每个编码 4 层。

## 9. 训练 T5（单卡）

本次 T5 seed=42，沿用 Instruments 的单次运行配置；不将其解释为论文指定种子。有效 batch 为 `1024×1×4=4096`。

```bash
cd "$R"
CUDA_VISIBLE_DEVICES="$GPU" nohup "$PY" -u finetune_contrastive.py \
  --seed 42 \
  --data_path "$R/data" \
  --dataset "$D" \
  --index_file .index_lemb_MACRec-paper-seed2023.json \
  --image_index_file .index_vitemb_MACRec-paper-seed2023.json \
  --output_dir "$R/log/$D-MACRec-rq2023-t542-single" \
  --base_model "$R/config/ckpt" \
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
  --valid_task seqrec  \
  > "$R/logs/$D/t5_rq2023_seed42_single.log" 2>&1 < /dev/null &
tail -n 30 -f "$R/logs/$D/t5_rq2023_seed42_single.log"
```

正常结束出现 `train_runtime`、`train_loss`；早停时进度条不必到 100%。最终目录保存最佳模型。本次 Arts 第 89 轮结束（第 79 轮最优），Games 第 116 轮结束（约第 106 轮最优）。

## 10. 测试 seqrec

使用仓库已适配单卡的 `test_ddp_save.py`，直接 Python 启动。

```bash
cd "$R"
CUDA_VISIBLE_DEVICES="$GPU" nohup "$PY" -u test_ddp_save.py \
  --seed 42 \
  --gpu_id 0 \
  --ckpt_path "$R/log/$D-MACRec-rq2023-t542-single" \
  --data_path "$R/data" \
  --dataset "$D" \
  --index_file .index_lemb_MACRec-paper-seed2023.json \
  --image_index_file .index_vitemb_MACRec-paper-seed2023.json \
  --test_task seqrec \
  --test_batch_size 64 \
  --num_beams 20 \
  --filter_items \
  --results_file "$R/log/$D-MACRec-rq2023-t542-single/results_seqrec_20.json" \
  --save_file "$R/log/$D-MACRec-rq2023-t542-single/save_seqrec_20.json"  \
  > "$R/logs/$D/test_seqrec_single.log" 2>&1 < /dev/null &
tail -n 30 -f "$R/logs/$D/test_seqrec_single.log"
```

## 11. 测试 seqimage

```bash
cd "$R"
CUDA_VISIBLE_DEVICES="$GPU" nohup "$PY" -u test_ddp_save.py \
  --seed 42 \
  --gpu_id 0 \
  --ckpt_path "$R/log/$D-MACRec-rq2023-t542-single" \
  --data_path "$R/data" \
  --dataset "$D" \
  --index_file .index_lemb_MACRec-paper-seed2023.json \
  --image_index_file .index_vitemb_MACRec-paper-seed2023.json \
  --test_task seqimage \
  --test_batch_size 64 \
  --num_beams 20 \
  --filter_items \
  --results_file "$R/log/$D-MACRec-rq2023-t542-single/results_seqimage_20.json" \
  --save_file "$R/log/$D-MACRec-rq2023-t542-single/save_seqimage_20.json"  \
  > "$R/logs/$D/test_seqimage_single.log" 2>&1 < /dev/null &
tail -n 30 -f "$R/logs/$D/test_seqimage_single.log"
```

两路测试均应以 `Save file:` 正常结束，且无 `Out of memory!` 或 `Beam:`（避免 beam 被自动降低）。

## 12. 融合与查看结果（CPU）

两路测试均结束后执行：

```bash
cd "$R"
nohup "$PY" -u ensemble.py \
  --output_dir "$R/log/$D-MACRec-rq2023-t542-single" \
  --data_path "$R/data" \
  --dataset "$D" \
  --index_file .index_lemb_MACRec-paper-seed2023.json \
  --image_index_file .index_vitemb_MACRec-paper-seed2023.json \
  --num_beams 20  \
  > "$R/logs/$D/ensemble_single.log" 2>&1 < /dev/null &
tail -n 30 -f "$R/logs/$D/ensemble_single.log"
```

日志打印指标和样本数、进程结束后查看：

```bash
cat "$R/log/$D-MACRec-rq2023-t542-single/results_ensemble_20.json"
```

本次融合结果（单次运行，非多种子均值）：

| 数据集 | HR@1 | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|
| Arts | 0.062785 | 0.098597 | 0.126020 | 0.081443 | 0.090247 |
| Games | 0.020729 | 0.062117 | 0.102037 | 0.041353 | 0.054152 |
