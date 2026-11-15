# MACRec Instruments 复现运行记录

完成日期：2026-11-10。按下列顺序执行，每一步正常结束后再进入下一步。

采用仓库已有的 Instruments 数据（6,250 商品），不重新预处理。模型和原始数据已提前下载；命令使用绝对路径，GPU 按 UUID 指定。`nohup` 防 SSH 断连；查看日志时 `Ctrl+C` 只退出查看。再次运行前检查 GPU 占用，避免覆盖已有实验。

## 1. 输入文件与目录

```text
/home/liuxuedi/projects/models/llama-7b/         # 完整模型、原版索引和 tokenizer
/home/liuxuedi/projects/models/clip/ViT-L-14.pt
/home/liuxuedi/projects/MACRec-di/data/Instruments/  # 仓库已有 JSON、ID 映射及划分
/home/liuxuedi/projects/MACRec-di/data/raw/
├── Metadata/meta_Musical_Instruments.json.gz
├── Ratings/Musical_Instruments.csv
└── Review/Musical_Instruments_5.json.gz
```

```bash
mkdir -p /home/liuxuedi/projects/MACRec-di/logs/Instruments   /home/liuxuedi/projects/MACRec-di/data/Images/Instruments
```

## 2. 下载图片

本机成功下载需要代理 `127.0.0.1:17891` 和 TLS 1.2。代理须保持可用。仅创建独立网络配置，不修改作者程序；以下省略了失败的诊断过程。

```bash
cat > /tmp/macrec_tls12.py <<'PY'
import ssl
import requests
from requests.adapters import HTTPAdapter

class TLS12Adapter(HTTPAdapter):
    def proxy_manager_for(self, proxy, **kwargs):
        context = ssl.create_default_context()
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.maximum_version = ssl.TLSVersion.TLSv1_2
        kwargs["ssl_context"] = context
        return super().proxy_manager_for(proxy, **kwargs)

original_init = requests.Session.__init__

def session_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.trust_env = False
    self.proxies.update({
        "http": "http://127.0.0.1:17891",
        "https": "http://127.0.0.1:17891",
    })
    self.mount("https://", TLS12Adapter())

requests.Session.__init__ = session_init
PY
```

```bash
cd /home/liuxuedi/projects/MACRec-di/data_process
nohup /home/liuxuedi/anaconda3/envs/macrec/bin/python -u -c \
'import sys, runpy; sys.path.insert(0, "/tmp"); import macrec_tls12; runpy.run_path("/home/liuxuedi/projects/MACRec-di/data_process/load_all_figures.py", run_name="__main__"); print("DOWNLOAD_IMAGES_DONE", flush=True)' \
  --dataset Instruments \
  --meta_data_path /home/liuxuedi/projects/MACRec-di/data/raw/Metadata \
  --rating_data_path /home/liuxuedi/projects/MACRec-di/data/raw/Ratings \
  --review_data_path /home/liuxuedi/projects/MACRec-di/data/raw/Review \
  --save_path /home/liuxuedi/projects/MACRec-di/data/Images \
  > /home/liuxuedi/projects/MACRec-di/logs/Instruments/download_images.log 2>&1 < /dev/null &

tail -n 50 -f /home/liuxuedi/projects/MACRec-di/logs/Instruments/download_images.log
```

## 3. 补齐本次唯一缺图

本次缺失商品为 `B000EEHEYW`；仅在该商品仍缺图时执行。图片验证后更新映射。

```bash
curl --fail --location --proxy http://127.0.0.1:17891 \
  --connect-timeout 10 --max-time 60 --retry 3 \
  -o /home/liuxuedi/projects/MACRec-di/data/Images/Instruments/219jgqSNj3L.jpg \
  https://images-na.ssl-images-amazon.com/images/I/219jgqSNj3L.jpg

/home/liuxuedi/anaconda3/envs/macrec/bin/python - <<'PY'
import json
from pathlib import Path
from PIL import Image
root = Path("/home/liuxuedi/projects/MACRec-di/data/Images")
with Image.open(root / "Instruments/219jgqSNj3L.jpg") as image:
    image.verify()
p = root / "Instruments_images_info.json"
data = json.loads(p.read_text())
data["B000EEHEYW"] = ["219jgqSNj3L.jpg"]
p.write_text(json.dumps(data, indent=4))
print("缺图已补齐")
PY
```

本次最终检查：6,250 个推荐商品均有有效图片。

## 4. 生成图片特征

```bash
cd /home/liuxuedi/projects/MACRec-di/data_process
CUDA_VISIBLE_DEVICES=GPU-96842970-f363-184f-2240-7aa2d37c9b21 \
nohup /home/liuxuedi/anaconda3/envs/macrec/bin/python -u /home/liuxuedi/projects/MACRec-di/data_process/clip_feature.py \
  --dataset Instruments \
  --image_root /home/liuxuedi/projects/MACRec-di/data/Images \
  --save_root /home/liuxuedi/projects/MACRec-di/data \
  --model_cache_dir /home/liuxuedi/projects/models/clip \
  --backbone ViT-L/14 \
  > /home/liuxuedi/projects/MACRec-di/logs/Instruments/image_features.log 2>&1 < /dev/null &

tail -n 50 -f /home/liuxuedi/projects/MACRec-di/logs/Instruments/image_features.log
```

完成输出：`Instruments.emb-ViT-L-14.npy`，形状 `(6250, 768)`。

## 5. 生成文本特征

保持作者代码默认 FP32 加载，使用 CPU，避免单张 24 GB GPU 无法容纳 LLaMA-7B；本次机器内存充足。

```bash
cd /home/liuxuedi/projects/MACRec-di/data_process
nohup /home/liuxuedi/anaconda3/envs/macrec/bin/python -u /home/liuxuedi/projects/MACRec-di/data_process/amazon_text_emb.py \
  --dataset Instruments \
  --root /home/liuxuedi/projects/MACRec-di/data \
  --gpu_id -1 \
  --plm_name llama \
  --model_name_or_path /home/liuxuedi/projects/models/llama-7b \
  --model_cache_dir /home/liuxuedi/projects/models/llama-7b \
  --max_sent_len 2048 \
  --word_drop_ratio -1 \
  > /home/liuxuedi/projects/MACRec-di/logs/Instruments/text_features.log 2>&1 < /dev/null &

tail -n 50 -f /home/liuxuedi/projects/MACRec-di/logs/Instruments/text_features.log
```

完成输出：`Instruments.emb-llama-td.npy`，形状 `(6250, 4096)`。

## 6. KMeans 伪标签

将 `data/kmeans.py` 中 `dataset = "Games"` 改为 `dataset = "Instruments"`；保留 `n_clusters=512`、`random_state=42`、`n_init=10`。

```bash
cd /home/liuxuedi/projects/MACRec-di/data
nohup /home/liuxuedi/anaconda3/envs/macrec/bin/python -u /home/liuxuedi/projects/MACRec-di/data/kmeans.py \
  > /home/liuxuedi/projects/MACRec-di/logs/Instruments/kmeans.log 2>&1 < /dev/null &

tail -n 50 -f /home/liuxuedi/projects/MACRec-di/logs/Instruments/kmeans.log
```

生成 `Instruments.index_lemb_kmeans512.json`、`Instruments.index_vitemb_kmeans512.json`；各含 6,250 个商品、512 个簇。

## 7. 训练跨模态 RQ-VAE

论文设置：batch 1024、4×256 码本、第三层起对比（索引 2）、对比权重 0.1、重建对齐权重 0.001。1000 epochs 和每两轮评估沿用作者脚本；种子 2023 在作者代码中固定。

```bash
mkdir -p /home/liuxuedi/projects/MACRec-di/cross_index/log/Instruments/MACRec-paper-seed2023
```

```bash
cd /home/liuxuedi/projects/MACRec-di/cross_index
CUDA_VISIBLE_DEVICES=GPU-96842970-f363-184f-2240-7aa2d37c9b21 \
nohup /home/liuxuedi/anaconda3/envs/macrec/bin/python -u /home/liuxuedi/projects/MACRec-di/cross_index/main.py \
  --text_data_path /home/liuxuedi/projects/MACRec-di/data/Instruments/Instruments.emb-llama-td.npy \
  --image_data_path /home/liuxuedi/projects/MACRec-di/data/Instruments/Instruments.emb-ViT-L-14.npy \
  --text_class_info /home/liuxuedi/projects/MACRec-di/data/Instruments/Instruments.index_lemb_kmeans512.json \
  --image_class_info /home/liuxuedi/projects/MACRec-di/data/Instruments/Instruments.index_vitemb_kmeans512.json \
  --ckpt_dir /home/liuxuedi/projects/MACRec-di/cross_index/log/Instruments/MACRec-paper-seed2023 \
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
  --epochs 1000 \
  > /home/liuxuedi/projects/MACRec-di/logs/Instruments/rqvae_seed2023.log 2>&1 < /dev/null &

tail -n 50 -f /home/liuxuedi/projects/MACRec-di/logs/Instruments/rqvae_seed2023.log
```

## 8. 生成文本语义 ID

```bash
cd /home/liuxuedi/projects/MACRec-di/cross_index
CUDA_VISIBLE_DEVICES=GPU-96842970-f363-184f-2240-7aa2d37c9b21 \
nohup /home/liuxuedi/anaconda3/envs/macrec/bin/python -u /home/liuxuedi/projects/MACRec-di/cross_index/generate_indices_distance.py \
  --dataset Instruments \
  --text_data_path /home/liuxuedi/projects/MACRec-di/data/Instruments/Instruments.emb-llama-td.npy \
  --image_data_path /home/liuxuedi/projects/MACRec-di/data/Instruments/Instruments.emb-ViT-L-14.npy \
  --device cuda:0 \
  --ckpt_path /home/liuxuedi/projects/MACRec-di/cross_index/log/Instruments/MACRec-paper-seed2023/best_text_collision_model.pth \
  --output_dir /home/liuxuedi/projects/MACRec-di/data/Instruments \
  --output_file Instruments.index_lemb_MACRec-paper-seed2023.json \
  --content text \
  > /home/liuxuedi/projects/MACRec-di/logs/Instruments/codes_text_seed2023.log 2>&1 < /dev/null &

tail -n 50 -f /home/liuxuedi/projects/MACRec-di/logs/Instruments/codes_text_seed2023.log
```

本次检查通过：6,250 个商品，6,250 个唯一编码，每个编码 4 层。

## 9. 生成图片语义 ID

```bash
cd /home/liuxuedi/projects/MACRec-di/cross_index
CUDA_VISIBLE_DEVICES=GPU-96842970-f363-184f-2240-7aa2d37c9b21 \
nohup /home/liuxuedi/anaconda3/envs/macrec/bin/python -u /home/liuxuedi/projects/MACRec-di/cross_index/generate_indices_distance.py \
  --dataset Instruments \
  --text_data_path /home/liuxuedi/projects/MACRec-di/data/Instruments/Instruments.emb-llama-td.npy \
  --image_data_path /home/liuxuedi/projects/MACRec-di/data/Instruments/Instruments.emb-ViT-L-14.npy \
  --device cuda:0 \
  --ckpt_path /home/liuxuedi/projects/MACRec-di/cross_index/log/Instruments/MACRec-paper-seed2023/best_image_collision_model.pth \
  --output_dir /home/liuxuedi/projects/MACRec-di/data/Instruments \
  --output_file Instruments.index_vitemb_MACRec-paper-seed2023.json \
  --content image \
  > /home/liuxuedi/projects/MACRec-di/logs/Instruments/codes_image_seed2023.log 2>&1 < /dev/null &

tail -n 50 -f /home/liuxuedi/projects/MACRec-di/logs/Instruments/codes_image_seed2023.log
```

本次检查通过：6,250 个商品，6,250 个唯一编码，每个编码 4 层。

## 10. 单卡训练 T5

采用作者模型配置和训练设置；单卡适配将梯度累积改为 4，保持有效 batch 为 `1024×1×4=4096`。本次 seed=42，第 81 轮早停，第 71 轮验证最优；最终目录已保存最佳模型。

```bash
mkdir -p /home/liuxuedi/projects/MACRec-di/log/Instruments-MACRec-rq2023-t542-single
```

```bash
cd /home/liuxuedi/projects/MACRec-di
CUDA_VISIBLE_DEVICES=GPU-96842970-f363-184f-2240-7aa2d37c9b21 \
nohup /home/liuxuedi/anaconda3/envs/macrec/bin/python -u /home/liuxuedi/projects/MACRec-di/finetune_contrastive.py \
  --seed 42 \
  --data_path /home/liuxuedi/projects/MACRec-di/data \
  --dataset Instruments \
  --index_file .index_lemb_MACRec-paper-seed2023.json \
  --image_index_file .index_vitemb_MACRec-paper-seed2023.json \
  --output_dir /home/liuxuedi/projects/MACRec-di/log/Instruments-MACRec-rq2023-t542-single \
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
  > /home/liuxuedi/projects/MACRec-di/logs/Instruments/t5_rq2023_seed42_single.log 2>&1 < /dev/null &

tail -n 50 -f /home/liuxuedi/projects/MACRec-di/logs/Instruments/t5_rq2023_seed42_single.log
```

## 11. 单卡测试：seqrec

使用已适配单卡的 `test_ddp_save.py`，直接 Python 启动，不使用 torchrun。batch 64、beam 20 沿用作者脚本；若出现 `Out of memory!` 或 `Beam:`，需检查是否自动降低了 beam。

```bash
cd /home/liuxuedi/projects/MACRec-di
CUDA_VISIBLE_DEVICES=GPU-96842970-f363-184f-2240-7aa2d37c9b21 \
nohup /home/liuxuedi/anaconda3/envs/macrec/bin/python -u /home/liuxuedi/projects/MACRec-di/test_ddp_save.py \
  --seed 42 \
  --gpu_id 0 \
  --ckpt_path /home/liuxuedi/projects/MACRec-di/log/Instruments-MACRec-rq2023-t542-single \
  --data_path /home/liuxuedi/projects/MACRec-di/data \
  --dataset Instruments \
  --index_file .index_lemb_MACRec-paper-seed2023.json \
  --image_index_file .index_vitemb_MACRec-paper-seed2023.json \
  --test_task seqrec \
  --test_batch_size 64 \
  --num_beams 20 \
  --filter_items \
  --results_file /home/liuxuedi/projects/MACRec-di/log/Instruments-MACRec-rq2023-t542-single/results_seqrec_20.json \
  --save_file /home/liuxuedi/projects/MACRec-di/log/Instruments-MACRec-rq2023-t542-single/save_seqrec_20.json \
  > /home/liuxuedi/projects/MACRec-di/logs/Instruments/test_seqrec_single.log 2>&1 < /dev/null &

tail -n 50 -f /home/liuxuedi/projects/MACRec-di/logs/Instruments/test_seqrec_single.log
```

## 12. 单卡测试：seqimage

使用已适配单卡的 `test_ddp_save.py`，直接 Python 启动，不使用 torchrun。batch 64、beam 20 沿用作者脚本；若出现 `Out of memory!` 或 `Beam:`，需检查是否自动降低了 beam。

```bash
cd /home/liuxuedi/projects/MACRec-di
CUDA_VISIBLE_DEVICES=GPU-96842970-f363-184f-2240-7aa2d37c9b21 \
nohup /home/liuxuedi/anaconda3/envs/macrec/bin/python -u /home/liuxuedi/projects/MACRec-di/test_ddp_save.py \
  --seed 42 \
  --gpu_id 0 \
  --ckpt_path /home/liuxuedi/projects/MACRec-di/log/Instruments-MACRec-rq2023-t542-single \
  --data_path /home/liuxuedi/projects/MACRec-di/data \
  --dataset Instruments \
  --index_file .index_lemb_MACRec-paper-seed2023.json \
  --image_index_file .index_vitemb_MACRec-paper-seed2023.json \
  --test_task seqimage \
  --test_batch_size 64 \
  --num_beams 20 \
  --filter_items \
  --results_file /home/liuxuedi/projects/MACRec-di/log/Instruments-MACRec-rq2023-t542-single/results_seqimage_20.json \
  --save_file /home/liuxuedi/projects/MACRec-di/log/Instruments-MACRec-rq2023-t542-single/save_seqimage_20.json \
  > /home/liuxuedi/projects/MACRec-di/logs/Instruments/test_seqimage_single.log 2>&1 < /dev/null &

tail -n 50 -f /home/liuxuedi/projects/MACRec-di/logs/Instruments/test_seqimage_single.log
```

## 13. 融合与最终结果

两路测试均正常结束后执行，融合不使用 GPU。

```bash
cd /home/liuxuedi/projects/MACRec-di
nohup /home/liuxuedi/anaconda3/envs/macrec/bin/python -u /home/liuxuedi/projects/MACRec-di/ensemble.py \
  --output_dir /home/liuxuedi/projects/MACRec-di/log/Instruments-MACRec-rq2023-t542-single \
  --data_path /home/liuxuedi/projects/MACRec-di/data \
  --dataset Instruments \
  --index_file .index_lemb_MACRec-paper-seed2023.json \
  --image_index_file .index_vitemb_MACRec-paper-seed2023.json \
  --num_beams 20 \
  > /home/liuxuedi/projects/MACRec-di/logs/Instruments/ensemble_single.log 2>&1 < /dev/null &

tail -n 50 -f /home/liuxuedi/projects/MACRec-di/logs/Instruments/ensemble_single.log
```

融合结束后查看：

```bash
cat /home/liuxuedi/projects/MACRec-di/log/Instruments-MACRec-rq2023-t542-single/results_ensemble_20.json
```

本次结果：

| HR@1 | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|
| 0.079476 | 0.109163 | 0.134175 | 0.094612 | 0.102581 |

这是一次运行结果，不是论文五个随机种子的均值。
