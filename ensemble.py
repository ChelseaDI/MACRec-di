import json
import math
import os
import argparse
from collections import defaultdict

from evaluate import get_topk_results, get_metrics_results

def get_sort_results(predictions, scores, targets_ids, users, k, index2id):
    
    B = len(users)
    predictions = [_.strip().replace(" ","") for _ in predictions]

    
    infos = []
    
    for b in range(B):
        batch_seqs = predictions[b * k: (b + 1) * k]
        batch_scores = scores[b * k: (b + 1) * k]
        
        batch_items = []
        for index in batch_seqs:
            if index in index2id:
                batch_items.append(index2id[index])
            else:
                batch_items.append([-1])
             
        
        target = targets_ids[b]
        user = users[b]
        
        infos.append((user, target, batch_items, batch_scores))
        
    sorted_infos = sorted(infos, key=lambda x: x[0])
    
    return sorted_infos

def get_topk_results_ensemble(text_info, image_info):
    
    assert len(text_info) == len(image_info)
    
    B = len(text_info)

    results = []
    
    for b in range(B):
        text_user, text_target, text_batch_items, text_batch_scores = text_info[b]
        image_user, image_target, image_batch_items, image_batch_scores = image_info[b]
        
        assert text_user == image_user
        if text_target != image_target:
            print(text_target, image_target)
        target_item = text_target

        item_id2score = {}
        
        for i in range(len(text_batch_items)):

            item_id = text_batch_items[i][0]
            score = text_batch_scores[i]
            
            if item_id == -1:
                score = -1000
            
            if item_id in item_id2score and item_id != -1:

                item_id2score[item_id] = (score + item_id2score[item_id]) / 2 + 1
            else:
                item_id2score[item_id] = score
            

            item_id = image_batch_items[i][0]
            score = image_batch_scores[i]
            
            if item_id == -1:
                score = -1000
            
            if item_id in item_id2score and item_id != -1:

                item_id2score[item_id] = (score + item_id2score[item_id]) / 2 + 1
            else:
                item_id2score[item_id] = score
                
                
        pairs = []
        for item_id, score in item_id2score.items():
            pairs.append((item_id, score))
        

        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

        one_results = []
        for sorted_pred in sorted_pairs:
            if sorted_pred[0] in target_item:
                one_results.append(1)
            else:
                one_results.append(0)

        results.append(one_results)

    return results

def get_candidate_distribution(items, scores, temperature):
    item_scores = {}
    for item_ids, score in zip(items, scores):
        item_id = item_ids[0]
        if item_id != -1 and math.isfinite(score):
            item_scores[item_id] = max(score, item_scores.get(item_id, -math.inf))

    if not item_scores:
        return {}

    max_score = max(item_scores.values())
    masses = {
        item_id: math.exp((score - max_score) / temperature)
        for item_id, score in item_scores.items()
    }
    total = sum(masses.values())
    return {item_id: mass / total for item_id, mass in masses.items()}


def normalized_entropy(probabilities):
    entropy = -sum(p * math.log(p) for p in probabilities.values() if p > 0)
    return entropy / math.log(len(probabilities))


def get_topk_results_weighted(text_info, image_info, args):
    if len(text_info) != len(image_info):
        raise ValueError("Text and image outputs have different sample counts.")

    results = []
    for text_sample, image_sample in zip(text_info, image_info):
        text_user, text_target, text_items, text_scores = text_sample
        image_user, _, image_items, image_scores = image_sample
        if text_user != image_user:
            raise ValueError(f"Text and image outputs are misaligned at user {text_user}.")

        text_probs = get_candidate_distribution(text_items, text_scores, args.text_temperature)
        image_probs = get_candidate_distribution(image_items, image_scores, args.image_temperature)

        text_weight = args.text_weight
        if not text_probs:
            text_weight = 0.0
        elif not image_probs:
            text_weight = 1.0
        elif args.fusion == "entropy" and min(len(text_probs), len(image_probs)) > 1:
            entropy_gap = normalized_entropy(image_probs) - normalized_entropy(text_probs)
            text_weight = min(1.0, max(0.0, text_weight + args.entropy_strength * entropy_gap))

        fused_scores = defaultdict(float)
        for item_id, probability in text_probs.items():
            fused_scores[item_id] += text_weight * probability
        for item_id, probability in image_probs.items():
            fused_scores[item_id] += (1.0 - text_weight) * probability

        ranked_items = sorted(fused_scores, key=fused_scores.__getitem__, reverse=True)
        results.append([int(item_id in text_target) for item_id in ranked_items])

    return results


def main(args):
    metrics = args.metrics.split(",")

    text_save_file = os.path.join(args.output_dir, f'save_seqrec_{args.num_beams}.json')
    text_info = json.load(open(text_save_file, 'r'))

    text_outputs = text_info['all_outputs'] 
    text_scores = text_info['all_scores'] 
    text_targets = text_info['all_targets']
    text_users = text_info['all_users'] 

    topk_res = get_topk_results(text_outputs, text_scores, text_targets, args.num_beams)
    metrics_results = get_metrics_results(topk_res, metrics)

    total = len(text_targets)

    for m in metrics_results:
        metrics_results[m] = metrics_results[m] / total
        
    print(metrics_results)

    #
    image_save_file = os.path.join(args.output_dir, f'save_seqimage_{args.num_beams}.json')
    image_info = json.load(open(image_save_file, 'r'))

    image_outputs = image_info['all_outputs'] 
    image_scores = image_info['all_scores'] 
    image_targets = image_info['all_targets'] 
    image_users = image_info['all_users'] 

    topk_res = get_topk_results(image_outputs, image_scores, image_targets, args.num_beams)
    metrics_results = get_metrics_results(topk_res, metrics)

    total = len(image_targets)

    for m in metrics_results:
        metrics_results[m] = metrics_results[m] / total
        
    print(metrics_results)




    index_text = os.path.join(args.data_path, args.dataset, f'{args.dataset}{args.index_file}')
    index_image = os.path.join(args.data_path, args.dataset, f'{args.dataset}{args.image_index_file}')

    item_id2text_index = json.load(open(index_text, 'r'))
    text_index2item_id = defaultdict(list)
    for item_id, text_index in item_id2text_index.items():
        text_index = ''.join(text_index)
        text_index2item_id[text_index].append(int(item_id)) 
        
    print(len(text_index2item_id))
        
    text_targets_ids = []
    for text_target in text_targets:

        text_targets_ids.append(text_index2item_id[text_target])

    ### 
    item_id2image_index = json.load(open(index_image, 'r'))
    image_index2item_id = defaultdict(list)
    for item_id, image_index in item_id2image_index.items():
        image_index = ''.join(image_index)
        image_index2item_id[image_index].append(int(item_id)) 
        
    print(len(image_index2item_id))
        
    image_targets_ids = []
    for image_target in image_targets:
        image_targets_ids.append(image_index2item_id[image_target])
        



    text_info = get_sort_results(text_outputs, text_scores, text_targets_ids, text_users, args.num_beams, text_index2item_id)
    image_info = get_sort_results(image_outputs, image_scores, image_targets_ids, image_users, args.num_beams, image_index2item_id)

    print('text info: ', len(text_info))
    print('image info: ', len(image_info))

    if args.fusion == "legacy":
        topk_res = get_topk_results_ensemble(text_info, image_info)
        result_name = f"results_ensemble_{args.num_beams}.json"
    else:
        topk_res = get_topk_results_weighted(text_info, image_info, args)
        result_name = f"results_ensemble_{args.fusion}_{args.num_beams}.json"
    metrics_results = get_metrics_results(topk_res, metrics)

    total = len(image_targets)

    for m in metrics_results:
        metrics_results[m] = metrics_results[m] / total
        
    print(metrics_results)

    print(total)
    
    save_file = os.path.join(args.output_dir, result_name)
    json.dump(metrics_results, open(save_file, 'w'), indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Index")

    parser.add_argument("--data_path", type=str, default="/userhome/dataset/LC-Rec_images", help="Input data path.")
    parser.add_argument("--dataset", type=str, default="Instruments", help="Input data path.")
    
    parser.add_argument("--output_dir", type=str, default="/userhome/projects/TIGER_image/log/encoder-decoder/Instruments/ckpt_b256_lr0.0005_wd0.01_dm0_e200_index_lemb_256_dis_seqrec,seqimage,item2image,image2item,fusionseqrec", help="Input data path.")
    
    parser.add_argument("--index_file", type=str, default=".index_lemb_256_dis.json", help="Input data path.")
    parser.add_argument("--image_index_file", type=str, default=".index_vitemb_256_dis.json", help="Input data path.")
    
    parser.add_argument("--metrics", type=str, default="hit@1,hit@5,hit@10,ndcg@5,ndcg@10", help="Input data path.")
    parser.add_argument("--num_beams", type=int, default=20, help="Input data path.")

    parser.add_argument("--fusion", choices=["legacy", "fixed", "entropy"], default="legacy")
    parser.add_argument("--text_weight", type=float, default=0.5, help="Base text fusion weight.")
    parser.add_argument("--entropy_strength", type=float, default=0.3, help="Weight adjustment per unit of normalized entropy difference.")
    parser.add_argument("--text_temperature", type=float, default=1.0)
    parser.add_argument("--image_temperature", type=float, default=1.0)

    args = parser.parse_args()
    if not 0.0 <= args.text_weight <= 1.0:
        parser.error("--text_weight must be between 0 and 1.")
    if not math.isfinite(args.entropy_strength) or args.entropy_strength < 0:
        parser.error("--entropy_strength must be finite and nonnegative.")
    for name in ("text_temperature", "image_temperature"):
        temperature = getattr(args, name)
        if not math.isfinite(temperature) or temperature <= 0:
            parser.error(f"--{name} must be finite and positive.")
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)



