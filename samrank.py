import json
import argparse
import os
import sys
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer

import torch

from transformers import AutoTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2Model

from swisscom_ai.research_keyphrase.preprocessing.postagging import PosTaggingCoreNLP
from swisscom_ai.research_keyphrase.model.input_representation import InputTextObj
from swisscom_ai.research_keyphrase.model.extractor import extract_candidates


# Run "stanford-corenlp-full-2018-02-27" with terminal before run this code.
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos -status_port 9000 -port 9000 -timeout 15000 &

host = 'localhost'
port = 9000
pos_tagger = PosTaggingCoreNLP(host, port)

# load stopwords
stopwords = []
with open('UGIR_stopwords.txt', "r") as f:
    for line in f:
        if line:
            stopwords.append(line.replace('\n', ''))

stemmer = PorterStemmer()



def get_col_sum_token_level(attention_map):
    tokens_score = torch.sum(attention_map, dim=0)
    return tokens_score


def redistribute_global_attention_score(attention_map, tokens_score):
    new_attention_map = attention_map * tokens_score.unsqueeze(0)
    return new_attention_map


def normalize_attention_map(attention_map):
    attention_map_sum = attention_map.sum(dim=0, keepdim=True)
    attention_map_sum += 1e-10
    attention_map_normalized = attention_map / attention_map_sum
    return attention_map_normalized


def get_row_sum_token_level(attention_map):
    tokens_score = torch.sum(attention_map, dim=1)
    return tokens_score


def aggregate_phrase_scores(index_list, tokens_scores):
    total_score = 0.0

    for p_index in index_list:
        part_sum = tokens_scores[p_index[0]:p_index[1]].sum()
        total_score += part_sum

    return total_score


def get_phrase_indices(text_tokens, phrase, prefix):
    text_tokens = [t.replace(prefix, '') for t in text_tokens]

    phrase = phrase.replace(' ', '')

    matched_indices = []
    matched_index = []
    target = phrase
    for i in range(len(text_tokens)):
        cur_token = text_tokens[i]
        sub_len = min(len(cur_token), len(phrase))
        if cur_token[:sub_len].lower() == target[:sub_len]:
            matched_index.append(i)
            target = target[sub_len:]
            if len(target) == 0:
                matched_indices.append([matched_index[0], matched_index[-1] + 1])
                target = phrase
        else:
            matched_index = []
            target = phrase
            if cur_token[:sub_len].lower() == target[:sub_len]:
                matched_index.append(i)
                target = target[sub_len:]
                if len(target) == 0:
                    matched_indices.append([matched_index[0], matched_index[-1] + 1])
                    target = phrase

    return matched_indices


def remove_repeated_sub_word(candidates_pos_dict):
    for phrase in candidates_pos_dict.keys():
        split_phrase = phrase.split()
        if len(split_phrase) > 1:
            for word in split_phrase:
                if word in candidates_pos_dict:
                    single_word_positions = candidates_pos_dict[word]
                    phrase_positions = candidates_pos_dict[phrase]
                    single_word_alone_positions = [pos for pos in single_word_positions if not any(
                        pos[0] >= phrase_pos[0] and pos[1] <= phrase_pos[1] for phrase_pos in phrase_positions)]
                    candidates_pos_dict[word] = single_word_alone_positions

    return candidates_pos_dict


def get_same_len_segments(total_tokens_ids, max_len):
    num_of_seg = (len(total_tokens_ids) // max_len) + 1
    seg_len = int(len(total_tokens_ids) / num_of_seg)
    segments = []
    attn_masks = []
    for _ in range(num_of_seg):
        if len(total_tokens_ids) > seg_len:
            segment = total_tokens_ids[:seg_len]
            total_tokens_ids = total_tokens_ids[seg_len:]
        else:
            segment = total_tokens_ids
        segments.append(segment)
        attn_masks.append([1] * len(segments[-1]))

    return segments, attn_masks


def read_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    return data


def get_candidates(core_nlp, text):
    tagged = core_nlp.pos_tag_raw_text(text)
    text_obj = InputTextObj(tagged, 'en')
    candidates = extract_candidates(text_obj)
    return candidates


def get_score_full(candidates, references, maxDepth=15):
    precision = []
    recall = []
    reference_set = set(references)
    referencelen = len(reference_set)
    true_positive = 0
    for i in range(maxDepth):
        if len(candidates) > i:
            kp_pred = candidates[i]
            if kp_pred in reference_set:
                true_positive += 1
            precision.append(true_positive / float(i + 1))
            recall.append(true_positive / float(referencelen))
        else:
            precision.append(true_positive / float(len(candidates)))
            recall.append(true_positive / float(referencelen))
    return precision, recall


def evaluate(candidates, references):
    results = {}
    precision_scores, recall_scores, f1_scores = {5: [], 10: [], 15: []}, \
                                                 {5: [], 10: [], 15: []}, \
                                                 {5: [], 10: [], 15: []}
    for candidate, reference in zip(candidates, references):
        p, r = get_score_full(candidate, reference)
        for i in [5, 10, 15]:
            precision = p[i - 1]
            recall = r[i - 1]
            if precision + recall > 0:
                f1_scores[i].append((2 * (precision * recall)) / (precision + recall))
            else:
                f1_scores[i].append(0)
            precision_scores[i].append(precision)
            recall_scores[i].append(recall)

    print("########################\nMetrics")
    for i in precision_scores:
        print("@{}".format(i))
        print("F1:{}".format(np.mean(f1_scores[i])))
        print("P:{}".format(np.mean(precision_scores[i])))
        print("R:{}".format(np.mean(recall_scores[i])))

        top_n_p = 'precision@' + str(i)
        top_n_r = 'recall@' + str(i)
        top_n_f1 = 'f1@' + str(i)
        results[top_n_p] = np.mean(precision_scores[i])
        results[top_n_r] = np.mean(recall_scores[i])
        results[top_n_f1] = np.mean(f1_scores[i])
    print("#########################")

    return results


def evaluate_all_heads(layer_head_predicted_top15, dataset):
    experiment_results = []

    for (layer, head), predicted_phrases in layer_head_predicted_top15.items():
        gold_keyphrase_list = []
        predicted_keyphrase_list = []

        for i in range(len(dataset)):
            predicted_keyphrase = predicted_phrases[i]
            predicted_keyphrase = [phrase.lower() for phrase in predicted_keyphrase]
            predicted_keyphrase_list.append(predicted_keyphrase)

            gold_keyphrase = [key.lower() for key in dataset[i]['keyphrases']]
            gold_keyphrase_list.append(gold_keyphrase)

        print(f"{layer + 1}th layer {head + 1}th head result:")
        total_score = evaluate(predicted_keyphrase_list, gold_keyphrase_list)
        total_score['layer'] = layer + 1
        total_score['head'] = head + 1
        experiment_results.append(total_score)
        print()

    df = pd.DataFrame(experiment_results)

    path = f'experiment_results/{args.dataset}/'
    os.makedirs(path, exist_ok=True)
    df.to_csv(f'{path}{args.plm}_{args.mode}.csv', index=False)

    top3_f1_5 = df.nlargest(3, 'f1@5').reset_index(drop=True)
    top3_f1_10 = df.nlargest(3, 'f1@10').reset_index(drop=True)
    top3_f1_15 = df.nlargest(3, 'f1@15').reset_index(drop=True)

    return top3_f1_5, top3_f1_10, top3_f1_15


def rank_short_documents(args, dataset, model, tokenizer):
    if args.plm == 'BERT':
        prefix = '##'
    elif args.plm == 'GPT2':
        prefix = 'Ġ'

    layer_head_predicted_top15 = defaultdict(list)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model.to(device)
    model.eval()

    for data in tqdm(dataset):
        with torch.no_grad():
            tokenized_text = tokenizer(data['text'], return_tensors='pt')
            outputs = model(**tokenized_text.to(device))

            attentions = outputs.attentions

            candidates = get_candidates(pos_tagger, data['text'])
            candidates = [phrase for phrase in candidates if phrase.split(' ')[0] not in stopwords]

            text_tokens = tokenizer.convert_ids_to_tokens(tokenized_text['input_ids'].squeeze(0))

            candidates_indices = {}
            for phrase in candidates:
                matched_indices = get_phrase_indices(text_tokens, phrase, prefix)
                if len(matched_indices) == 0:
                    continue
                candidates_indices[phrase] = matched_indices

            candidates_indices = remove_repeated_sub_word(candidates_indices)

            for layer in range(12):
                for head in range(12):
                    n_layer_attentions = attentions[layer].squeeze(0)
                    attention_map = n_layer_attentions[head]

                    global_attention_scores = get_col_sum_token_level(attention_map)

                    if args.plm == "BERT":
                        global_attention_scores[-1] = 0
                    elif args.plm == "GPT2":
                        global_attention_scores[0] = 0

                    redistributed_attention_map = redistribute_global_attention_score(attention_map,
                                                                                      global_attention_scores)

                    redistributed_attention_map = normalize_attention_map(redistributed_attention_map)

                    proportional_attention_scores = get_row_sum_token_level(redistributed_attention_map)

                    if args.mode == 'Both':
                        final_tokens_score = global_attention_scores + proportional_attention_scores
                    elif args.mode == 'Global':
                        final_tokens_score = global_attention_scores
                    elif args.mode == 'Proportional':
                        final_tokens_score = proportional_attention_scores

                    phrase_score_dict = {}
                    for phrase in candidates_indices.keys():
                        try:
                            phrase_indices = candidates_indices[phrase]
                            if len(phrase_indices) == 0:
                                continue
                        except KeyError:
                            continue

                        final_phrase_score = aggregate_phrase_scores(phrase_indices, final_tokens_score)

                        if len(phrase.split()) == 1:
                            final_phrase_score = final_phrase_score / len(phrase_indices)
                        phrase_score_dict[phrase] = final_phrase_score

                    sorted_scores = sorted(phrase_score_dict.items(), key=lambda item: item[1], reverse=True)
                    stemmed_sorted_scores = [(" ".join(stemmer.stem(word) for word in phrase.split()), score) for
                                             phrase, score in sorted_scores]

                    set_stemmed_scores_list = []
                    for phrase, score in stemmed_sorted_scores:
                        if phrase not in set_stemmed_scores_list:
                            set_stemmed_scores_list.append(phrase)

                    pred_stemmed_phrases = set_stemmed_scores_list[:15]
                    layer_head_predicted_top15[(layer, head)].append(pred_stemmed_phrases)

    top3_f1_5, top3_f1_10, top3_f1_15 = evaluate_all_heads(layer_head_predicted_top15, dataset)

    print("top@5_f1  Top3 heads:")
    print(top3_f1_5[['f1@5', 'f1@10', 'f1@15', 'layer', 'head']].to_string(index=False))
    print("top@10_f1  Top3 heads:")
    print(top3_f1_10[['f1@5', 'f1@10', 'f1@15', 'layer', 'head']].to_string(index=False))
    print("top@15_f1  Top3 heads:")
    print(top3_f1_15[['f1@5', 'f1@10', 'f1@15', 'layer', 'head']].to_string(index=False))


def rank_long_documents(args, dataset, model, tokenizer):
    if args.plm == 'BERT':
        prefix = '##'
        max_len = 512
    elif args.plm == 'GPT2':
        prefix = 'Ġ'
        max_len = 1024

    layer_head_predicted_top15 = defaultdict(list)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model.to(device)
    model.eval()

    for data in tqdm(dataset):
        with torch.no_grad():
            tokenized_text = tokenizer(data['text'], return_tensors='pt')

            candidates = get_candidates(pos_tagger, data['text'])
            candidates = [phrase for phrase in candidates if phrase.split(' ')[0] not in stopwords]

            total_tokens_ids = tokenized_text['input_ids'].squeeze(0).tolist()
            if args.plm == 'BERT':
                total_tokens_ids = total_tokens_ids[1:-1]
                max_len = 512

            windows, attention_masks = get_same_len_segments(total_tokens_ids, max_len)

            layer_head_scores = defaultdict(lambda: defaultdict(float))

            for i in range(len(windows)):

                window = windows[i]
                attention_mask = attention_masks[i]

                if args.plm == 'BERT':
                    window = [101] + window + [102]
                    attention_mask = [1] + attention_mask + [1]

                window = torch.tensor([window])
                attention_mask = torch.tensor([attention_mask])
                # print(window.shape)

                outputs = model(window.to(device), attention_mask=attention_mask.to(device))
                attentions = outputs.attentions

                text_tokens = tokenizer.convert_ids_to_tokens(window[0])

                candidates_indices = {}
                for phrase in candidates:
                    matched_indices = get_phrase_indices(text_tokens, phrase, prefix)
                    if len(matched_indices) == 0:
                        continue
                    candidates_indices[phrase] = matched_indices

                candidates_indices = remove_repeated_sub_word(candidates_indices)

                for layer in range(12):
                    for head in range(12):
                        n_layer_attentions = attentions[layer].squeeze(0)
                        attention_map = n_layer_attentions[head]

                        global_attention_scores = get_col_sum_token_level(attention_map)

                        if args.plm == "BERT":
                            global_attention_scores[-1] = 0
                        elif args.plm == "GPT2":
                            global_attention_scores[0] = 0

                        redistributed_attention_map = redistribute_global_attention_score(attention_map,
                                                                                          global_attention_scores)

                        redistributed_attention_map = normalize_attention_map(redistributed_attention_map)

                        proportional_attention_scores = get_row_sum_token_level(redistributed_attention_map)

                        if args.mode == 'Both':
                            final_tokens_score = global_attention_scores + proportional_attention_scores
                        elif args.mode == 'Global':
                            final_tokens_score = global_attention_scores
                        elif args.mode == 'Proportional':
                            final_tokens_score = proportional_attention_scores

                        for phrase in candidates_indices.keys():
                            try:
                                phrase_indices = candidates_indices[phrase]
                                if len(phrase_indices) == 0:
                                    continue
                            except KeyError:
                                continue

                            final_phrase_score = aggregate_phrase_scores(phrase_indices, final_tokens_score)

                            if len(phrase.split()) == 1:
                                final_phrase_score = final_phrase_score / len(phrase_indices)

                            layer_head_scores[(layer, head)][phrase] += final_phrase_score

            for layer in range(12):
                for head in range(12):
                    scores = layer_head_scores[(layer, head)]

                    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
                    stemmed_scores = [(" ".join(stemmer.stem(word) for word in keyword.split()), score) for
                                      keyword, score in sorted_scores]

                    set_stemmed_scores_list = []
                    for phrase, score in stemmed_scores:
                        if phrase not in set_stemmed_scores_list:
                            set_stemmed_scores_list.append(phrase)

                    pred_stemmed_phrases = set_stemmed_scores_list[:15]
                    layer_head_predicted_top15[(layer, head)].append(pred_stemmed_phrases)

    top3_f1_5, top3_f1_10, top3_f1_15 = evaluate_all_heads(layer_head_predicted_top15, dataset)

    print("top@5_f1  Top3 heads:")
    print(top3_f1_5[['f1@5', 'f1@10', 'f1@15', 'layer', 'head']].to_string(index=False))
    print("top@10_f1  Top3 heads:")
    print(top3_f1_10[['f1@5', 'f1@10', 'f1@15', 'layer', 'head']].to_string(index=False))
    print("top@15_f1  Top3 heads:")
    print(top3_f1_15[['f1@5', 'f1@10', 'f1@15', 'layer', 'head']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default='Inspec',
                        type=str,
                        required=True,
                        help="Inspec or SemEval2010 or SemEval2017")

    parser.add_argument("--plm",
                        default='BERT',
                        type=str,
                        required=True,
                        help="BERT or GPT2")

    parser.add_argument("--mode",
                        default='Both',
                        type=str,
                        required=False,
                        help="Both or Global or Proportional")

    args = parser.parse_args()

    if args.dataset == 'Inspec' or args.dataset == 'inpsec':
        data_path = 'data/Inspec.jsonl'
        doc_type = 'short'
    elif args.dataset == 'SemEval2010' or args.dataset == 'semeval2010':
        data_path = 'data/SemEval2010.jsonl'
        doc_type = 'long'
    elif args.dataset == 'SemEval2017' or args.dataset == 'semeval2017':
        data_path = 'data/SemEval2017.jsonl'
        doc_type = 'short'
    elif args.dataset == 'Krapivin' or args.dataset == 'krapivin':
        data_path = 'data/krapivin.jsonl'
        doc_type = 'long'
    else:
        print('Invalid dataset')
        sys.exit(1)

    dataset = read_jsonl(data_path)

    if args.plm == 'BERT':
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True, add_pooling_layer=False)

    elif args.plm == 'GPT2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2Model.from_pretrained('gpt2', output_hidden_states=True, output_attentions=True)

    if doc_type == 'short':
        rank_short_documents(args, dataset, model, tokenizer)
    elif doc_type == 'long':
        rank_long_documents(args, dataset, model,tokenizer)
