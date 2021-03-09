from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
import os
from bs4 import BeautifulSoup

'''KorQuAD 2.0에 대한 공식 평가 스크립트 '''
'''본 스크립트는 SQuAD v1.1 평가 스크립트 https://rajpurkar.github.io/SQuAD-explorer/ 를 바탕으로 작성됨.'''
from transformers.tokenization_electra import ElectraTokenizer
tokenizer = ElectraTokenizer.from_pretrained(
        "../../baseline/checkpoint-24000",
        do_lower_case=False,
    )

def normalize_answer(s):
    def tag_clean(t):
        return BeautifulSoup(t).get_text()

    def remove_(text):
        ''' 불필요한 기호 제거 '''
        text = re.sub("'", " ", text)
        text = re.sub('"', " ", text)
        text = re.sub('《', " ", text)
        text = re.sub('》', " ", text)
        text = re.sub('<', " ", text)
        text = re.sub('>', " ", text)
        text = re.sub('〈', " ", text)
        text = re.sub('〉', " ", text)
        text = re.sub("\(", " ", text)
        text = re.sub("\)", " ", text)
        text = re.sub("‘", " ", text)
        text = re.sub("’", " ", text)
        return text

    def white_space_fix(text):
        return ' '.join(text.split()).replace('\n', '').replace('\t', '').replace(' ', '')

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(remove_(tag_clean(s)))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    # F1 by character
    prediction_Char = []
    for tok in prediction_tokens:
        now = [a for a in tok]
        prediction_Char.extend(now)

    ground_truth_Char = []
    for tok in ground_truth_tokens:
        now = [a for a in tok]
        ground_truth_Char.extend(now)

    common = Counter(prediction_Char) & Counter(ground_truth_Char)
    num_same = sum(common.values())
    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_Char)
    recall = 1.0 * num_same / len(ground_truth_Char)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def eval_during_train_2(args, prefix):
    expected_version = 'KorQuAD_v2.0'

    dataset_file = os.path.join(args.dev_file_path, "korquad2.1_dev_{}.json".format("02"))
    # dataset_file = os.path.join(args.dev_file_path, "korquad2.1_dev_{}_preprocessed.json".format("02"))
    # dataset_file = os.path.join(args.data_dir, "korquad2.0/dev_sep/tmp.1_dev_08.json")
    prediction_file = os.path.join(args.output_dir, 'predictions_{}.json'.format(prefix))

    with open(dataset_file) as dataset_f:
        dataset_json = json.load(dataset_f)
        read_version = "_".join(dataset_json['version'].split("_")[:-1])
        if (read_version != expected_version):
            print('Evaluation expects ' + expected_version +
                  ', but got dataset with ' + read_version,
                  file=sys.stderr)
        dataset = dataset_json['data']
    with open(prediction_file) as prediction_f:
        predictions = json.load(prediction_f)

    return evaluate(dataset, predictions)
def eval_n_best(args, prefix):
    expected_version = 'KorQuAD_v2.0'

    dataset_file = os.path.join(args.dev_file_path, "korquad2.1_dev_{}.json".format("02"))
    # dataset_file = os.path.join(args.dev_file_path, "korquad2.1_dev_{}_preprocessed.json".format("02"))

    # dataset_file = os.path.join(args.data_dir, "korquad2.0/dev_sep/tmp.1_dev_08.json")
    # prediction_file = os.path.join(args.output_dir, 'predictions_{}.json'.format(prefix))
    nbest_prediction_file = os.path.join(args.output_dir, 'nbest_predictions_{}.json'.format(prefix))

    with open(dataset_file) as dataset_f:
        dataset_json = json.load(dataset_f)
        read_version = "_".join(dataset_json['version'].split("_")[:-1])
        if (read_version != expected_version):
            print('Evaluation expects ' + expected_version +
                  ', but got dataset with ' + read_version,
                  file=sys.stderr)
        dataset = dataset_json['data']
    # with open(prediction_file) as prediction_f:
    #     predictions = json.load(prediction_f)
    with open(nbest_prediction_file) as prediction_f:
        nbest_predictions = json.load(prediction_f)
    return nbest_evaluate(dataset, None, nbest_predictions)

def nbest_evaluate(dataset, predictions, nbest_predictions):
    num = 0
    pos = 0
    neg = 0
    f1 = exact_match = total = 0
    for document in dataset:
        for paragraph in document['paragraphs']:
            for qa in paragraph['qas']:

                # if qa['id'] not in predictions:
                #     message = 'Unanswered question ' + qa['id'] + \
                #               ' will receive score 0.'
                #     print(message, file=sys.stderr)
                #     continue
                answer_type = qa['answers'][0]['answer_type']
                if qa["question_type"] == 'narrative' and answer_type != 'sentence':
                    continue
                answers = [e['text'].replace("[table]", "").replace("[/table]", "").replace("[list]", "").replace("[/list]", "").replace("[tr]", "").replace("[/tr]", "").replace("[td]", "").replace("[/td]", "").replace("[th]", "").replace("[/th]", "").replace("[li]", "").replace("[/li]", "").replace("[dl]", "").replace("[/dl]", "").replace("[dd]", "").replace("[/dd]", "").replace("|", "").replace("[a]","").replace("[/a]","").replace("[p]","").replace("[/p]","").replace("[b]","").replace("[/b]","") for e in nbest_predictions[qa['id']][:2]]
                ground_truth = qa['answers'][0]['text'].replace("[table]", "").replace("[/table]", "").replace("[list]", "").replace("[/list]", "").replace("[tr]", "").replace("[/tr]", "").replace("[td]", "").replace("[/td]", "").replace("[th]", "").replace("[/th]", "").replace("[li]", "").replace("[/li]", "").replace("[dl]", "").replace("[/dl]", "").replace("[dd]", "").replace("[/dd]", "").replace("|", "").replace("[a]","").replace("[/a]","").replace("[p]","").replace("[/p]","").replace("[b]","").replace("[/b]","")
                # prediction = predictions[qa['id']].replace("[table]", "").replace("[/table]", "").replace("[list]", "").replace("[/list]", "").replace("[tr]", "").replace("[/tr]", "").replace("[td]", "").replace("[/td]", "").replace("[th]", "").replace("[/th]", "").replace("[li]", "").replace("[/li]", "").replace("[dl]", "").replace("[/dl]", "").replace("[dd]", "").replace("[/dd]", "").replace("|", "")
                f1s = [f1_score(answer, ground_truth) for answer in answers]
                ems = [exact_match_score(answer, ground_truth) for answer in answers]
                if f1s[0] < 0.3:
                    num+=1
                    if nbest_predictions[qa['id']][0]['is_answerable'] and nbest_predictions[qa['id']][1]['is_answerable']:
                        pos+=1

                        f1 += max(f1s)
                    else:
                        f1 += f1s[0]
                else:
                    f1 += f1s[0]
                exact_match += max(ems)
                # f1 += f1_score(prediction, ground_truth)
                total += 1
    print(total)
    print(pos/num)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}

def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    for document in dataset:
        for paragraph in document['paragraphs']:
            for qa in paragraph['qas']:

                # if qa['id'] not in predictions:
                #     message = 'Unanswered question ' + qa['id'] + \
                #               ' will receive score 0.'
                #     print(message, file=sys.stderr)
                #     continue
                answer_type = qa['answers'][0]['answer_type']
                if answer_type != 'list':
                    continue
                question = qa['question']
                ground_truth = qa['answers'][0]['text'].replace("[table]", "").replace("[/table]", "").replace("[list]", "").replace("[/list]", "").replace("[tr]", "").replace("[/tr]", "").replace("[td]", "").replace("[/td]", "").replace("[th]", "").replace("[/th]", "").replace("[li]", "").replace("[/li]", "").replace("[dl]", "").replace("[/dl]", "").replace("[dd]", "").replace("[/dd]", "").replace("|", "").replace("[a]","").replace("[/a]","").replace("[p]","").replace("[/p]","").replace("[b]","").replace("[/b]","")
                # if len(tokenizer.tokenize(question)) + len(tokenizer.tokenize(ground_truth)) + 3+128 > 512:
                #     continue
                prediction = predictions[qa['id']].replace("[table]", "").replace("[/table]", "").replace("[list]", "").replace("[/list]", "").replace("[tr]", "").replace("[/tr]", "").replace("[td]", "").replace("[/td]", "").replace("[th]", "").replace("[/th]", "").replace("[li]", "").replace("[/li]", "").replace("[dl]", "").replace("[/dl]", "").replace("[dd]", "").replace("[/dd]", "").replace("|", "").replace("[a]","").replace("[/a]","").replace("[p]","").replace("[/p]","").replace("[b]","").replace("[/b]","")

                exact_match += exact_match_score(prediction, ground_truth)
                f1 += f1_score(prediction, ground_truth)
                total += 1
    print(total)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}



from attrdict import AttrDict
if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--dev_file_path", default="../../data/json/baseline/dev", type=str, required=False)
    cli_parser.add_argument("--output_dir", type=str, default="../../baseline", required=False)
    cli_args = cli_parser.parse_args()
    result = eval_during_train_2(cli_args, '')
    print(result)
    # result = eval_n_best(cli_args, '')
    # print(result)