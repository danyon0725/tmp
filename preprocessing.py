from bs4 import BeautifulSoup
import os
import json
import numpy as np
import re

def index_match(origin_context, context, origin_answer, answer, origin_answer_start):
    answer_start_positions = context.count(answer)
    if answer_start_positions > 1:
        p = answer.replace('[', '\[').replace(']', '\]').replace(')', '\)').replace('(', '\(').replace('*','\*').replace(".", "\.").replace('+', '\+').replace("$", "\$")
        answer_starts = [m.start(0) for m in re.finditer(p, context)]

        origin_answer = origin_answer.replace(" ", " ").replace('[', '\[').replace(']', '\]').replace(')','\)').replace('(', '\(').replace(".", "\.").replace('*', '\*').replace('+', '\+').replace("$", "\$")
        origin_answer_starts = [m.start(0) for m in re.finditer(origin_answer, origin_context.replace(" ", " "))]
        try:
            refine_answer_start = answer_starts[origin_answer_starts.index(origin_answer_start)]
        except:
            return -1

    else:
        try:
            refine_answer_start = context.index(answer)
        except:
            return -1
    return refine_answer_start

def sent_tokenizer(sent):
    sent_list = '|'.join(['. | '.join(e.strip().split('. ')) for e in sent.split('|')]).split('|')
    tmp = []
    for s in sent_list:
        tmp.append(s)
    result = []
    result = ' | '.join(tmp)
    return result
def tag_replace(s):
    result = s.replace(" ", " ").replace("]", "] | ").replace("<table>", " | [table] ").replace("</table>"," [/table] | ").replace("<ul>", " | [list] ").replace("</ul>", " [/list] | ").replace("<ol>", " | [list] ").replace("</ol>", " [/list] | ")
    return result
def tag_replace_2(s):
    result = s.replace(" ", " ").replace("]", "] | ").replace("<table>", " | [table] ").replace("</table>"," [/table] | ").replace("<ul>", " | [list] ").replace("</ul>", " [/list] | ").replace("<ol>", " | [list] ").replace("</ol>", " [/list] | ").replace("<td>","[td]").replace("</td>","[/td]").replace("<tr>","[tr]").replace("</td>","[tr]").replace("<th>","[th]").replace("</th>","[/th]").replace("<li>","[li]").replace("</li>","[/li]").replace("<dl>","[dl]").replace("</dl>","[/dl]").replace("<dd>","[dd]").replace("</dd>","[/dd]")
    return result
lengths = []
origin_length = []
max_len = 0
for data_type in ["dev", "train"]:
    data_dir = "./data/json/origin" + data_type
    out_dir = "./data/json/baseline/" + data_type
    f_list = os.listdir(data_dir)
    print(f_list)
    for f_name in f_list:
        print(f_name)
        num = 0
        c, m = 0, 0
        with open(os.path.join(data_dir, f_name),'r',encoding='utf8') as infile, open(os.path.join(out_dir, f_name),'w',encoding='utf8') as outfile:
            data_dict = json.load(infile)
            result_dict = {"data":[], "version":"KorQuAD_v2.0_" + data_type}
            for document in data_dict["data"]:

                title = BeautifulSoup(document["title"], "lxml").get_text()

                # context = BeautifulSoup(tag_replace(document["context"]), "lxml").get_text().strip()
                context = BeautifulSoup(tag_replace_2(document["context"]), "lxml").get_text().strip()
                context = re.sub('[\n\s]+', ' ', context)
                context = sent_tokenizer(context)
                context = re.sub('[\n\s]+', ' ', context)

                tmp_context = context.replace(" | | ", " | ")
                tmp_context = '| '+tmp_context.strip()
                p = "\|"
                sep_positions = list([m.start(0) for m in re.finditer(p, tmp_context)])

                context_sent = {e:tmp_context[sep_positions[idx]:sep_positions[idx+1]] for idx, e in enumerate(sep_positions) if idx +1 < len(sep_positions)}
                # split_context = [e.strip() for e in context.split("<h2>")]
                context_dict = {}
                paragraph_dict = {"context": tmp_context,"context_sent":context_sent, "qas": []}
                for qas in document["qas"]:
                    question = BeautifulSoup(qas['question'].replace(" ", " "), "lxml").get_text()
                    id = qas["id"]

                    a = qas["answer"]["text"].replace(" ", " ")
                    answer = BeautifulSoup(tag_replace_2(qas["answer"]["text"]), "lxml").get_text()
                    answer = re.sub('[\n\s]+', ' ', answer)
                    tmp_answers = sent_tokenizer(answer)
                    tmp_answers = re.sub('[\n\s]+', ' ', tmp_answers)

                    tmp_answers = tmp_answers.replace(" | | ", " | ")

                    if tmp_answers not in tmp_context:
                        num+=1
                        continue

                    if len(answer) > 50 and len(sent_tokenizer(tmp_answers).split('|')) > 1:
                        question_type = 'narrative'
                    else:
                        question_type = 'factoid'
                    if 'table' in a or '<td>' in a:
                        answer_type = 'table'
                    elif "li" in a:
                        answer_type = 'list'
                    else:
                        answer_type = 'sentence'
                    if question_type == 'factoid':
                        refine_answer_start = index_match(document['context'], tmp_context, qas["answer"]["text"], answer, qas["answer"]["answer_start"])
                    else:
                        answer_start = tmp_context.index(tmp_answers.strip())
                        answer_end = answer_start+len(tmp_answers)
                        start, end = 0, 0
                        for idx in range(len(sep_positions)):
                            if idx == len(sep_positions)-1:
                                break
                            cur_start, cur_end = sep_positions[idx], sep_positions[idx+1]
                            if cur_start <= answer_start and answer_start < cur_end:
                                start = idx
                            if cur_start <= answer_end and answer_end < cur_end:
                                end = idx
                                if answer_end-sep_positions[idx] < sep_positions[idx+1]-answer_end:
                                    end = idx -1
                        if start > end:
                            print("?????")
                        refine_answer_start = sep_positions[start:end+1]
                        tmp_answers = ''.join([context_sent[e] for e in refine_answer_start])
                    tmp_answers = tmp_answers.strip()
                    if tmp_answers[-2:] == ' |':
                        tmp_answers = tmp_answers[:-2]
                    num_sent = tmp_answers.count("|")
                    if num_sent > max_len:
                        max_len = num_sent
                    if  not tmp_answers:
                        continue
                    qas_dict = {"question": question, "id": id, "question_type":question_type,"answers": [{"text": tmp_answers, "answer_start": refine_answer_start, "answer_type":answer_type}]}
                    if tmp_context not in context_dict.keys():
                        context_dict[tmp_context] = []
                    paragraph_dict["qas"].append(qas_dict)
                if not paragraph_dict["qas"]:
                    continue
                document_dict = {"title": title, "paragraphs": []}
                document_dict["paragraphs"].append(paragraph_dict)
                result_dict["data"].append(document_dict)
            json.dump(result_dict, outfile, indent='\t', ensure_ascii=False)
        print(num)
