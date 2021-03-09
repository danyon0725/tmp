import os
import torch
import timeit
from fastprogress.fastprogress import master_bar, progress_bar
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
# from src.functions.squad_metric import (
#     compute_predictions_logits,
#     compute_predictions_logits_v2,
#     squad_evaluate
# )
from src.functions.squad_metric import (
    compute_predictions_logits,
    squad_evaluate
)
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

from src.functions.utils import load_examples, set_seed, to_list
from src.functions.processor import SquadResult
from src.functions.evaluate_v2_0 import eval_during_train_2

def prepro(args, tokenizer):
    train_json_file_list = os.listdir(args.train_file_path)
    dev_json_file_list = os.listdir(args.dev_file_path)

    train_json_file_list.sort()
    dev_json_file_list.sort()
    for f_name in dev_json_file_list:
        flag = int(f_name.split('.')[1][-2:])
        if flag != 2:
            continue
        print(f_name)
        load_examples(args, tokenizer, evaluate=True, do_cache=True, data=f_name)

    for f_name in train_json_file_list:
        print(f_name)
        #korquad2.1_train_38
        flag = int(f_name.split('.')[1][-2:])
        # if flag < 30:
        #     continue
        load_examples(args, tokenizer, evaluate=False, do_cache=True, data=f_name)


f1 = 0
def train(args, model, tokenizer, logger):
    global f1
    global_step = 1
    tr_loss, logging_loss = 0.0, 0.0
    mb = master_bar(range(int(args.num_train_epochs)))

    for epoch in mb:
        for dataset in os.listdir(args.train_cache_path):

            train_dataset = load_examples(args, tokenizer, evaluate=False, output_examples=False, data=dataset)
            """ Train the model """
            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

            if args.max_steps > 0:
                t_total = args.max_steps
                args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
            else:
                t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

            # Prepare optimizer and schedule (linear warmup and decay)
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": args.weight_decay,
                },
                {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
            )

            # Check if saved optimizer or scheduler states exist
            if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
                    os.path.join(args.model_name_or_path, "scheduler.pt")
            ):
                # Load in optimizer and scheduler states
                optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
                scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

            # Train!
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(train_dataset))
            logger.info("  Num Epochs = %d", args.num_train_epochs)
            logger.info("  Train batch size per GPU = %d", args.train_batch_size)
            logger.info(
                "  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size
                * args.gradient_accumulation_steps)
            logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
            logger.info("  Total optimization steps = %d", t_total)

            epochs_trained = 0
            steps_trained_in_current_epoch = 0
            # Check if continuing training from a checkpoint
            logger.info("  Starting fine-tuning.")

            model.zero_grad()
            # Added here for reproductibility
            set_seed(args)

            epoch_iterator = progress_bar(train_dataloader, parent=mb)
            for step, batch in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                model.train()
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "question_mask":batch[2],
                    "sentence_mask": batch[3],
                    "token_type_ids": batch[4],
                    "question_type_label" : batch[5],
                    "sent_start_positions": batch[6],
                    "sent_end_positions": batch[7],
                    "tok_start_positions": batch[8],
                    "tok_end_positions": batch[9],
                }

                loss, span_loss, tok_loss, qt_loss = model(**inputs)
                # model outputs are always tuple in transformers (see doc)
                if (global_step +1) % 50 == 0:
                    print("{} Processing,,,, Current Total Loss : {}".format(global_step+1, loss.item()))
                    print("Sent Span Loss : {}\tTok Span Loss : {}\tQuestion Type Loss : {}".format(span_loss.item(), tok_loss.item(), qt_loss.item()))

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % 1 == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    # # Log metrics
                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))


                        cur_f1 = evaluate(args=args, model=model, tokenizer=tokenizer, global_step=global_step)
                        if cur_f1 > f1:
                            f1 = cur_f1

                            # Take care of distributed/parallel training

                if args.max_steps > 0 and global_step > args.max_steps:
                    break

            mb.write("Epoch {} done".format(epoch+1))

            if args.max_steps > 0 and global_step > args.max_steps:
                break
        # evaluate(args, model, tokenizer, global_step=global_step, all_predict=True)

    return global_step, tr_loss / global_step
from src.functions.utils import f1_measure

question_type2idx = {"narrative":0, "factoid":1}
idx2question_type = {0:"narrative", 1:"factoid"}
answer_type2idx = {"sentence":0, "list":1, "table":2}

def evaluate(args, model, tokenizer, prefix="", global_step=None, all_predict=False, logger=None):
    f_list = os.listdir(args.dev_cache_path)
    f_list.sort()
    for data in f_list:
        qt_preds = []
        qt_labels = []
        qt_dict = {}
        print(data)
        if '02' not in data:
            continue
        # prefix = data.split('_')[4]
        dataset, examples, features = load_examples(args, tokenizer, evaluate=True, output_examples=True, data=data)
        for example in examples:
            qas_id = example.qas_id
            qt_label = example.answers[0]['question_type']

            qt_dict[qas_id] = question_type2idx[qt_label]
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)


        all_results = []
        start_time = timeit.default_timer()

        for batch in progress_bar(eval_dataloader):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "question_mask":batch[2],
                    "sentence_mask": batch[3],
                    "token_type_ids": batch[4],
                }

                example_indices = batch[5]

                outputs = model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]

                unique_id = int(eval_feature.unique_id)

                sent_start_logits, sent_end_logits, tok_start_logits, tok_end_logits, question_type_outputs = [to_list(output[i]) for output in outputs]

                qt_preds.append(question_type_outputs)
                # at_preds.append(answer_type_outputs)
                qt_labels.append(qt_dict[eval_feature.qas_id])
                # at_labels.append(at_dict[eval_feature.qas_id])
                result = SquadResult(unique_id, sent_start_logits, sent_end_logits, tok_start_logits, tok_end_logits, idx2question_type[question_type_outputs])

                all_results.append(result)
        f1_measure(qt_labels, qt_preds, 2)

        # Compute predictions
        output_prediction_file = os.path.join(args.output_dir, "predictions_.json")
        output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_.json")

        if args.version_2_with_negative:
            output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
        else:
            output_null_log_odds_file = None
        torch.save({"features": features, "results": all_results, "examples": examples}, args.result_file)
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            args.verbose_logging,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
            tokenizer,
        )

        # Write the evaluation result on file
        output_dir = os.path.join(args.output_dir, 'eval')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_eval_file = os.path.join(output_dir, "eval_result_{}_{}.txt".format(list(filter(None, args.model_name_or_path.split("/"))).pop(),
                                                                                   global_step))
        print()
        print(output_eval_file)

        with open(output_eval_file, "w", encoding='utf-8') as f:
            official_eval_results = eval_during_train_2(args, prefix)
            cur_f1 = official_eval_results['f1']
            for key in sorted(official_eval_results.keys()):
                f.write(" {} = {}\n".format(key, str(official_eval_results[key])))
                print(" {} = {}\n".format(key, str(official_eval_results[key])))
        if not all_predict:
            break
    return cur_f1
def only_scoring(args, tokenizer):
    results = torch.load(args.result_file)
    features, result, examples = (
        results["features"],
        results["results"],
        results["examples"],
    )
    output_prediction_file = os.path.join(args.output_dir, "predictions_1.json")
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_1.json")

    predictions = compute_predictions_logits(
            examples,
            features,
            result,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            None,
            args.verbose_logging,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
            tokenizer,
        )

    # Write the evaluation result on file
    output_dir = os.path.join(args.output_dir, 'eval')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    official_eval_results = eval_during_train_2(args, '1')
    cur_f1 = official_eval_results['f1']
    for key in sorted(official_eval_results.keys()):
        print(" {} = {}\n".format(key, str(official_eval_results[key])))