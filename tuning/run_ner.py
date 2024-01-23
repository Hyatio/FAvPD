import os
import sys
sys.path.append('.')
import argparse
import random
import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm, trange
from transformers import (
    BertConfig, BertTokenizer, BertForTokenClassification, 
    RobertaConfig, RobertaTokenizer, RobertaForTokenClassification,
    get_linear_schedule_with_warmup
)
from process.datasets.NERProcessor import FewNERDProcessor, BC5CDRProcessor
from process.metrics.ner_metric import compute_metric, align_predictions
import wandb

DATA_PROCESSOR_CLASS = {
    'FewNERD': FewNERDProcessor,
    'BC5CDR': BC5CDRProcessor
}

MODEL_CLASS = {
    'bert': (BertConfig, BertTokenizer, BertForTokenClassification),
    'roberta': (RobertaConfig, RobertaTokenizer, RobertaForTokenClassification)
}

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default=None, type=str, required=True, help="the name of dataset")
    parser.add_argument("--data_dir", default=None, type=str, required=True, help="the dataset path")
    parser.add_argument("--model_type", default=None, type=str, required=True, help="the model architecture, e.g. bert")
    parser.add_argument("--model_path", default=None, type=str, required=True, help="the model path")
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="the output path for evaluation and checkpoint")
    parser.add_argument("--eval_set", default='dev', type=str, required=True, help="the split for evaluation")
    parser.add_argument("--wandb_project", default=None, type=str, required=True, help="wandb project")
    parser.add_argument("--wandb_name", default=None, type=str, required=True, help="wandb name")
    parser.add_argument('--seed', type=int, default=2023, required=True, help="random seed")
    parser.add_argument("--train_epoch", default=4, type=int, required=True, help="train epoch")
    parser.add_argument("--train_batch_size", default=4, type=int, required=True, help="batch size for training")
    parser.add_argument("--eval_batch_size", default=4, type=int, required=True, help="batch size for evaluation")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, required=True, help="gradient accumulation for backpropagation")
    parser.add_argument("--learning_rate", default=1e-4, type=float, required=True, help="learning rate")
    parser.add_argument("--adam_beta_1", default=0.9, type=float, required=True, help="adam beta 1")
    parser.add_argument("--adam_beta_2", default=0.999, type=float, required=True, help="adam beta 2")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, required=True, help="adam epsilon")
    parser.add_argument("--warmup_proportion", default=0.0, type=float, required=True, help="proportion for warmup steps")
    parser.add_argument("--weight_decay", default=0.0, type=float, required=True, help="weight decay")
    parser.add_argument("--max_grad_norm", default=0.0, type=float, required=True, help="max gradient norm")
    parser.add_argument("--max_seq_length", default=512, type=int, required=True, help="max sequence length after tokenization.")
    parser.add_argument('--logging_steps', type=int, default=0, required=True, help="logging steps")
    parser.add_argument("--earlystop_patience", default=4, type=int, required=True, help="the patience of earlystop")
    parser.add_argument("--do_lower_case", default=False, action='store_true', help="do lower case for a uncased model")
    parser.add_argument("--save_checkpoint", default=False, action='store_true', help="save checkpoint when evaluating")
    return parser.parse_args()

def wandbconfig(args):
    wandb.config.seed = args.seed

def evaluate(model, eval_dataloader, args):
    preds, golds = None, None
    eval_steps, eval_loss= 0, .0
    for sample in tqdm(eval_dataloader, desc="Eval Iteration"):
        model.eval()
        input_ids = sample[0].to(args.device)
        attention_mask = sample[1].to(args.device)
        labels = sample[2].to(args.device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        tmp_loss = outputs[0]
        eval_loss += tmp_loss.mean().item()
        eval_steps += 1
        logits = outputs[1]
        if preds is None:
            golds = labels.detach().cpu().numpy()
            preds = logits.detach().cpu().numpy()
        else:
            golds = np.append(golds, labels.detach().cpu().numpy(), axis=0)
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
    eval_loss = eval_loss / eval_steps
    preds = np.argmax(preds, axis=2)
    golds, preds = align_predictions(golds, preds, args.label_list)
    micro_p, micro_r, micro_f1, accuracy = compute_metric(golds, preds, args)
    result = {
        "accuracy": accuracy,
        "micro-p": micro_p,
        "micro-r": micro_r,
        "micro-f1": micro_f1
    }
    return eval_loss, result

def main():
    args = argparser()
    if args.wandb_project:
        wandb.init(project=args.wandb_project, name=args.wandb_name)
        wandbconfig(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    args.device = device

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    config_class, tokenizer_class, model_class = MODEL_CLASS[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_path, do_lower_case=args.do_lower_case) \
        if args.model_type == 'bert' else tokenizer_class.from_pretrained(args.model_path)
    data_processor_class = DATA_PROCESSOR_CLASS[args.task_name]
    data_processor = data_processor_class(args, tokenizer)
    trainset, label_list = data_processor.load_dataset('train')
    args.label_list = label_list
    config = config_class.from_pretrained(args.model_path)
    config.num_labels=len(label_list)
    model = model_class.from_pretrained(args.model_path, config=config)
    model.to(device)

    num_train_steps = int(len(trainset) / args.train_batch_size / args.gradient_accumulation_steps * args.train_epoch)
    num_warmup_steps = num_train_steps * args.warmup_proportion
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(args.adam_beta_1, args.adam_beta_2), eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)

    train_dataloader = data_processor.create_dataloader('train')
    eval_dataloader = data_processor.create_dataloader(args.eval_set)

    global_step = 0
    best_step = None
    best_score = .0
    cnt_patience = 0

    for _ in trange(args.train_epoch, desc="Epoch"):
        train_loss, train_steps = .0, 0
        for step, sample in enumerate(tqdm(train_dataloader, desc="Train Iteration")):
            model.train()
            input_ids = sample[0].to(args.device)
            attention_mask = sample[1].to(args.device)
            labels = sample[2].to(args.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            train_loss += loss.item()
            train_steps += 1

            if (step+1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    eval_loss, result = evaluate(model, eval_dataloader, args)
                    if os.path.split(args.output_dir)[-1] != 'T':
                        with open(os.path.join(args.output_dir, 'result.txt'), 'a') as f:
                            f.write('Step: {} EVAL LOSS: {} EVAL_RESULT: {}\n'.format(global_step, eval_loss, result))
                    if result['micro-f1'] > best_score:
                        best_score = result['micro-f1']
                        best_step = global_step
                        cnt_patience = 0
                    else:
                        cnt_patience += 1
                    if args.save_checkpoint:
                        output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model.save_pretrained(output_dir)
                        tokenizer.save_vocabulary(output_dir)
                    if args.wandb_project:
                        wandb.log({'eval loss': eval_loss,
                                    'micro-p': result['micro-p'],
                                    'micro-r': result['micro-r'],
                                    'micro-f1': result['micro-f1'],
                                    'accuracy': result['accuracy']})     
        train_loss /= train_steps
        if args.wandb_project:
            wandb.log({'train loss': train_loss}) 
        if cnt_patience >= args.earlystop_patience:
            break
    if args.save_checkpoint:
        model.save_pretrained(args.output_dir)
        tokenizer.save_vocabulary(args.output_dir)
    print('best step: ', best_step)
    print('best score: ', best_score)
    with open(args.output_dir+'/args.txt', 'w') as f:
        f.write(str(args))

if __name__ == "__main__":
    main()