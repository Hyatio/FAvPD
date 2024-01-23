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
    BertConfig, BertTokenizer, BertForMaskedLM, 
    RobertaConfig, RobertaTokenizer, RobertaForMaskedLM,
    get_linear_schedule_with_warmup
)
from process.datasets.AdaptingProcessor import AdaptingProcessor
from process.datasets.TypingProcessor import TypingProcessor
from process.datasets.REProcessor import REProcessor
from process.datasets.NERProcessor import FewNERDProcessor
from process.metrics.mlm_metric import logme_score
import wandb

EVAL_PROCESSOR_CLASS = {
    'OpenEntity': TypingProcessor,
    'FIGER': TypingProcessor,
    'TACRED': REProcessor,
    'FewRel': REProcessor,
    'CHEMPROT': REProcessor,
    'FewNERD': FewNERDProcessor
}

MODEL_CLASS = {
    'bert': (BertConfig, BertTokenizer, BertForMaskedLM),
    'roberta': (RobertaConfig, RobertaTokenizer, RobertaForMaskedLM)
}

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_type", default=None, type=str, required=True, help="prompt type")
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
    parser.add_argument("--mlm_probability", type=float, default=0.15, required=True, help="ratio of tokens to mask for MLM")
    parser.add_argument("--k_mlm_probability", type=float, default=0.15, required=True, help="ratio of tokens(knowledge) to mask for MLM")
    parser.add_argument("--t_loss_weight", default=1.0, type=float, required=True, help="loss weight distribution for text")
    parser.add_argument("--k_loss_weight", default=1.0, type=float, required=True, help="loss weight distribution for knowledge")
    parser.add_argument("--l_loss_weight", default=1.0, type=float, required=True, help="loss weight distribution for label")
    parser.add_argument("--dynamic_mask", default=False, action='store_true', help="dynamically random mask a sentence each epoch")
    parser.add_argument("--add_entity_embedding", default=False, action='store_true', help="add entity embeddings for classification")
    parser.add_argument("--do_lower_case", default=False, action='store_true', help="do lower case for a uncased model")
    parser.add_argument("--save_checkpoint", default=False, action='store_true', help="save checkpoint when evaluating")
    return parser.parse_args()

def wandbconfig(args):
    wandb.config.seed = args.seed

def evaluate(model, eval_dataloader, args):
    features, targets = None, None
    for sample in tqdm(eval_dataloader, desc="Eval Iteration"):
        model.eval()
        input_ids = sample[0].to(args.device)
        attention_masks = sample[1].to(args.device)
        labels = sample[2].to(args.device)
        if args.task_name == 'OpenEntity' or args.task_name == 'FIGER':
            ent_masks = (sample[3].to(args.device),)
        elif args.task_name == 'TACRED' or args.task_name == 'FewRel' or args.task_name == 'CHEMPROT':
            ent_masks = (sample[3].to(args.device), sample[4].to(args.device),)
        elif args.task_name=='FewNERD':
            ent_masks = labels
        else:
            raise ValueError("wrong task name")
        if not args.add_entity_embedding and args.task_name != 'FewNERD':
            ent_masks = ()
        with torch.no_grad():
            feature, ner_labels = model(input_ids=input_ids, attention_mask=attention_masks, ent_masks=ent_masks, device=args.device)
        if args.task_name=='FewNERD':
            labels = ner_labels
        if features is None:
            features = feature.detach().cpu().numpy()
            targets = labels.detach().cpu().numpy()
        else:
            features = np.append(features, feature.detach().cpu().numpy(), axis=0)
            targets = np.append(targets, labels.detach().cpu().numpy(), axis=0)
    return logme_score(args, features, targets)

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

    # TACREV
    os.makedirs(args.output_dir+'_VV', exist_ok=True)
    
    config_class, tokenizer_class, model_class = MODEL_CLASS[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_path, do_lower_case=args.do_lower_case) \
        if args.model_type == 'bert' else tokenizer_class.from_pretrained(args.model_path)
    config = config_class.from_pretrained(args.model_path)
    model = model_class.from_pretrained(args.model_path, config=config)
    model.to(device)

    train_processor = AdaptingProcessor(args, tokenizer)
    train_dataloader = train_processor.create_dataloader()
    eval_processor = EVAL_PROCESSOR_CLASS[args.task_name](args, tokenizer)
    _, label_list = eval_processor.load_dataset('train')
    args.label_list = label_list
    eval_dataloader = eval_processor.create_dataloader(args.eval_set)

    # TACREV
    args.data_dir = "../DATA/dataset/TACREV"
    eval_processor_2 = EVAL_PROCESSOR_CLASS[args.task_name](args, tokenizer)
    _, label_list = eval_processor_2.load_dataset('train')
    args.label_list = label_list
    eval_dataloader_2 = eval_processor.create_dataloader(args.eval_set)
    args.data_dir = "../DATA/dataset/ReTACRED"
    

    num_train_steps = int(len(train_dataloader) / args.gradient_accumulation_steps * args.train_epoch)
    num_warmup_steps = num_train_steps * args.warmup_proportion
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(args.adam_beta_1, args.adam_beta_2), eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)

    global_step = 0
    best_step = None
    best_score = .0
    cnt_patience = 0

    # TACREV
    best_step_2 = None
    best_score_2 = .0
    cnt_patience_2 = 0

    # logme = evaluate(model, eval_dataloader, args)
    # print(logme)
    # exit()
    
    for _ in trange(int(args.train_epoch), desc="Epoch"):
        train_loss, train_steps = .0, 0
        for step, sample in enumerate(tqdm(train_dataloader, desc="Train Iteration")):
            model.train()
            attention_masks = sample[1].to(device)
            pos_ids = sample[2].to(device)
            vms = sample[4].to(device)
            label_labels = sample[5].to(device)
            loss_weight = (
                args.t_loss_weight,
                args.k_loss_weight,
                args.l_loss_weight
            )
            if args.dynamic_mask:
                input_ids = sample[0]
                label_tips = sample[3]
                dii, dli_t, dli_k, dli_l = train_processor.mask_tokens(input_ids, label_tips)
                dynamic_input_ids = dii.to(device)
                dynamic_t_label_ids = dli_t.to(device)
                dynamic_k_label_ids = dli_k.to(device)
                dynamic_l_label_ids = dli_l.to(device) if int(dli_l.sum()) != 0 \
                        else torch.zeros(args.max_seq_length).to(device)
                outputs = model(
                    input_ids=dynamic_input_ids,
                    attention_mask=attention_masks,
                    position_ids=pos_ids,
                    vms=vms,
                    t_labels=dynamic_t_label_ids,
                    k_labels=dynamic_k_label_ids,
                    l_labels=dynamic_l_label_ids,
                    label_labels=label_labels,
                    loss_weight = loss_weight
                )
            else:
                static_input_ids = sample[6].to(device)
                static_t_label_ids = sample[7].to(device)
                static_k_label_ids = sample[8].to(device)
                static_l_label_ids = sample[9].to(device) if int(sample[9].sum()) != 0 \
                        else torch.zeros(args.max_seq_length).to(device)
                outputs = model(
                    input_ids=static_input_ids,
                    attention_mask=attention_masks,
                    position_ids=pos_ids,
                    vms=vms,
                    t_labels=static_t_label_ids,
                    k_labels=static_k_label_ids,
                    l_labels=static_l_label_ids,
                    label_labels=label_labels,
                    loss_weight=loss_weight
                )
            
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
                optimizer.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logme = evaluate(model, eval_dataloader, args)
                    if logme > best_score:
                        best_score = logme
                        best_step = global_step
                        cnt_patience = 0
                    else:
                        cnt_patience += 1
                    if os.path.split(args.output_dir)[-1] != 'T':
                        with open(args.output_dir+'/result.txt', 'a') as f:
                            f.write('Step: {} LogME Score: {}\n'.format(global_step, logme))
                    if args.save_checkpoint:
                        output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model.save_pretrained(output_dir)
                        tokenizer.save_vocabulary(save_directory=output_dir)
                    if args.wandb_project:
                        wandb.log({'LogME Score': logme})
                    
                    # TACREV
                    logme = evaluate(model, eval_dataloader_2, args)
                    if logme > best_score_2:
                        best_score_2 = logme
                        best_step_2 = global_step
                        cnt_patience_2 = 0
                    else:
                        cnt_patience_2 += 1
                    if os.path.split(args.output_dir)[-1] != 'T':
                        with open(args.output_dir+'/result.txt', 'a') as f:
                            f.write('Step: {} LogME Score: {}\n'.format(global_step, logme))
                    if args.save_checkpoint:
                        output_dir = os.path.join(args.output_dir+'_VV', 'checkpoint-{}'.format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model.save_pretrained(output_dir)
                        tokenizer.save_vocabulary(save_directory=output_dir)
                    if args.wandb_project:
                        wandb.log({'LogME Score': logme})

        train_loss /= train_steps
        if args.wandb_project:
            wandb.log({'train loss': train_loss})

        # TACREV
        if cnt_patience >= args.earlystop_patience and cnt_patience_2 >= args.earlystop_patience:
            break

    print('best step: ', best_step)
    print('best score: ', best_score)
    print('best step 2: ', best_step_2)
    print('best score 2: ', best_score_2)
    with open(args.output_dir+'/args.txt', 'w') as f:
        f.write(str(args))

if __name__ == "__main__":
    main()