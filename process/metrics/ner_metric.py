import torch.nn as nn
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score

def compute_metric(golds, preds, args, average='micro'):
    micro_p, micro_r, micro_f1 = 0, 0, 0
    macro_p, macro_r, macro_f1 = 0, 0, 0
    accuracy = 0
    if args.task_name == 'FewNERD':
        metric = Metrics()
        micro_p, micro_r, micro_f1, accuracy = metric.metrics_by_entity(preds, golds)
    elif args.task_name == 'BC5CDR':
        accuracy = accuracy_score(golds, preds)
        micro_p = precision_score(golds, preds)
        micro_r = recall_score(golds, preds)
        micro_f1 = f1_score(golds, preds)
    if average == 'micro':
        return micro_p, micro_r, micro_f1, accuracy
    else:
        return macro_p, macro_r, macro_f1, accuracy

def align_predictions(golds, preds, label_list):
    label_map = {i : label for i, label in enumerate(label_list)}
    seq_num, seq_len = preds.shape
    golds_list = [[] for _ in range(seq_num)]
    preds_list = [[] for _ in range(seq_num)]
    for i in range(seq_num):
        for j in range(seq_len):
            if golds[i, j] != nn.CrossEntropyLoss().ignore_index:
                golds_list[i].append(label_map[golds[i][j]])
                preds_list[i].append(label_map[preds[i][j]])
    return golds_list, preds_list

class Metrics():

    def __init__(self, ignore_index=-100):
        '''
        word_encoder: Sentence encoder
        
        You need to set self.cost as your own loss function.
        '''
        self.ignore_index = ignore_index

    def __get_class_span_dict__(self, label, is_string=False):
        '''
        return a dictionary of each class label/tag corresponding to the entity positions in the sentence
        {label:[(start_pos, end_pos), ...]}
        '''
        class_span = {}
        current_label = None
        i = 0
        if not is_string:
            # having labels in [0, num_of_class] 
            while i < len(label):
                if label[i] > 0:
                    start = i
                    current_label = label[i]
                    i += 1
                    while i < len(label) and label[i] == current_label:
                        i += 1
                    if current_label in class_span:
                        class_span[current_label].append((start, i))
                    else:
                        class_span[current_label] = [(start, i)]
                else:
                    assert label[i] == 0
                    i += 1
        else:
            # having tags in string format ['O', 'O', 'person-xxx', ..]
            while i < len(label):
                if label[i] != 'O':
                    start = i
                    current_label = label[i]
                    i += 1
                    while i < len(label) and label[i] == current_label:
                        i += 1
                    if current_label in class_span:
                        class_span[current_label].append((start, i))
                    else:
                        class_span[current_label] = [(start, i)]
                else:
                    i += 1
        return class_span

    def __get_intersect_by_entity__(self, pred_class_span, label_class_span):
        '''
        return the count of correct entity
        '''
        cnt = 0
        for label in label_class_span:
            cnt += len(list(set(label_class_span[label]).intersection(set(pred_class_span.get(label,[])))))
        return cnt

    def __get_cnt__(self, label_class_span):
        '''
        return the count of entities
        '''
        cnt = 0
        for label in label_class_span:
            cnt += len(label_class_span[label])
        return cnt

    def __get_correct_span__(self, pred_span, label_span):
        '''
        return count of correct entity spans
        '''
        pred_span_list = []
        label_span_list = []
        for pred in pred_span:
            pred_span_list += pred_span[pred]
        for label in label_span:
            label_span_list += label_span[label]
        return len(list(set(pred_span_list).intersection(set(label_span_list))))

    def __get_wrong_within_span__(self, pred_span, label_span):
        '''
        return count of entities with correct span, correct coarse type but wrong finegrained type
        '''
        cnt = 0
        for label in label_span:
            coarse = label.split('-')[0]
            within_pred_span = []
            for pred in pred_span:
                if pred != label and pred.split('-')[0] == coarse:
                    within_pred_span += pred_span[pred]
            cnt += len(list(set(label_span[label]).intersection(set(within_pred_span))))
        return cnt

    def __get_wrong_outer_span__(self, pred_span, label_span):
        '''
        return count of entities with correct span but wrong coarse type
        '''
        cnt = 0
        for label in label_span:
            coarse = label.split('-')[0]
            outer_pred_span = []
            for pred in pred_span:
                if pred != label and pred.split('-')[0] != coarse:
                    outer_pred_span += pred_span[pred]
            cnt += len(list(set(label_span[label]).intersection(set(outer_pred_span))))
        return cnt

    def __get_type_error__(self, pred, label, query):
        '''
        return finegrained type error cnt, coarse type error cnt and total correct span count
        '''
        pred_tag, label_tag = self.__transform_label_to_tag__(pred, query)
        pred_span = self.__get_class_span_dict__(pred_tag, is_string=True)
        label_span = self.__get_class_span_dict__(label_tag, is_string=True)
        total_correct_span = self.__get_correct_span__(pred_span, label_span) + 1e-6
        wrong_within_span = self.__get_wrong_within_span__(pred_span, label_span)
        wrong_outer_span = self.__get_wrong_outer_span__(pred_span, label_span)
        return wrong_within_span, wrong_outer_span, total_correct_span
                
    def metrics_by_entity_(self, pred, label):
        '''
        return entity level count of total prediction, true labels, and correct prediction
        '''
        pred_class_span = self.__get_class_span_dict__(pred, is_string=True)
        label_class_span = self.__get_class_span_dict__(label, is_string=True)
        pred_cnt = self.__get_cnt__(pred_class_span)
        label_cnt = self.__get_cnt__(label_class_span)
        correct_cnt = self.__get_intersect_by_entity__(pred_class_span, label_class_span)
        return pred_cnt, label_cnt, correct_cnt

    def get_acc_by_token(self, pred, label):
        correct_tokens = 0
        all_tokens = len(pred)
        for tp, tl in zip(pred, label):
            if tp == tl:
                correct_tokens += 1
        return correct_tokens, all_tokens

    def metrics_by_entity(self, pred, label):
        pred_cnt = 0
        label_cnt = 0
        correct_cnt = 0
        acc_correct = 0
        acc_all = 0
        for i in range(len(pred)):
            p_cnt, l_cnt, c_cnt = self.metrics_by_entity_(pred[i], label[i])
            correct_tokens, all_tokens = self.get_acc_by_token(pred[i], label[i])
            pred_cnt += p_cnt
            label_cnt += l_cnt
            correct_cnt += c_cnt
            acc_correct += correct_tokens
            acc_all += all_tokens
        precision = correct_cnt / pred_cnt
        recall = correct_cnt / label_cnt
        if precision == 0 and recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        accuracy = acc_correct / acc_all
        return precision, recall, f1, accuracy

