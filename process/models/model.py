import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, RobertaModel, RobertaPreTrainedModel

class FCLayer(nn.Module):
    
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        if self.use_activation:
            x = self.tanh(x)
        return x

class BertForEntityTyping(BertPreTrainedModel):
    
    def __init__(self, config):
        super(BertForEntityTyping, self).__init__(config)
        self.bert = BertModel(config=config)
        self.num_labels = config.num_labels
        self.label_classifier = FCLayer(
            config.hidden_size*2,
            config.num_labels,
            config.cls_dropout_rate,
            use_activation=False
            )

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask).sum(dim=1).unsqueeze(1)  # [batch_size, 1]
        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector
    
    def forward(self, input_ids, attention_mask, labels, ent_mask):

        # get features
        outputs = self.bert(input_ids, attention_mask)
        sequence_output = outputs[0]
        cls_features = sequence_output[:,0,:]
        ent_features = self.entity_average(sequence_output, ent_mask)
        features = torch.cat([cls_features, ent_features], dim=-1)

        # classification
        logits = self.label_classifier(features)

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        return (loss,) + (logits,)

class BertForRelationClassification(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForRelationClassification, self).__init__(config)
        self.bert = BertModel(config=config)
        self.num_labels = config.num_labels
        self.label_classifier = FCLayer(
            config.hidden_size*3,
            config.num_labels,
            config.cls_dropout_rate,
            use_activation=False
            )

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask).sum(dim=1).unsqueeze(1)  # [batch_size, 1]
        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, attention_mask, labels, e1_mask, e2_mask):

        # get features
        outputs = self.bert(input_ids, attention_mask)
        sequence_output = outputs[0]
        cls_features = sequence_output[:,0,:]
        e1_features = self.entity_average(sequence_output, e1_mask)
        e2_features = self.entity_average(sequence_output, e2_mask)
        features = torch.cat([cls_features, e1_features, e2_features], dim=-1)

        # classification
        logits = self.label_classifier(features)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return (loss,) + (logits,)

class RobertaForEntityTyping(RobertaPreTrainedModel):

    def __init__(self, config):
        super(RobertaForEntityTyping, self).__init__(config)
        self.roberta = RobertaModel(config=config)
        self.num_labels = config.num_labels
        self.label_classifier = FCLayer(
            config.hidden_size*2,
            config.num_labels,
            config.cls_dropout_rate,
            use_activation=False
            )

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask).sum(dim=1).unsqueeze(1)  # [batch_size, 1]
        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector
    
    def forward(self, input_ids, attention_mask, labels, ent_mask):
    
        # get features
        outputs = self.roberta(input_ids, attention_mask)
        sequence_output = outputs[0]
        cls_features = sequence_output[:,0,:]
        ent_features = self.entity_average(sequence_output, ent_mask)
        features = torch.cat([cls_features, ent_features], dim=-1)

        # classification
        logits = self.label_classifier(features)

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        return (loss,) + (logits,)

class RobertaForRelationClassification(RobertaPreTrainedModel):
    
    def __init__(self, config):
        super(RobertaForRelationClassification, self).__init__(config)
        self.roberta = RobertaModel(config=config)
        self.num_labels = config.num_labels
        self.label_classifier = FCLayer(
            config.hidden_size*3,
            config.num_labels,
            config.cls_dropout_rate,
            use_activation=False
            )

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask).sum(dim=1).unsqueeze(1)  # [batch_size, 1]
        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, attention_mask, labels, e1_mask, e2_mask):

        # get features
        outputs = self.roberta(input_ids, attention_mask)
        sequence_output = outputs[0]
        # cls_features = sequence_output[:,0,:]
        cls_features = outputs[1]
        e1_features = self.entity_average(sequence_output, e1_mask)
        e2_features = self.entity_average(sequence_output, e2_mask)
        features = torch.cat([cls_features, e1_features, e2_features], dim=-1)

        # classification
        logits = self.label_classifier(features)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return (loss,) + (logits,)