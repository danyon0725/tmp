from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn as nn
import torch
from transformers.modeling_electra import ElectraModel, ElectraPreTrainedModel
from transformers.modeling_bert import BertModel, BertPreTrainedModel
import torch.nn.functional as F
# 라벨 분포에 따른 weight 생성 함수
def get_label_weight(labels, num_label):
    number_of_labels = []

    # 입력 데이터의 전체 수를 저장하기위한 변수
    # 데이터 10개 중, 9개가 1번 라벨이고 1개가 2번 라벨일 경우
    # weight는 [0.1, 0.9]로 계산
    number_of_total = torch.zeros(size=(1,), dtype=torch.float, device=torch.device("cuda"))

    for label_index in range(0, num_label):
        # 라벨 index를 순차적으로 받아와 현재 라벨(label_index)에 해당하는 데이터 수를 계산
        number_of_label = (labels == label_index).sum(dim=-1).float()

        # 현재 라벨 분포 저장
        number_of_labels.append(number_of_label)

        # 전체 분모에 현재 라벨을 가진 데이터를 합치는 과정
        number_of_total = torch.add(number_of_total, number_of_label).float()

    # 리스트로 선언된 number_of_labels를 torch.tensor() 형태로 변환
    label_weight = torch.stack(tensors=number_of_labels, dim=0)

    # 각 라벨 분포를 전체 데이터 수로 나누어서 라벨 웨이트 계산
    label_weight = torch.ones(size=(1,), dtype=torch.float, device=torch.device("cuda")) - torch.div(label_weight,
                                                                                                     number_of_total)
    return label_weight


class AttentivePooling(nn.Module):
    def __init__(self, hidden_size):
        super(AttentivePooling, self).__init__()
        self.hidden_size = hidden_size
        self.q_projection = nn.Linear(self.hidden_size, self.hidden_size)
        self.c_projection = nn.Linear(self.hidden_size, self.hidden_size)

    def __call__(self, query, context, context_mask):
        # query : [batch, hidden]
        # context : [batch, window, seq, hidden]
        # context_mask : [batch, window, seq]

        # q : [batch, hidden, 1]
        # c : [batch, window, seq, hidden]
        q = self.q_projection(query).unsqueeze(-1)
        c = self.c_projection(context)

        reshape_c = c.view(-1, 200*512, self.hidden_size)
        # att : [batch, window, seq]
        att = reshape_c.bmm(q).squeeze().view(-1, 200, 512)

        # masked_att : [batch, window, seq]
        masked_att = att*context_mask

        # att_alienment : [batch, window, seq, 1]
        att_alienment = F.softmax(masked_att, dim=-1).unsqueeze(-1)

        # expand_att_alienment : [batch, window, seq, hidden]
        expand_att_alienment = att_alienment.expand(-1, -1, -1, self.hidden_size)
        weighted_context = expand_att_alienment*context
        # result : [batch, window, hidden]
        result = torch.sum(weighted_context, dim=2)
        # token result : [batch, seq, hidden]
        token_result = torch.sum(weighted_context, dim=1)

        return result, token_result
# class AttentivePooling_v2(nn.Module):
#     def __init__(self, hidden_size):
#         super(AttentivePooling_v2, self).__init__()
#         self.hidden_size = hidden_size
#         self.q_projection = nn.Linear(self.hidden_size, 1)
#         self.c_projection = nn.Linear(self.hidden_size, self.hidden_size)
#
#     def __call__(self, query, context, context_mask):
#         # query : [batch, q_step, hidden]
#         # context : [batch, window, seq, hidden]
#         # context_mask : [batch, window, seq]
#
#         # q : [batch, hidden, 1]
#         # c : [batch, window, seq, hidden]
#         q = self.q_projection(query).squeeze()
#         q_weight = F.softmax(q, dim=-1).unsqueeze(-1)
#         expand_q_weight = q_weight.expand(-1, -1, self.hidden_size)
#         query_vector = torch.sum(expand_q_weight*query, dim=1).unsqueeze(-1)
#
#
#         c = self.c_projection(context)
#
#         reshape_c = c.view(-1, 200*512, self.hidden_size)
#         # att : [batch, window, seq]
#         att = reshape_c.bmm(query_vector).squeeze().view(-1, 200, 512)
#
#         # masked_att : [batch, window, seq]
#         masked_att = att*context_mask
#
#         # att_alienment : [batch, window, seq, 1]
#         att_alienment = F.softmax(masked_att, dim=-1).unsqueeze(-1)
#
#         # expand_att_alienment : [batch, window, seq, hidden]
#         expand_att_alienment = att_alienment.expand(-1, -1, -1, self.hidden_size)
#         weighted_context = expand_att_alienment*context
#         # result : [batch, window, hidden]
#         result = torch.sum(weighted_context, dim=2)
#         # token result : [batch, seq, hidden]
#         # token_result = torch.sum(weighted_context, dim=1)
#
#         return result

class ElectraForQuestionAnswering(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraForQuestionAnswering, self).__init__(config)
        # 분류 해야할 라벨 개수 (start/end)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        # ELECTRA 모델 선언
        self.electra = ElectraModel(config)

        # bi-gru layer 선언
        self.bi_gru = nn.GRU(input_size=config.hidden_size, hidden_size=int(config.hidden_size / 2),
                             num_layers=1, batch_first=True, dropout=0.2, bidirectional=True)
        self.question_encoder = nn.GRU(input_size=config.hidden_size, hidden_size=int(config.hidden_size / 2),
                                       num_layers=1, batch_first=True, dropout=0.2, bidirectional=True)
        self.question_type_outputs = nn.Linear(config.hidden_size, 2)
        # self.sent_att = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=1, bias=True)
        # bi-gru layer output을 2의 크기로 줄여주기 위한 fnn
        self.sent_qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.tok_qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.att_pool = AttentivePooling(config.hidden_size)
        self.sent_gru =  nn.GRU(input_size=config.hidden_size, hidden_size=int(config.hidden_size / 2),
                             num_layers=1, batch_first=True, dropout=0.2, bidirectional=True)


        # ELECTRA weight 초기화
        self.init_weights()
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            question_type_label=None,
            sentence_mask=None,
            question_mask=None,
            inputs_embeds=None,
            sent_start_positions=None,
            sent_end_positions=None,
            tok_start_positions=None,
            tok_end_positions=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """

        # ELECTRA output 저장
        # outputs : [1, batch_size, seq_length, hidden_size]
        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )

        # sequence_output : [batch_size, seq_length, hidden_size]
        sequence_output = outputs[0]

        # gru_output : [batch_size, seq_length, gru_hidden_size*2]
        gru_output, _ = self.bi_gru(sequence_output)

        question_mask = question_mask.unsqueeze(-1).expand(-1, -1, self.hidden_size)

        encoded_question = gru_output * question_mask
        encoded_question = encoded_question[:, :64, :]
        question_gru_outputs, question_gru_states = self.question_encoder(encoded_question)
        question_vector = torch.cat([question_gru_states[0], question_gru_states[1]], -1)

        question_type_logits = self.question_type_outputs(question_vector)

        # # one_hot_sent_mask : [batch, 200, 512]
        one_hot_sent_mask = F.one_hot(sentence_mask, 200).transpose(1, 2)

        # expanded_sent_mask / expanded_seq_output : [batch, 200, 512, 768]
        expanded_sent_mask = one_hot_sent_mask.unsqueeze(-1).expand(-1, -1, -1, self.config.hidden_size)
        expanded_seq_output = gru_output.unsqueeze(1).expand(-1, 200, -1, -1)

        # sep_seq_output : [batch, window, seq, hidden]
        sep_seq_output = expanded_sent_mask*expanded_seq_output

        # sent_output : [batch, window, hidden]
        sent_output, _ = self.att_pool(question_vector, sep_seq_output, one_hot_sent_mask)

        sent_gru_output, _ = self.sent_gru(sent_output)
        # logits : [batch_size, window, 2]
        sent_logits = self.sent_qa_outputs(sent_gru_output)
        tok_logits = self.tok_qa_outputs(gru_output)

        # start_logits : [batch_size, window, 1]
        # end_logits : [batch_size, window, 1]
        sent_start_logits, sent_end_logits = sent_logits.split(1, dim=-1)
        tok_start_logits, tok_end_logits = tok_logits.split(1, dim=-1)

        # start_logits : [batch_size, window]
        # end_logits : [batch_size, window]
        sent_start_logits = sent_start_logits.squeeze(-1)
        sent_end_logits = sent_end_logits.squeeze(-1)
        tok_start_logits = tok_start_logits.squeeze(-1)
        tok_end_logits = tok_end_logits.squeeze(-1)

        # 학습 시
        if tok_start_positions is not None and tok_end_positions is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms

            # ignored_index : max_length
            sent_ignored_index = sent_start_logits.size(1)
            tok_ignored_index = tok_logits.size(1)
            # 코드의 안정성을 위해 인덱스 범위 지정 (0~max_length)
            sent_start_positions.clamp_(0, sent_ignored_index)
            sent_end_positions.clamp_(0, sent_ignored_index)
            tok_start_positions.clamp_(0, tok_ignored_index)
            tok_end_positions.clamp_(0, tok_ignored_index)
            sent_end_weight = get_label_weight(sent_end_positions, 200)
            sent_start_weight = get_label_weight(sent_start_positions, 200)
            tok_end_weight = get_label_weight(tok_start_positions, 512)
            tok_start_weight = get_label_weight(tok_end_positions, 512)
            # logg_fct 선언
            sent_start_loss_fct = CrossEntropyLoss(ignore_index=sent_ignored_index, weight=sent_start_weight)
            sent_end_loss_fct = CrossEntropyLoss(ignore_index=sent_ignored_index, weight=sent_end_weight)
            tok_start_loss_fct = CrossEntropyLoss(ignore_index=tok_ignored_index, weight=tok_start_weight)
            tok_end_loss_fct = CrossEntropyLoss(ignore_index=tok_ignored_index, weight=tok_end_weight)
            qt_loss_fct = nn.CrossEntropyLoss()
            # start/end에 대해 loss 계산

            sent_start_loss = sent_start_loss_fct(sent_start_logits, sent_start_positions)
            sent_end_loss = sent_end_loss_fct(sent_end_logits, sent_end_positions)
            tok_start_loss = tok_start_loss_fct(tok_start_logits, tok_start_positions)
            tok_end_loss = tok_end_loss_fct(tok_end_logits, tok_end_positions)
            question_type_loss = qt_loss_fct(question_type_logits, question_type_label)


            # 최종 loss 계산
            total_loss = (sent_start_loss+ sent_end_loss+tok_start_loss +tok_end_loss + question_type_loss) / 5

            # outputs : (total_loss, start_logits, end_logits)

            return total_loss,  sent_start_loss+ sent_end_loss,tok_start_loss + tok_end_loss, question_type_loss
        question_type_outputs = question_type_logits.argmax(dim=-1)
        return sent_start_logits, sent_end_logits,tok_start_logits, tok_end_logits, question_type_outputs# (loss), start_logits, end_logits, sent_token_logits

class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 분류 해야할 라벨 개수 (start/end)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        # ELECTRA 모델 선언
        self.bert = BertModel(config)

        # bi-gru layer 선언
        self.bi_gru = nn.GRU(input_size=config.hidden_size, hidden_size=int(config.hidden_size / 2),
                             num_layers=1, batch_first=True, dropout=0.2, bidirectional=True)
        self.question_encoder = nn.GRU(input_size=config.hidden_size, hidden_size=int(config.hidden_size / 2),
                                       num_layers=1, batch_first=True, dropout=0.2, bidirectional=True)
        self.question_type_outputs = nn.Linear(config.hidden_size, 2)
        # self.sent_att = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=1, bias=True)
        # bi-gru layer output을 2의 크기로 줄여주기 위한 fnn
        self.sent_qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.tok_qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.att_pool = AttentivePooling(config.hidden_size)
        self.sent_gru =  nn.GRU(input_size=config.hidden_size, hidden_size=int(config.hidden_size / 2),
                             num_layers=1, batch_first=True, dropout=0.2, bidirectional=True)


        # ELECTRA weight 초기화
        self.init_weights()
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            question_type_label=None,
            sentence_mask=None,
            question_mask=None,
            inputs_embeds=None,
            sent_start_positions=None,
            sent_end_positions=None,
            tok_start_positions=None,
            tok_end_positions=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """

        # ELECTRA output 저장
        # outputs : [1, batch_size, seq_length, hidden_size]
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )

        # sequence_output : [batch_size, seq_length, hidden_size]
        sequence_output = outputs[0]

        # gru_output : [batch_size, seq_length, gru_hidden_size*2]
        gru_output, _ = self.bi_gru(sequence_output)

        question_mask = question_mask.unsqueeze(-1).expand(-1, -1, self.hidden_size)

        encoded_question = gru_output * question_mask
        encoded_question = encoded_question[:, :64, :]
        question_gru_outputs, question_gru_states = self.question_encoder(encoded_question)
        question_vector = torch.cat([question_gru_states[0], question_gru_states[1]], -1)

        question_type_logits = self.question_type_outputs(question_vector)

        # # one_hot_sent_mask : [batch, 200, 512]
        one_hot_sent_mask = F.one_hot(sentence_mask, 200).transpose(1, 2)

        # expanded_sent_mask / expanded_seq_output : [batch, 200, 512, 768]
        expanded_sent_mask = one_hot_sent_mask.unsqueeze(-1).expand(-1, -1, -1, self.config.hidden_size)
        expanded_seq_output = gru_output.unsqueeze(1).expand(-1, 200, -1, -1)

        # sep_seq_output : [batch, window, seq, hidden]
        sep_seq_output = expanded_sent_mask*expanded_seq_output

        # sent_output : [batch, window, hidden]
        sent_output, _ = self.att_pool(question_vector, sep_seq_output, one_hot_sent_mask)

        sent_gru_output, _ = self.sent_gru(sent_output)
        # logits : [batch_size, window, 2]
        sent_logits = self.sent_qa_outputs(sent_gru_output)
        tok_logits = self.tok_qa_outputs(gru_output)

        # start_logits : [batch_size, window, 1]
        # end_logits : [batch_size, window, 1]
        sent_start_logits, sent_end_logits = sent_logits.split(1, dim=-1)
        tok_start_logits, tok_end_logits = tok_logits.split(1, dim=-1)

        # start_logits : [batch_size, window]
        # end_logits : [batch_size, window]
        sent_start_logits = sent_start_logits.squeeze(-1)
        sent_end_logits = sent_end_logits.squeeze(-1)
        tok_start_logits = tok_start_logits.squeeze(-1)
        tok_end_logits = tok_end_logits.squeeze(-1)

        # 학습 시
        if tok_start_positions is not None and tok_end_positions is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms

            # ignored_index : max_length
            sent_ignored_index = sent_start_logits.size(1)
            tok_ignored_index = tok_logits.size(1)
            # 코드의 안정성을 위해 인덱스 범위 지정 (0~max_length)
            sent_start_positions.clamp_(0, sent_ignored_index)
            sent_end_positions.clamp_(0, sent_ignored_index)
            tok_start_positions.clamp_(0, tok_ignored_index)
            tok_end_positions.clamp_(0, tok_ignored_index)
            sent_end_weight = get_label_weight(sent_end_positions, 200)
            sent_start_weight = get_label_weight(sent_start_positions, 200)
            tok_end_weight = get_label_weight(tok_start_positions, 512)
            tok_start_weight = get_label_weight(tok_end_positions, 512)
            # logg_fct 선언
            sent_start_loss_fct = CrossEntropyLoss(ignore_index=sent_ignored_index, weight=sent_start_weight)
            sent_end_loss_fct = CrossEntropyLoss(ignore_index=sent_ignored_index, weight=sent_end_weight)
            tok_start_loss_fct = CrossEntropyLoss(ignore_index=tok_ignored_index, weight=tok_start_weight)
            tok_end_loss_fct = CrossEntropyLoss(ignore_index=tok_ignored_index, weight=tok_end_weight)
            qt_loss_fct = nn.CrossEntropyLoss()
            # start/end에 대해 loss 계산

            sent_start_loss = sent_start_loss_fct(sent_start_logits, sent_start_positions)
            sent_end_loss = sent_end_loss_fct(sent_end_logits, sent_end_positions)
            tok_start_loss = tok_start_loss_fct(tok_start_logits, tok_start_positions)
            tok_end_loss = tok_end_loss_fct(tok_end_logits, tok_end_positions)
            question_type_loss = qt_loss_fct(question_type_logits, question_type_label)


            # 최종 loss 계산
            total_loss = (sent_start_loss+ sent_end_loss+tok_start_loss +tok_end_loss + question_type_loss) / 5

            # outputs : (total_loss, start_logits, end_logits)

            return total_loss,  sent_start_loss+ sent_end_loss,tok_start_loss + tok_end_loss, question_type_loss
        question_type_outputs = question_type_logits.argmax(dim=-1)
        return sent_start_logits, sent_end_logits,tok_start_logits, tok_end_logits, question_type_outputs# (loss), start_logits, end_logits, sent_token_logits

class tmp(ElectraPreTrainedModel):
    def __init__(self, config):
        super(tmp, self).__init__(config)
        # 분류 해야할 라벨 개수 (start/end)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        # ELECTRA 모델 선언
        self.electra = ElectraModel(config)

        # bi-gru layer 선언
        self.bi_gru = nn.GRU(input_size=config.hidden_size, hidden_size=int(config.hidden_size / 2),
                             num_layers=1, batch_first=True, dropout=0.2, bidirectional=True)
        self.question_encoder = nn.GRU(input_size=config.hidden_size, hidden_size=int(config.hidden_size / 2),
                                       num_layers=1, batch_first=True, dropout=0.2, bidirectional=True)
        self.question_type_outputs = nn.Linear(config.hidden_size, 2)
        # self.sent_att = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=1, bias=True)
        # bi-gru layer output을 2의 크기로 줄여주기 위한 fnn
        self.tok_qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.sent_gru = nn.GRU(input_size=config.hidden_size, hidden_size=int(config.hidden_size / 2),
                               num_layers=1, batch_first=True, dropout=0.2, bidirectional=True)

        # ELECTRA weight 초기화
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            question_type_label=None,
            sentence_mask=None,
            question_mask=None,
            inputs_embeds=None,
            sent_start_positions=None,
            sent_end_positions=None,
            tok_start_positions=None,
            tok_end_positions=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """

        # ELECTRA output 저장
        # outputs : [1, batch_size, seq_length, hidden_size]
        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )

        # sequence_output : [batch_size, seq_length, hidden_size]
        sequence_output = outputs[0]

        # gru_output : [batch_size, seq_length, gru_hidden_size*2]
        gru_output, _ = self.bi_gru(sequence_output)

        question_mask = question_mask.unsqueeze(-1).expand(-1, -1, self.hidden_size)

        encoded_question = gru_output * question_mask
        encoded_question = encoded_question[:, :64, :]
        question_gru_outputs, question_gru_states = self.question_encoder(encoded_question)
        question_vector = torch.cat([question_gru_states[0], question_gru_states[1]], -1)

        question_type_logits = self.question_type_outputs(question_vector)

        tok_logits = self.tok_qa_outputs(gru_output)

        # start_logits : [batch_size, window, 1]
        # end_logits : [batch_size, window, 1]
        tok_start_logits, tok_end_logits = tok_logits.split(1, dim=-1)

        # start_logits : [batch_size, window]
        # end_logits : [batch_size, window]
        tok_start_logits = tok_start_logits.squeeze(-1)
        tok_end_logits = tok_end_logits.squeeze(-1)

        # 학습 시
        if tok_start_positions is not None and tok_end_positions is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms

            # ignored_index : max_length
            tok_ignored_index = tok_logits.size(1)
            # 코드의 안정성을 위해 인덱스 범위 지정 (0~max_length)
            tok_start_positions.clamp_(0, tok_ignored_index)
            tok_end_positions.clamp_(0, tok_ignored_index)
            sent_end_weight = get_label_weight(sent_end_positions, 200)
            sent_start_weight = get_label_weight(sent_start_positions, 200)
            tok_end_weight = get_label_weight(tok_start_positions, 512)
            tok_start_weight = get_label_weight(tok_end_positions, 512)
            # logg_fct 선언
            tok_start_loss_fct = CrossEntropyLoss(ignore_index=tok_ignored_index, weight=tok_start_weight)
            tok_end_loss_fct = CrossEntropyLoss(ignore_index=tok_ignored_index, weight=tok_end_weight)
            qt_loss_fct = nn.CrossEntropyLoss()
            # start/end에 대해 loss 계산

            tok_start_loss = tok_start_loss_fct(tok_start_logits, tok_start_positions)
            tok_end_loss = tok_end_loss_fct(tok_end_logits, tok_end_positions)
            question_type_loss = qt_loss_fct(question_type_logits, question_type_label)

            # 최종 loss 계산
            total_loss = (tok_start_loss + tok_end_loss + question_type_loss) / 3

            # outputs : (total_loss, start_logits, end_logits)

            return total_loss, 0, tok_start_loss + tok_end_loss, question_type_loss
        question_type_outputs = question_type_logits.argmax(dim=-1)
        return None, None, tok_start_logits, tok_end_logits, question_type_outputs  # (loss), start_logits, end_logits, sent_token_logits

                # class ElectraForQuestionAnswering_v2(ElectraPreTrainedModel):
#     def __init__(self, config):
#         super(ElectraForQuestionAnswering_v2, self).__init__(config)
#         # 분류 해야할 라벨 개수 (start/end)
#         self.num_labels = config.num_labels
#         self.hidden_size = config.hidden_size
#
#         # ELECTRA 모델 선언
#         self.electra = ElectraModel(config)
#
#         # bi-gru layer 선언
#         self.bi_gru = nn.GRU(input_size=config.hidden_size, hidden_size=int(config.hidden_size / 2),
#                              num_layers=1, batch_first=True, dropout=0.2, bidirectional=True)
#         self.question_encoder = nn.GRU(input_size=config.hidden_size, hidden_size=int(config.hidden_size / 2),
#                                        num_layers=1, batch_first=True, dropout=0.2, bidirectional=True)
#         self.question_type_outputs = nn.Linear(config.hidden_size, 2)
#         # self.sent_att = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=1, bias=True)
#         # bi-gru layer output을 2의 크기로 줄여주기 위한 fnn
#         self.sent_qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
#         self.tok_qa_outputs = nn.Linear(config.hidden_size*2, config.num_labels)
#         self.att_pool = AttentivePooling(config.hidden_size)
#         self.sent_gru = nn.GRU(input_size=config.hidden_size, hidden_size=int(config.hidden_size / 2),
#                                num_layers=1, batch_first=True, dropout=0.2, bidirectional=True)
#
#         # ELECTRA weight 초기화
#         self.init_weights()
#
#     def forward(
#             self,
#             input_ids=None,
#             attention_mask=None,
#             token_type_ids=None,
#             position_ids=None,
#             head_mask=None,
#             question_type_label=None,
#             sentence_mask=None,
#             question_mask=None,
#             inputs_embeds=None,
#             sent_start_positions=None,
#             sent_end_positions=None,
#             tok_start_positions=None,
#             tok_end_positions=None,
#     ):
#         r"""
#         start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
#             Labels for position (index) of the start of the labelled span for computing the token classification loss.
#             Positions are clamped to the length of the sequence (`sequence_length`).
#             Position outside of the sequence are not taken into account for computing the loss.
#         end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
#             Labels for position (index) of the end of the labelled span for computing the token classification loss.
#             Positions are clamped to the length of the sequence (`sequence_length`).
#             Position outside of the sequence are not taken into account for computing the loss.
#     Returns:
#         :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
#         loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
#             Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
#         start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
#             Span-start scores (before SoftMax).
#         end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
#             Span-end scores (before SoftMax).
#         hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
#             Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
#             of shape :obj:`(batch_size, sequence_length, hidden_size)`.
#             Hidden-states of the model at the output of each layer plus the initial embedding outputs.
#         attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
#             Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
#             :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
#             Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
#             heads.
#         """
#
#         # ELECTRA output 저장
#         # outputs : [1, batch_size, seq_length, hidden_size]
#         outputs = self.electra(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds
#         )
#
#         # sequence_output : [batch_size, seq_length, hidden_size]
#         sequence_output = outputs[0]
#
#         # gru_output : [batch_size, seq_length, gru_hidden_size*2]
#         gru_output, _ = self.bi_gru(sequence_output)
#
#         question_mask = question_mask.unsqueeze(-1).expand(-1, -1, self.hidden_size)
#
#         encoded_question = gru_output * question_mask
#         encoded_question = encoded_question[:, :64, :]
#         question_gru_outputs, question_gru_states = self.question_encoder(encoded_question)
#         question_vector = torch.cat([question_gru_states[0], question_gru_states[1]], -1)
#
#         question_type_logits = self.question_type_outputs(question_vector)
#
#         # # one_hot_sent_mask : [batch, 200, 512]
#         one_hot_sent_mask = F.one_hot(sentence_mask, 200).transpose(1, 2)
#
#         # expanded_sent_mask / expanded_seq_output : [batch, 200, 512, 768]
#         expanded_sent_mask = one_hot_sent_mask.unsqueeze(-1).expand(-1, -1, -1, self.config.hidden_size)
#         expanded_seq_output = gru_output.unsqueeze(1).expand(-1, 200, -1, -1)
#
#         # sep_seq_output : [batch, window, seq, hidden]
#         sep_seq_output = expanded_sent_mask * expanded_seq_output
#
#         # sent_output : [batch, window, hidden]
#         sent_output, sent_tok_output = self.att_pool(question_vector, sep_seq_output, one_hot_sent_mask)
#
#         sent_gru_output, _ = self.sent_gru(sent_output)
#         # logits : [batch_size, window, 2]
#         sent_logits = self.sent_qa_outputs(sent_gru_output)
#         tok_logits = self.tok_qa_outputs(torch.cat([gru_output, sent_tok_output], -1))
#
#         # start_logits : [batch_size, window, 1]
#         # end_logits : [batch_size, window, 1]
#         sent_start_logits, sent_end_logits = sent_logits.split(1, dim=-1)
#         tok_start_logits, tok_end_logits = tok_logits.split(1, dim=-1)
#
#         # start_logits : [batch_size, window]
#         # end_logits : [batch_size, window]
#         sent_start_logits = sent_start_logits.squeeze(-1)
#         sent_end_logits = sent_end_logits.squeeze(-1)
#         tok_start_logits = tok_start_logits.squeeze(-1)
#         tok_end_logits = tok_end_logits.squeeze(-1)
#
#         # 학습 시
#         if tok_start_positions is not None and tok_end_positions is not None:
#             # sometimes the start/end positions are outside our model inputs, we ignore these terms
#
#             # ignored_index : max_length
#             sent_ignored_index = sent_start_logits.size(1)
#             tok_ignored_index = tok_logits.size(1)
#             # 코드의 안정성을 위해 인덱스 범위 지정 (0~max_length)
#             sent_start_positions.clamp_(0, sent_ignored_index)
#             sent_end_positions.clamp_(0, sent_ignored_index)
#             tok_start_positions.clamp_(0, tok_ignored_index)
#             tok_end_positions.clamp_(0, tok_ignored_index)
#             sent_end_weight = get_label_weight(sent_end_positions, 200)
#             sent_start_weight = get_label_weight(sent_start_positions, 200)
#             tok_end_weight = get_label_weight(tok_start_positions, 512)
#             tok_start_weight = get_label_weight(tok_end_positions, 512)
#             # logg_fct 선언
#             sent_start_loss_fct = CrossEntropyLoss(ignore_index=sent_ignored_index, weight=sent_start_weight)
#             sent_end_loss_fct = CrossEntropyLoss(ignore_index=sent_ignored_index, weight=sent_end_weight)
#             tok_start_loss_fct = CrossEntropyLoss(ignore_index=tok_ignored_index, weight=tok_start_weight)
#             tok_end_loss_fct = CrossEntropyLoss(ignore_index=tok_ignored_index, weight=tok_end_weight)
#             qt_loss_fct = nn.CrossEntropyLoss()
#             # start/end에 대해 loss 계산
#
#             sent_start_loss = sent_start_loss_fct(sent_start_logits, sent_start_positions)
#             sent_end_loss = sent_end_loss_fct(sent_end_logits, sent_end_positions)
#             tok_start_loss = tok_start_loss_fct(tok_start_logits, tok_start_positions)
#             tok_end_loss = tok_end_loss_fct(tok_end_logits, tok_end_positions)
#             question_type_loss = qt_loss_fct(question_type_logits, question_type_label)
#
#             # 최종 loss 계산
#             total_loss = (
#                          sent_start_loss + sent_end_loss + tok_start_loss + tok_end_loss + question_type_loss) / 5
#
#             # outputs : (total_loss, start_logits, end_logits)
#
#             return total_loss, sent_start_loss + sent_end_loss, tok_start_loss + tok_end_loss, question_type_loss
#         question_type_outputs = question_type_logits.argmax(dim=-1)
#         return sent_start_logits, sent_end_logits, tok_start_logits, tok_end_logits, question_type_outputs  # (loss), start_logits, end_logits, sent_token_logits
# class ElectraForQuestionAnsweringMTL(ElectraPreTrainedModel):
#     def __init__(self, config):
#         super(ElectraForQuestionAnsweringMTL, self).__init__(config)
#         # 분류 해야할 라벨 개수 (start/end)
#         self.num_labels = config.num_labels
#         self.hidden_size = config.hidden_size
#
#         # ELECTRA 모델 선언
#         self.electra = ElectraModel(config)
#
#         # bi-gru layer 선언
#         self.bi_gru = nn.GRU(input_size=config.hidden_size, hidden_size=int(config.hidden_size / 2),
#                              num_layers=1, batch_first=True, dropout=0.2, bidirectional=True)
#         self.question_encoder = nn.GRU(input_size=config.hidden_size, hidden_size=int(config.hidden_size / 2),
#                                        num_layers=1, batch_first=True, dropout=0.2, bidirectional=True)
#         self.question_type_outputs = nn.Linear(config.hidden_size, 2)
#         # self.sent_att = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=1, bias=True)
#         # bi-gru layer output을 2의 크기로 줄여주기 위한 fnn
#         self.sent_qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
#         self.tok_qa_outputs = nn.Linear(200, config.num_labels)
#         self.att_pool = AttentivePooling(config.hidden_size)
#         self.sent_gru =  nn.GRU(input_size=config.hidden_size, hidden_size=int(config.hidden_size / 2),
#                              num_layers=1, batch_first=True, dropout=0.2, bidirectional=True)
#
#
#         # ELECTRA weight 초기화
#         self.init_weights()
#     def forward(
#             self,
#             input_ids=None,
#             attention_mask=None,
#             token_type_ids=None,
#             position_ids=None,
#             head_mask=None,
#             question_type_label=None,
#             sentence_mask=None,
#             question_mask=None,
#             inputs_embeds=None,
#             sent_start_positions=None,
#             sent_end_positions=None,
#             tok_start_positions=None,
#             tok_end_positions=None,
#     ):
#         r"""
#         start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
#             Labels for position (index) of the start of the labelled span for computing the token classification loss.
#             Positions are clamped to the length of the sequence (`sequence_length`).
#             Position outside of the sequence are not taken into account for computing the loss.
#         end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
#             Labels for position (index) of the end of the labelled span for computing the token classification loss.
#             Positions are clamped to the length of the sequence (`sequence_length`).
#             Position outside of the sequence are not taken into account for computing the loss.
#     Returns:
#         :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
#         loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
#             Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
#         start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
#             Span-start scores (before SoftMax).
#         end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
#             Span-end scores (before SoftMax).
#         hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
#             Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
#             of shape :obj:`(batch_size, sequence_length, hidden_size)`.
#             Hidden-states of the model at the output of each layer plus the initial embedding outputs.
#         attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
#             Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
#             :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
#             Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
#             heads.
#         """
#
#         # ELECTRA output 저장
#         # outputs : [1, batch_size, seq_length, hidden_size]
#         outputs = self.electra(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds
#         )
#
#         # sequence_output : [batch_size, seq_length, hidden_size]
#         sequence_output = outputs[0]
#
#         # gru_output : [batch_size, seq_length, gru_hidden_size*2]
#         gru_output, _ = self.bi_gru(sequence_output)
#
#         question_mask = question_mask.unsqueeze(-1).expand(-1, -1, self.hidden_size)
#
#         encoded_question = gru_output * question_mask
#         encoded_question = encoded_question[:, :64, :]
#         question_gru_outputs, question_gru_states = self.question_encoder(encoded_question)
#         question_vector = torch.cat([question_gru_states[0], question_gru_states[1]], -1)
#
#         question_type_logits = self.question_type_outputs(question_vector)
#
#         # # one_hot_sent_mask : [batch, 200, 512]
#         one_hot_sent_mask = F.one_hot(sentence_mask, 200).transpose(1, 2)
#
#         # expanded_sent_mask / expanded_seq_output : [batch, 200, 512, 768]
#         expanded_sent_mask = one_hot_sent_mask.unsqueeze(-1).expand(-1, -1, -1, self.config.hidden_size)
#         expanded_seq_output = gru_output.unsqueeze(1).expand(-1, 200, -1, -1)
#
#         # sep_seq_output : [batch, window, seq, hidden]
#         sep_seq_output = expanded_sent_mask*expanded_seq_output
#
#         # sent_output : [batch, window, hidden]
#         sent_output = self.att_pool(question_vector, sep_seq_output, one_hot_sent_mask)
#
#         # sent_gru_output : [batch, window, hidden]
#         sent_gru_output, _ = self.sent_gru(sent_output)
#
#         # matual_token_output : [batch, length, window]
#         matual_token_output = gru_output.bmm(sent_gru_output.transpose(1, 2))
#         # logits : [batch_size, window, 2]
#         sent_logits = self.sent_qa_outputs(sent_gru_output)
#         tok_logits = self.tok_qa_outputs(matual_token_output)
#
#         # start_logits : [batch_size, window, 1]
#         # end_logits : [batch_size, window, 1]
#         sent_start_logits, sent_end_logits = sent_logits.split(1, dim=-1)
#         tok_start_logits, tok_end_logits = tok_logits.split(1, dim=-1)
#
#         # start_logits : [batch_size, window]
#         # end_logits : [batch_size, window]
#         sent_start_logits = sent_start_logits.squeeze(-1)
#         sent_end_logits = sent_end_logits.squeeze(-1)
#         tok_start_logits = tok_start_logits.squeeze(-1)
#         tok_end_logits = tok_end_logits.squeeze(-1)
#
#         # 학습 시
#         if tok_start_positions is not None and tok_end_positions is not None:
#             # sometimes the start/end positions are outside our model inputs, we ignore these terms
#
#             # ignored_index : max_length
#             sent_ignored_index = sent_start_logits.size(1)
#             tok_ignored_index = tok_logits.size(1)
#             # 코드의 안정성을 위해 인덱스 범위 지정 (0~max_length)
#             sent_start_positions.clamp_(0, sent_ignored_index)
#             sent_end_positions.clamp_(0, sent_ignored_index)
#             tok_start_positions.clamp_(0, tok_ignored_index)
#             tok_end_positions.clamp_(0, tok_ignored_index)
#             sent_end_weight = get_label_weight(sent_end_positions, 200)
#             sent_start_weight = get_label_weight(sent_start_positions, 200)
#             tok_end_weight = get_label_weight(tok_start_positions, 512)
#             tok_start_weight = get_label_weight(tok_end_positions, 512)
#             # logg_fct 선언
#             sent_start_loss_fct = CrossEntropyLoss(ignore_index=sent_ignored_index, weight=sent_start_weight)
#             sent_end_loss_fct = CrossEntropyLoss(ignore_index=sent_ignored_index, weight=sent_end_weight)
#             tok_start_loss_fct = CrossEntropyLoss(ignore_index=tok_ignored_index, weight=tok_start_weight)
#             tok_end_loss_fct = CrossEntropyLoss(ignore_index=tok_ignored_index, weight=tok_end_weight)
#             qt_loss_fct = nn.CrossEntropyLoss()
#             # start/end에 대해 loss 계산
#
#             sent_start_loss = sent_start_loss_fct(sent_start_logits, sent_start_positions)
#             sent_end_loss = sent_end_loss_fct(sent_end_logits, sent_end_positions)
#             tok_start_loss = tok_start_loss_fct(tok_start_logits, tok_start_positions)
#             tok_end_loss = tok_end_loss_fct(tok_end_logits, tok_end_positions)
#             question_type_loss = qt_loss_fct(question_type_logits, question_type_label)
#
#
#             # 최종 loss 계산
#             total_loss = (sent_start_loss+ sent_end_loss+tok_start_loss +tok_end_loss + question_type_loss) / 5
#
#             # outputs : (total_loss, start_logits, end_logits)
#
#             return total_loss,  sent_start_loss+ sent_end_loss,tok_start_loss + tok_end_loss, question_type_loss
#         question_type_outputs = question_type_logits.argmax(dim=-1)
#         return sent_start_logits, sent_end_logits,tok_start_logits, tok_end_logits, question_type_outputs# (loss), start_logits, end_logits, sent_token_logits