import torch
import torch.nn as nn

from model.modeling import BertModel


class BaseModel(nn.Module):
    def __init__(self, bert_model):
        super(BaseModel, self).__init__()
        assert isinstance(bert_model, BertModel), 'Wrong type for bert!'
        self.bert_model = bert_model
        self.bert_config = bert_model.config
        self.scorer = nn.Linear(self.bert_config.hidden_size*2, 2)
        #self.context_lstm = LSTM(input_size=hidden_size, hidden_size=hidden_size, bidirectional=True)
        #self.gloss_lstm = LSTM(input_size=hidden_size, hidden_size=hidden_size, bidirectional=True)

    def forward(self, input_id1s_tensor, input_mask1_tensor, segment_ids1_tensor, selection_mask_tensor,
              input_id2s_tensor, input_mask2_tensor, segment_ids2_tensor):
        real_selection_mask_tensor = selection_mask_tensor.unsqueeze(-1).expand(-1, -1,
                                                                                self.bert_config.hidden_size)
        bert_hiddens1, _ = self.bert_model(input_id1s_tensor, token_type_ids=segment_ids1_tensor,
                        attention_mask=input_mask1_tensor, output_all_encoded_layers=False)
        hiddens_selected = torch.where((real_selection_mask_tensor==1), bert_hiddens1, torch.zeros_like(bert_hiddens1))
        # The scaling factor is used to obtain the mean hiddens states of target word that contains more than 1 token.
        scaling_factor_tensor = selection_mask_tensor.sum(1).unsqueeze(-1)
        # Restrict the scaling factor >= 1 to avoid x/0 in case of invalid inputs
        scaling_factor_tensor = torch.where(scaling_factor_tensor >= 1, scaling_factor_tensor, torch.ones_like(scaling_factor_tensor))
        final_target_hidden_batch = (hiddens_selected.sum(1))/scaling_factor_tensor

        bert_hiddens2, _ = self.bert_model(input_id2s_tensor, token_type_ids=segment_ids2_tensor,
                                           attention_mask=input_mask2_tensor, output_all_encoded_layers=False)
        final_gloss_hidden_batch = bert_hiddens2[:, 0]

        cat_feature = torch.cat((final_target_hidden_batch, final_gloss_hidden_batch),dim=-1)
        logits = self.scorer(cat_feature)

        return logits


class BaseModel2(nn.Module):
    def __init__(self, bert_model):
        super(BaseModel2, self).__init__()
        assert isinstance(bert_model, BertModel), 'Wrong type for bert!'
        self.bert_model = bert_model
        self.bert_config = bert_model.config
        self.classifier = nn.Linear(self.bert_config.hidden_size, 2)
        #self.context_lstm = LSTM(input_size=hidden_size, hidden_size=hidden_size, bidirectional=True)
        #self.gloss_lstm = LSTM(input_size=hidden_size, hidden_size=hidden_size, bidirectional=True)

    def forward(self, input_id1s_tensor, input_mask1_tensor, segment_ids1_tensor, selection_mask_tensor):
        real_selection_mask_tensor = selection_mask_tensor.unsqueeze(-1).expand(-1, -1,
                                                                                self.bert_config.hidden_size)
        bert_hiddens1, _ = self.bert_model(input_id1s_tensor, token_type_ids=segment_ids1_tensor,
                        attention_mask=input_mask1_tensor, output_all_encoded_layers=False)
        hiddens_selected = torch.where((real_selection_mask_tensor==1), bert_hiddens1, torch.zeros_like(bert_hiddens1))
        # The scaling factor is used to obtain the mean hiddens states of target word that contains more than 1 token.
        scaling_factor_tensor = selection_mask_tensor.sum(1).unsqueeze(-1)
        # Restrict the scaling factor >= 1 to avoid x/0 in case of invalid inputs
        scaling_factor_tensor = torch.where(scaling_factor_tensor >= 1, scaling_factor_tensor, torch.ones_like(scaling_factor_tensor))
        final_target_hidden_batch = (hiddens_selected.sum(1))/scaling_factor_tensor

        logits = self.classifier(final_target_hidden_batch)

        return logits
