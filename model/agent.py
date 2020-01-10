import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def random_pick(some_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
         cumulative_probability += item_probability
         if x < cumulative_probability:
               break
    return item


class Agent(nn.Module):
    def __init__(self, hidden_size):
        super(Agent, self).__init__()
        self.scorer = nn.Sequential(nn.Linear(hidden_size+hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))
        self.W1 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.Wa = nn.Linear(hidden_size, 1)
        self.V1 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.V2 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)

    def forward(self, base_model, instances, init_all_decision_vecs_tensor, mention_aware_gloss_tensor, sep_pos, init_pred_list, mode='single-step-train'):
        assert mode in ['train', 'single-step-train', 'eval']
        left_instances_idx = list(range(len(instances)))
        all_decision_vec_tensors_list = []
        all_mention_aware_gloss_tensors_list = []
        for i in range(len(instances)):
            decision_vec_tensor = init_all_decision_vecs_tensor[sep_pos[left_instances_idx[i]]:sep_pos[left_instances_idx[i] + 1]]
            all_decision_vec_tensors_list.append(decision_vec_tensor)
            all_mention_aware_gloss_tensors_list.append(mention_aware_gloss_tensor[sep_pos[left_instances_idx[i]]:sep_pos[left_instances_idx[i] + 1]])
        pred_list = init_pred_list
        while len(left_instances_idx) > 0:
            decision_vecs_tensor_list = []
            for left_instance_idx in left_instances_idx:
                decision_vec_tensor = all_decision_vec_tensors_list[left_instance_idx]
                decision_vecs_tensor_list.append(decision_vec_tensor)
            next_sample_probs = self.choose_next(decision_vecs_tensor_list, all_mention_aware_gloss_tensors_list, [pred_list[idx] for idx in left_instances_idx])
            if mode != 'eval':
                picked_idx = random_pick(list(range(len(left_instances_idx))), next_sample_probs)
                selected_instance_idx = left_instances_idx.pop(picked_idx)
                # selected_instance = instances[selected_instance_idx]
                selected_decision_vec_tensor = all_decision_vec_tensors_list[selected_instance_idx][pred_list[selected_instance_idx]]
                selected_gloss_vec_tensor = mention_aware_gloss_tensor[sep_pos[selected_instance_idx]+pred_list[selected_instance_idx]]
                for left_instance_idx in left_instances_idx:
                    target_decision_vecs_tensor = all_decision_vec_tensors_list[left_instance_idx]
                    target_gloss_vecs_tensor = mention_aware_gloss_tensor[sep_pos[left_instance_idx]:sep_pos[left_instance_idx+1]]
                    updated_target_decision_vecs_tensor = self.update_decision_vec(selected_decision_vec_tensor, selected_gloss_vec_tensor,
                                                                                   target_decision_vecs_tensor, target_gloss_vecs_tensor)
                    all_decision_vec_tensors_list[left_instance_idx] = updated_target_decision_vecs_tensor
                new_logits = base_model.classifier(torch.cat(all_decision_vec_tensors_list, dim=0))
                #pred_list = 1
            else:
                picked_idx = next_sample_probs.argmax(dim=0)

            if mode == 'single-step-train':
                return new_logits, next_sample_probs, selected_instance_idx

        return new_logits

    def choose_next(self, all_decision_vec_tensors_list, all_mention_aware_gloss_tensors_list, pred_list):
        num_instance = len(all_decision_vec_tensors_list)
        #maxpooled_decision_vecs = []
        avgpooled_decision_vec_tensors = []
        max_prob_decision_vec_tensors = []
        avgpooled_gloss_tensors = []
        max_prob_gloss_tensors = []
        for i in range(len(all_decision_vec_tensors_list)):
            # maxpooled_decision_vec = decision_vecs[i].max(dim=0)[0]
            avgpooled_decision_vec = all_decision_vec_tensors_list[i].mean(dim=0)
            max_prob_decision_vec = all_decision_vec_tensors_list[i][pred_list[i]]
            avgpooled_gloss_tensor = all_mention_aware_gloss_tensors_list[i].mean(dim=0)
            max_prob_gloss_tensor = all_mention_aware_gloss_tensors_list[i][pred_list[i]]
            # maxpooled_decision_vecs.append(maxpooled_decision_vec)
            avgpooled_decision_vec_tensors.append(avgpooled_decision_vec)
            max_prob_decision_vec_tensors.append(max_prob_decision_vec)
            avgpooled_gloss_tensors.append(avgpooled_gloss_tensor)
            max_prob_gloss_tensors.append(max_prob_gloss_tensor)


        # stacked_maxpooled_decision_vecs = torch.stack(maxpooled_decision_vecs)
        stacked_avgpooled_decision_vec_tensors = torch.stack(avgpooled_decision_vec_tensors)
        stacked_max_prob_decision_vec_tensors = torch.stack(max_prob_decision_vec_tensors)
        stacked_avgpooled_gloss_tensors = torch.stack(avgpooled_gloss_tensors)
        stacked_max_prob_gloss_tensors = torch.stack(max_prob_gloss_tensors)
        # stacked_pooled_decision_vecs = stacked_maxpooled_decision_vecs - stacked_avgpooled_decision_vecs
        stacked_pooled_decision_vec_tensors = stacked_max_prob_decision_vec_tensors - stacked_avgpooled_decision_vec_tensors
        stacked_pooled_gloss_vecs = stacked_max_prob_gloss_tensors - stacked_avgpooled_gloss_tensors

        score = self.scorer(torch.cat((stacked_pooled_decision_vec_tensors, stacked_pooled_gloss_vecs), dim=-1))
        sample_prob = F.softmax(score.squeeze(1), dim=0)

        return sample_prob

    def update_decision_vec(self, selected_decision_vec_tensor, selected_gloss_vec_tensor,
                            target_decision_vecs_tensor, target_gloss_vecs_tensor):
        selected_mention_gloss_vec_tensor = torch.cat((selected_decision_vec_tensor, selected_gloss_vec_tensor), dim=-1)
        target_mention_gloss_vecs_tensor = torch.cat((target_decision_vecs_tensor, target_gloss_vecs_tensor), dim=-1)
        N, d = target_mention_gloss_vecs_tensor.size()
        attn_h = self.W1(selected_mention_gloss_vec_tensor.unsqueeze(0)).expand(N, -1) + self.W2(target_mention_gloss_vecs_tensor)
        logits = self.Wa(F.tanh(attn_h))
        attn = F.softmax(logits, dim=0)

        '''Q = self.W_q(selected_mention_gloss_vec_tensor.unsqueeze(0))
        K = self.W_kv(target_decision_vecs_tensor)
        attn = F.softmax(torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(d), dim=-1)
        #max_id = attn.argmax()
        #attn_ = attn.new_zeros(attn.size())
        delta_target_decision_vecs_tensor = torch.matmul(attn.transpose(0, 1), selected_decision_vec.unsqueeze(0))
        updated_target_decision_vecs_tensor = target_decision_vecs_tensor + delta_target_decision_vecs_tensor
        '''
        '''
        delta_target_decision_vecs_tensor = self.gru(
            selected_mention_gloss_vec_tensor.unsqueeze(0).expand(N, -1),
            target_decision_vecs_tensor
        )
        '''
        delta_target_decision_vecs_tensor = self.U(F.relu(self.V1(selected_mention_gloss_vec_tensor)+self.V2(target_mention_gloss_vecs_tensor)))
        updated_target_decision_vecs_tensor = target_decision_vecs_tensor + attn * delta_target_decision_vecs_tensor

        return updated_target_decision_vecs_tensor