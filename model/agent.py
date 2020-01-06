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
        self.scorer = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_kv = nn.Linear(hidden_size, hidden_size)

    def forward(self, base_model, instances, init_all_decision_vecs_tensor, sep_pos, init_pred_list, mode='single-step-train'):
        assert mode in ['train', 'single-step-train', 'eval']
        left_instances_idx = list(range(len(instances)))
        all_decision_vecs = []
        for i in range(len(instances)):
            decision_vec = init_all_decision_vecs_tensor[sep_pos[left_instances_idx[i]]:sep_pos[left_instances_idx[i] + 1]]
            all_decision_vecs.append(decision_vec)
        pred_list = init_pred_list
        while len(left_instances_idx) > 0:
            decision_vecs = []
            for left_instance_idx in left_instances_idx:
                decision_vec = all_decision_vecs[left_instance_idx]
                decision_vecs.append(decision_vec)
            next_sample_prob = self.choose_next(decision_vecs, [pred_list[idx] for idx in left_instances_idx])
            if mode != 'eval':
                picked_idx = random_pick(list(range(len(left_instances_idx))), next_sample_prob)
                selected_instance_idx = left_instances_idx.pop(picked_idx)
                # selected_instance = instances[selected_instance_idx]
                selected_decision_vec = all_decision_vecs[selected_instance_idx][pred_list[selected_instance_idx]]
                for left_instance_idx in left_instances_idx:
                    target_decision_vecs_tensor = all_decision_vecs[left_instance_idx]
                    updated_target_decision_vecs_tensor = self.update_decision_vec(selected_decision_vec, target_decision_vecs_tensor)
                    all_decision_vecs[left_instance_idx] = updated_target_decision_vecs_tensor
                new_logits = base_model.classifier(torch.cat(all_decision_vecs, dim=0))
                #pred_list = 1
            else:
                picked_idx = next_sample_prob.argmax(dim=0)

            if mode == 'single-step-train':
                break

        return new_logits

    def choose_next(self, decision_vecs, pred_list):
        num_instance = len(decision_vecs)
        #maxpooled_decision_vecs = []
        avgpooled_decision_vecs = []
        max_prob_decision_vecs = []
        for i in range(len(decision_vecs)):
            maxpooled_decision_vec = decision_vecs[i].max(dim=0)[0]
            avgpooled_decision_vec = decision_vecs[i].mean(dim=0)
            max_prob_decision_vec = decision_vecs[i][pred_list[i]]
            # maxpooled_decision_vecs.append(maxpooled_decision_vec)
            avgpooled_decision_vecs.append(avgpooled_decision_vec)
            max_prob_decision_vecs.append(max_prob_decision_vec)
        # stacked_maxpooled_decision_vecs = torch.stack(maxpooled_decision_vecs)
        stacked_avgpooled_decision_vecs = torch.stack(avgpooled_decision_vecs)
        stacked_max_prob_decision_vecs = torch.stack(max_prob_decision_vecs)
        # stacked_pooled_decision_vecs = stacked_maxpooled_decision_vecs - stacked_avgpooled_decision_vecs
        stacked_pooled_decision_vecs = stacked_max_prob_decision_vecs - stacked_avgpooled_decision_vecs
        score = self.scorer(stacked_pooled_decision_vecs)
        sample_prob = F.softmax(score.squeeze(1), dim=0)

        return sample_prob

    def update_decision_vec(self, selected_decision_vec, target_decision_vecs_tensor):
        d = selected_decision_vec.size(0)
        Q = self.W_q(selected_decision_vec.unsqueeze(0))
        K = self.W_kv(target_decision_vecs_tensor)
        attn = F.softmax(torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(d))
        #max_id = attn.argmax()
        #attn_ = attn.new_zeros(attn.size())
        delta_target_decision_vecs_tensor = torch.matmul(attn.transpose(0, 1), selected_decision_vec.unsqueeze(0))
        updated_target_decision_vecs_tensor = target_decision_vecs_tensor + delta_target_decision_vecs_tensor

        return updated_target_decision_vecs_tensor