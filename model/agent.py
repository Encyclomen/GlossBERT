import copy
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
    def __init__(self, hidden_size, init_classifier):
        super(Agent, self).__init__()
        self.scorer = nn.Sequential(nn.Linear(hidden_size+hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))
        self.gru1 = nn.GRUCell(hidden_size+hidden_size, hidden_size)
        self.gru2 = nn.GRUCell(hidden_size, hidden_size)
        #self.W1 = nn.Linear(hidden_size, hidden_size)
        #self.W2 = nn.Linear(hidden_size, hidden_size)
        #self.Wa = nn.Linear(hidden_size, 1)
        self.V1 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.V2 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.classifier = copy.deepcopy(init_classifier)

    def forward(self, base_model, instances, init_all_decision_vecs_tensor, mention_aware_gloss_tensor, sep_pos, init_pred_list, mode='single-step-train', num_sample=5):
        assert mode in ['train', 'single-step-train', 'eval']
        left_instances_idx = list(range(len(instances)))
        all_decision_vec_tensors_list = []
        all_mention_aware_gloss_tensors_list = []
        for i in range(len(instances)):
            decision_vec_tensor = init_all_decision_vecs_tensor[sep_pos[left_instances_idx[i]]:sep_pos[left_instances_idx[i]+1]]
            all_decision_vec_tensors_list.append(decision_vec_tensor)
            all_mention_aware_gloss_tensors_list.append(mention_aware_gloss_tensor[sep_pos[left_instances_idx[i]]:sep_pos[left_instances_idx[i]+1]])
        pred_list = init_pred_list
        #sep_len = [sep_pos[i+1] - sep_pos[i] for i in range(len(sep_pos)-1)]
        while len(left_instances_idx) > 0:
            decision_vec_tensors_list = []
            mention_aware_gloss_tensors_list = []
            for left_instance_idx in left_instances_idx:
                decision_vec_tensor = all_decision_vec_tensors_list[left_instance_idx]
                decision_vec_tensors_list.append(decision_vec_tensor)
                mention_aware_gloss_tensors_list.append(all_mention_aware_gloss_tensors_list[left_instance_idx])
            next_sample_probs = self.choose_next(decision_vec_tensors_list, mention_aware_gloss_tensors_list, [pred_list[idx] for idx in left_instances_idx])
            if mode != 'eval':
                new_logits_list = []
                selected_instance_idx_list = []
                for i in range(num_sample):
                    tmp_left_instances_idx = copy.deepcopy(left_instances_idx)
                    tmp_all_decision_vec_tensors_list = copy.deepcopy(all_decision_vec_tensors_list)
                    picked_idx = random_pick(list(range(len(tmp_left_instances_idx))), next_sample_probs)
                    selected_instance_idx = tmp_left_instances_idx.pop(picked_idx)
                    # selected_instance = instances[selected_instance_idx]
                    selected_decision_vec_tensor = tmp_all_decision_vec_tensors_list[selected_instance_idx][
                        pred_list[selected_instance_idx]]
                    selected_gloss_vec_tensor = mention_aware_gloss_tensor[
                        sep_pos[selected_instance_idx] + pred_list[selected_instance_idx]]
                    for left_instance_idx in tmp_left_instances_idx:
                        target_decision_vecs_tensor = tmp_all_decision_vec_tensors_list[left_instance_idx]
                        target_gloss_vecs_tensor = mention_aware_gloss_tensor[
                                                   sep_pos[left_instance_idx]:sep_pos[left_instance_idx + 1]]
                        updated_target_decision_vecs_tensor = self.update_decision_vec(selected_decision_vec_tensor,
                                                                                       selected_gloss_vec_tensor,
                                                                                       target_decision_vecs_tensor,
                                                                                       target_gloss_vecs_tensor)
                        tmp_all_decision_vec_tensors_list[left_instance_idx] = updated_target_decision_vecs_tensor
                    new_logits = self.classifier(torch.cat(tmp_all_decision_vec_tensors_list, dim=0))
                    new_logits_list.append(new_logits)
                    selected_instance_idx_list.append(selected_instance_idx)
                if mode == 'single-step-train':
                    return new_logits_list, next_sample_probs, selected_instance_idx_list
            else:
                #new_sep_pos = [0]
                #for i in range(len(left_instances_idx)):
                    #new_sep_pos.append(new_sep_pos[i]+sep_len[left_instances_idx[i]])
                picked_idx = next_sample_probs.argmax(dim=0)
                selected_instance_idx = left_instances_idx.pop(int(picked_idx))
                # selected_instance = instances[selected_instance_idx]
                selected_decision_vec_tensor = all_decision_vec_tensors_list[selected_instance_idx][pred_list[selected_instance_idx]]
                selected_gloss_vec_tensor = mention_aware_gloss_tensor[
                    sep_pos[selected_instance_idx] + pred_list[selected_instance_idx]]

                for left_instance_idx in left_instances_idx:
                    target_decision_vecs_tensor = all_decision_vec_tensors_list[left_instance_idx]
                    target_gloss_vecs_tensor = mention_aware_gloss_tensor[
                                               sep_pos[left_instance_idx]:sep_pos[left_instance_idx+1]]
                    updated_target_decision_vecs_tensor = self.update_decision_vec(selected_decision_vec_tensor,
                                                                                   selected_gloss_vec_tensor,
                                                                                   target_decision_vecs_tensor,
                                                                                   target_gloss_vecs_tensor)
                    all_decision_vec_tensors_list[left_instance_idx] = updated_target_decision_vecs_tensor

                new_logits = base_model.classifier(torch.cat(all_decision_vec_tensors_list, dim=0))
                new_probs = F.softmax(new_logits, dim=-1)

                for i in range(len(instances)):
                    new_pred = new_probs[sep_pos[i]: sep_pos[i+1], 1].argmax().item()
                    pred_list[i] = new_pred

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
        #selected_mention_gloss_vec_tensor = torch.cat((selected_decision_vec_tensor, selected_gloss_vec_tensor), dim=-1)
        target_mention_gloss_vecs_tensor = torch.cat((target_decision_vecs_tensor, target_gloss_vecs_tensor), dim=-1)
        N, d = target_mention_gloss_vecs_tensor.size()
        '''
        attn_h = self.W1(selected_gloss_vec_tensor.unsqueeze(0)).expand(N, -1) + self.W2(target_gloss_vecs_tensor)
        logits = self.Wa(torch.tanh(attn_h))
        attn = F.softmax(logits, dim=0)
        delta_target_decision_vecs_tensor = self.U(torch.relu((self.V1(selected_mention_gloss_vec_tensor)+self.V2(target_mention_gloss_vecs_tensor))))
        updated_target_decision_vecs_tensor = target_decision_vecs_tensor + attn * delta_target_decision_vecs_tensor
        '''
        proposal = self.gru1(target_mention_gloss_vecs_tensor, selected_gloss_vec_tensor.unsqueeze(0).expand(N, -1))
        updated_target_decision_vecs_tensor = self.gru2(proposal, selected_decision_vec_tensor.unsqueeze(0).expand(N, -1))

        return updated_target_decision_vecs_tensor