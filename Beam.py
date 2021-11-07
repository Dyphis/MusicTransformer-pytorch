# Filename: Beam.py
# Date Created: 15-Mar-2019 2:42:12 pm
# Description: Functions used for beam search.
import torch
import torch.nn.functional as F
import math

def init_vars(src, out, max_len, k):
    # calculate probablites for beam search
    # takes the last output from the model, hence out[:, -1]
    probs, ix = out[:, -1].data.topk(k)
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)

    # store the model outputs
    #outputs = torch.zeros(opt.k, opt.max_seq_len).long().to(opt.device)
    outputs = torch.zeros(k, max_len)
    outputs[:, 0:src.shape[1]] = src
    outputs[:, src.shape[1]] = ix[0]

    return outputs, log_scores

def k_best_outputs(outputs, out, log_scores, i, k):
    # calculate probablities for each step in the sequence
    probs, ix = out[:, -1].data.topk(k)
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0,1)
    k_probs, k_ix = log_probs.view(-1).topk(k)

    row = k_ix // k
    col = k_ix % k

    # update outputs
    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]
    #new_outputs = torch.zeros(outputs.shape[0], outputs.shape[1]+1)
    #new_outputs[:,:outputs.shape[1]] = outputs[row.tolist(),:outputs.shape[1]]
    #new_outputs[:,outputs.shape[1]] = ix[row.tolist(),col.tolist()]

    log_scores = k_probs.unsqueeze(0)

    return outputs, log_scores
