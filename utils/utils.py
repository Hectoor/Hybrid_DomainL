import torch
import torch.nn.functional as F


#
def eightway_affinity_kld(probs, size=1):
    b, c, h, w = probs.size()
    if probs.dim() != 4:
        raise Exception('Only support for 4-D tensors!')
    p = size
    probs_pad = F.pad(probs, [p]*4, mode='replicate')
    bot_epsilon = 1e-4
    top_epsilon = 1.0
    neg_probs_clamp = torch.clamp(1.0 - probs, bot_epsilon, top_epsilon)
    probs_clamp = torch.clamp(probs, bot_epsilon, top_epsilon)
    kldiv_groups = []
    for st_y in range(0, 2*size+1, size):  # 0, size, 2*size
        for st_x in range(0, 2*size+1, size):
            if st_y == size and st_x == size:
                # Ignore the center pixel/feature.
                continue
            probs_paired = probs_pad[:, :, st_y:st_y+h, st_x:st_x+w] * probs
            neg_probs_paired = torch.clamp(
                1.0-probs_paired, bot_epsilon, top_epsilon)
            probs_paired = torch.clamp(probs_paired, bot_epsilon, top_epsilon)
            kldiv = probs_paired * torch.log(probs_paired/probs_clamp) \
                + neg_probs_paired * \
                torch.log(neg_probs_paired/neg_probs_clamp)
            kldiv_groups.append(kldiv)
    return torch.cat(kldiv_groups, dim=1)

