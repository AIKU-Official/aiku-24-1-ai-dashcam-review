# a_0 : Probability of Accident
# a_1 : Probability of Non-Accident
# p(prob, frame)
# t(prob, frame)

import torch
import torch.nn.functional as F


class Loss():
  def __init__(self):
    pass

  def prediction(self, pred):  # [100]
    stacked_pred = torch.stack([torch.tensor([p, frame]) for frame, p in enumerate(pred)])

    return stacked_pred


  def CrossEntropyLoss(self, pred):  # for negative
    label = torch.stack([p[0] for p in pred])
    loss = -torch.sum(torch.log(1-label))
    loss.requires_grad = True

    return loss


  def AnticipationLoss(self, pred):  # for positive
    label = torch.stack([p[0] for p in pred])  # pred prob
    timestep = torch.stack([p[1] for p in pred]).to(int)
    loss = -torch.sum(torch.exp(-torch.maximum(torch.tensor(0), (90 - timestep)))*torch.log(label))
    loss.requires_grad = True

    return loss