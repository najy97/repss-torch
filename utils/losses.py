# import package

# model
import torch
import torch.nn as nn

class neg_pear_loss(nn.Module):
    def __init__(self):
        super(neg_pear_loss, self).__init__()
        return

    def forward(self, preds, labels):
        loss = 0
        for i in range(preds.shape[0]):  # pred -> [batch x time_length]
            sum_x = torch.sum(preds[i])  # predict 배치의 i번째 요소의 합
            sum_y = torch.sum(labels[i])  # labels 배치의 i번째 요소의 합
            sum_xy = torch.sum(preds[i] * labels[i])  # *(아마다르 곱)
            sum_x_sq = torch.pow(preds[i], 2)
            sum_y_sq = torch.pow(labels[i], 2)
            t = preds.shape[1]

            num = t * sum_xy - sum_x * sum_y
            den = torch.sqrt((t * sum_x_sq - torch.pow(sum_x, 2)) * ((t * sum_y_sq) - torch.pow(sum_y, 2)))
            loss += 1 - num / den

        loss = loss / preds.shape[0]  # batch 평균 loss 반환
        return loss
