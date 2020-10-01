import torch
import numpy as np

smoothing_lambda = 1.25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class torch_loss_centroid(torch.autograd.Function):

    @staticmethod
    def forward(ctx, location_pred, location_true):

        ctx.save_for_backward(location_pred, location_true)

        loss = 0
        for batch_idx in range(location_pred.size()[0]):

            x_true, y_true = location_true[batch_idx, :, 0], location_true[batch_idx, :, 1]
            x_pred, y_pred = location_pred[batch_idx, :, 0], location_pred[batch_idx, :, 1]

            len_true, len_pred = x_true.size()[0], x_pred.size()[0]

            term1 = np.pi * np.square(
                smoothing_lambda) / 2 * torch.exp(
                - ((x_true.unsqueeze(0).expand(len_true, -1) - x_true.unsqueeze(0).expand(len_true, -1).transpose(0,
                                                                                                                  1)) ** 2
                   + (y_true.unsqueeze(0).expand(len_true, -1) - y_true.unsqueeze(0).expand(len_true, -1).transpose(
                            0,
                            1)) ** 2)
                / (2 * np.square(smoothing_lambda)))

            term2 = np.pi * np.square(smoothing_lambda) / 2 * torch.exp(
                - ((x_pred.unsqueeze(0).expand(len_pred, -1) - x_pred.unsqueeze(0).expand(len_pred, -1).transpose(0,
                                                                                                                  1)) ** 2
                   + (y_pred.unsqueeze(0).expand(len_pred, -1) - y_pred.unsqueeze(0).expand(len_pred, -1).transpose(
                            0,
                            1)) ** 2)
                / (2 * np.square(smoothing_lambda)))

            term3 = np.pi * np.square(smoothing_lambda) / 2 * torch.exp(
                - ((x_pred.unsqueeze(0).expand(len_true, -1) - x_true.unsqueeze(0).expand(len_pred, -1).transpose(0,
                                                                                                                  1)) ** 2
                   + (y_pred.unsqueeze(0).expand(len_true, -1) - y_true.unsqueeze(0).expand(len_pred, -1).transpose(
                            0,
                            1)) ** 2)
                / (2 * np.square(smoothing_lambda)))

            for channel_idx in range(location_pred.size()[2] - 2):
                p_pred = location_pred[batch_idx, :, 2 + channel_idx]
                p_true = location_true[batch_idx, :, 2 + channel_idx]

                loss += torch.sum(p_true.unsqueeze(0) * p_true.unsqueeze(0).transpose(0, 1) * term1) \
                        + torch.sum(p_pred.unsqueeze(0) * p_pred.unsqueeze(0).transpose(0, 1) * term2) \
                        - 2 * torch.sum(p_pred.unsqueeze(0).expand(len_true, -1) * p_true.unsqueeze(0).expand(len_pred, -1).transpose(0,1) * term3)

        return loss

    @staticmethod
    def backward(ctx, grad):

        location_pred, location_true, = ctx.saved_tensors

        FullGradient = []
        for batch_idx in range(location_pred.size()[0]):

            x_pred, y_pred = location_pred[batch_idx, :, 0], location_pred[batch_idx, :, 1]
            x_true, y_true = location_true[batch_idx, :, 0], location_true[batch_idx, :, 1]


            len_true, len_pred = x_true.size()[0], x_pred.size()[0]

            x_dist_pred_pred, y_dist_pred_pred = \
                x_pred.unsqueeze(0).expand(len_pred, -1) - x_pred.unsqueeze(0).expand(len_pred, -1).transpose(0, 1), \
                y_pred.unsqueeze(0).expand(len_pred, -1) - y_pred.unsqueeze(0).expand(len_pred, -1).transpose(0, 1)

            exp_dist_pred_pred = torch.exp(
                -(x_dist_pred_pred ** 2 + y_dist_pred_pred ** 2) / (2 * smoothing_lambda ** 2))

            x_dist_pred_true, y_dist_pred_true = \
                x_pred.unsqueeze(0).expand(len_true, -1) - x_true.unsqueeze(0).expand(len_pred, -1).transpose(0, 1), \
                y_pred.unsqueeze(0).expand(len_true, -1) - y_true.unsqueeze(0).expand(len_pred, -1).transpose(0, 1)

            exp_dist_pred_true = torch.exp(
                -(x_dist_pred_true ** 2 + y_dist_pred_true ** 2) / (2 * smoothing_lambda ** 2))

            gradients = 0
            for channel_idx in range(location_pred.size()[2] - 2):
                p_pred = location_pred[batch_idx, :, 2 + channel_idx]
                p_pred_ex_tr = p_pred.unsqueeze(0).expand(len_pred, -1).transpose(0, 1)

                p_true = location_true[batch_idx, :, 2 + channel_idx]
                p_true_ex_tr = p_true.unsqueeze(0).expand(len_pred, -1).transpose(0, 1)

                xx = np.pi * p_pred * \
                     (torch.sum(p_true_ex_tr * exp_dist_pred_true * x_dist_pred_true, axis=0)
                      - torch.sum(p_pred_ex_tr * exp_dist_pred_pred * x_dist_pred_pred, axis=0))

                yy = np.pi * p_pred * \
                     (torch.sum(p_true_ex_tr * exp_dist_pred_true * y_dist_pred_true, axis=0)
                      - torch.sum(p_pred_ex_tr * exp_dist_pred_pred * y_dist_pred_pred, axis=0))

                pp = np.pi * smoothing_lambda ** 2 * \
                     (torch.sum(p_pred_ex_tr * exp_dist_pred_pred, axis=0)
                      - torch.sum(p_true_ex_tr * exp_dist_pred_true, axis=0))


                pp_mask = (location_pred.size()[2] - 2) * [0]
                pp_mask[channel_idx] = 1

                pp_gradients = [x * pp.unsqueeze(1) for x in pp_mask]
                gradients += torch.cat([xx.unsqueeze(1), yy.unsqueeze(1)] + pp_gradients, axis=1)

            FullGradient.append(gradients.unsqueeze(0))

        return torch.cat(FullGradient, axis=0) * grad, location_true * 0



####### Counting Loss
def CountingLoss(pred, targets, threshold=0):

    loss_count = 0

    # cut-off for speed
    probs = pred[:, (torch.sum(pred[:, :, 2:],axis=(0,2))>threshold).type(torch.bool), 2:]

    contribution = torch.unbind(probs, 1)

    max_occurence = 50
    count_prediction = torch.FloatTensor(pred.size()[0], pred.size()[2] -2, max_occurence).fill_(0).to(device)
    count_prediction[:, :, 0] = 1  # (batch x class x max_occ)
    for increment in contribution:
        # mass_movement = (count_prediction * increment.unsqueeze(2))[:, :, :max_occurence - 1]
        # count_prediction[:, :, :max_occurence - 1] -= mass_movement
        # count_prediction[:, :, 1:] += mass_movement
        mass_movement = (count_prediction * increment.unsqueeze(2))[:, :, :max_occurence - 1]
        move = - torch.cat([mass_movement,
                            torch.cuda.FloatTensor(count_prediction.size()[0], count_prediction.size()[1], 1).fill_(
                                0)], axis=2) \
               + torch.cat(
            [torch.cuda.FloatTensor(count_prediction.size()[0], count_prediction.size()[1], 1).fill_(0),
             mass_movement], axis=2)

        count_prediction = count_prediction + move

    # Compute Target Counts
    target_counts = torch.sum(targets[:,:,2:],axis=1)

    # Compute Target Count Distributions
    target_distribution = torch.FloatTensor(count_prediction.size()[0], count_prediction.size()[1],max_occurence).fill_(0).to(device)
    for batch_idx in range(targets.size()[0]):
        for class_idx in range(targets.size()[2]-2):
            target_distribution[batch_idx, class_idx, target_counts[batch_idx, class_idx].type(torch.LongTensor)] = 1

    # Compare Estimated and Target distributions
    loss_count -= torch.sum(torch.log(count_prediction + 1e-12) * target_distribution)

    return loss_count


