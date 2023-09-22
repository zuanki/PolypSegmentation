import torch


def jaccard_index(input, target):
    """
    Computes the Jaccard index between input and target tensors.

    Args:
        input (torch.Tensor): Input tensor with shape [batch_size, 1, height, width]
        target (torch.Tensor): Target tensor with shape [batch_size, 1, height, width]

    Returns:
        float: Jaccard index value
    """
    intersection = torch.sum(input * target).item()
    union = torch.sum(input) + torch.sum(target) - intersection

    if union == 0:
        return float("nan")
    else:
        return float(intersection) / float(max(union, 1))


def dice_coeff(input, target):
    """
    Computes the Dice coefficient between input and target tensors.

    Args:
        input (torch.Tensor): Input tensor with shape [batch_size, 1, height, width]
        target (torch.Tensor): Target tensor with shape [batch_size, 1, height, width]

    Returns:
        float: Dice coefficient value
    """
    smooth = 1.0

    num = input.size(0)

    pred = input.view(num, -1)
    truth = target.view(num, -1)

    intersection = torch.sum(pred * truth, dim=1)
    dice = (2.0 * intersection + smooth) / \
        (torch.sum(pred, dim=1) + torch.sum(truth, dim=1) + smooth)

    return dice.mean().item()
