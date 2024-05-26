import torch
import torch.nn.functional as F

def loss_D(D_real_output, D_fake_output):
    D_fake_loss = torch.mean(D_fake_output ** 2)
    D_real_loss = torch.mean((1 - D_real_output) ** 2)
    D_total_loss = D_fake_loss + D_real_loss
    return D_total_loss, D_real_loss, D_fake_loss
def l1_loss(output, target):
    loss = torch.mean(torch.abs(output - target))
    return loss
def huber_reverse_loss(pred, label, delta=0.2, adaptive=True):
    diff = torch.abs(pred - label)
    if adaptive:
        delta = delta * torch.std(label)  # batch-adaptive
    loss = torch.mean(
        (diff <= delta).float() * diff +
        (diff > delta).float() * (diff**2 / (2 * delta) + delta / 2)
    )
    return loss
#def loss_G(D_fake_output, G_outputs, target_transformed, train_config, cur_epoch):