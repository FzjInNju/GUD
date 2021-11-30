import math
import os
import torch.nn.functional as F
import torch


def save_checkpoint(state, dataset, exp_name, filename='checkpoint.pth.tar'):
    """将检查点保存到磁盘"""
    directory = "runs/%s/%s/" % (dataset, exp_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)


class AverageMeter(object):
    """计算并存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """计算指定的 k 值"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reconstruction_loss(x_recon, x, distribution):
    """计算通用自动编码器框架的重建损失。
    Args:
        x_recon (Tensor): 重建任意形状的图像
        x (Tensor): 目标图像，形状与 x_recon 相同。
        distribution (str): 解码器的输出分布，为伯努利或高斯分布。
    """
    assert x_recon.size() == x.size()

    n = x.size(0)
    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(n)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(n)
    else:
        raise NotImplementedError('supported distributions: bernoulli/gaussian')

    return recon_loss


def mmd(z_tilde, z, z_var):
    """计算最大平均差异。

    Args:
        z_tilde (Tensor): 来自确定性非随机编码器 Q(Z|X) 的样本、二维张量（batch_size x 维度）
        z (Tensor): 来自先验分布的样本，与 z_tilde 形状相同。
        z_var (Number): 各向同性高斯先验 P(Z) 的标量方差。
    """
    assert z_tilde.size() == z.size()
    assert z.ndimension() == 2

    n = z.size(0)
    out = im_kernel_sum(z, z, z_var, exclude_diag=True).div(n * (n - 1)) + \
          im_kernel_sum(z_tilde, z_tilde, z_var, exclude_diag=True).div(n * (n - 1)) + \
          -im_kernel_sum(z, z_tilde, z_var, exclude_diag=False).div(n * n).mul(2)

    return out


def im_kernel_sum(z1, z2, z_var, exclude_diag=True):
    """计算逆多二次函数内核的样本度量总和。

    Args:
        z1 (Tensor): 来自多元高斯分布的一批样本，具有 z_var 的标量方差。
        z2 (Tensor): 来自另一个多元高斯分布的一批样本，具有 z_var 的标量方差。
        exclude_diag (bool): 在求和之前是否排除对角内核度量。
    """
    assert z1.size() == z2.size()
    assert z1.ndimension() == 2

    z_dim = z1.size(1)
    C = 2 * z_dim * z_var

    z11 = z1.unsqueeze(1).repeat(1, z2.size(0), 1)
    z22 = z2.unsqueeze(0).repeat(z1.size(0), 1, 1)

    kernel_matrix = C / (1e-9 + C + (z11 - z22).pow(2).sum(2))
    kernel_sum = kernel_matrix.sum()
    # numerically identical to the formulation. but..
    if exclude_diag:
        kernel_sum -= kernel_matrix.diag().sum()

    return kernel_sum


def log_density_igaussian(z, z_var):
    """计算给定 z 和 z_var 的零均值各向同性高斯分布的对数密度。"""
    assert z.ndimension() == 2
    assert z_var > 0

    z_dim = z.size(1)

    return -(z_dim / 2) * math.log(2 * math.pi * z_var) + z.pow(2).sum(1).div(-2 * z_var)


def multistep_lr_decay(optimizer, current_step, schedules):
    """手动 LR 调度器"""
    for step in schedules:
        if current_step == step:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / schedules[step]

    return optimizer


def cuda(tensor, uses_cuda):
    return tensor.cuda() if uses_cuda else tensor


def kl_divergence(mu, logvar):
    assert mu.size() == logvar.size()
    assert mu.size(0) != 0

    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean()
    mean_kld = klds.mean()
    dimension_wise_kld = klds.mean(0)

    return total_kld, mean_kld, dimension_wise_kld


def squared_distance(tensor1, tensor2):
    assert tensor1.size() == tensor2.size()
    assert tensor1.ndimension() == 2

    return (tensor1 - tensor2).pow(2).sum(1).mean()
