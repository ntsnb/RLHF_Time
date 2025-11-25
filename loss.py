# losses.py 里先放：
class SFTLoss(nn.Module): 
    ...  # 简版

class PairWiseLoss(nn.Module):
    ...  # 简版

class PolicyLoss(nn.Module):
    ...  # 简版，仅支持 PPO

class ValueLoss(nn.Module):
    ...  # 简版 MSE 或 clipped MSE

class KDLoss(nn.Module):
    ...  # 如果你需要大模型蒸馏到小模型
