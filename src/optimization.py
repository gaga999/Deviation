import torch
import numpy as np

# 導入先前的模組 (假設它們分別存放在對應的檔案)
from .operators import prox

#################################
# Optimize |Tx-y|_2 + lam*|Wx|_1
#################################

# 論文中的最佳化超參數
lam_smooth = 0.0015
lam_nonsmooth = 0.0005
gamma_n = 0.5
beta = 0.5
alpha = 0.999
omega = alpha
use_dx2 = True
Huber_param = 0.01

# =====================================================================
# 輔助函數 (Helpers) - 注意 PyTorch 維度為 (B, C, H, W)
# =====================================================================

def dot(x, y):
    # 針對 NCHW 格式，計算除 Batch 外的內積
    return torch.sum(x * y, dim=[1, 2, 3], keepdim=True)

def norm_sq(x):
    return torch.sum(x**2, dim=[1, 2, 3], keepdim=True)

def norm(x):
    # 避免梯度為 0 時產生 NaN，加上極小值 1e-8
    return torch.sqrt(norm_sq(x) + 1e-8)

def safely_normalize(x):
    # 對應原版的 safely_normalize
    safe_sqrt = torch.sqrt(norm_sq(x) + 1.0)
    return x / safe_sqrt

# =====================================================================
# 損失函數與 Huber 梯度
# =====================================================================

def l2(x, y):
    return torch.mean(norm_sq(x - y))

def l1(x):
    """
    計算 L1 損失。支援一般 Tensor 或小波轉換輸出的巢狀 List/Tuple
    """
    if isinstance(x, list) or isinstance(x, tuple):
        loss = 0.0
        for item in x:
            loss += l1(item)
        return loss
    
    # 針對單一張量，拉平後算 L1 總和，再對 Batch 取平均
    B = x.shape[0]
    x_flat = x.view(B, -1)
    return torch.mean(torch.sum(torch.abs(x_flat), dim=-1))

def huber_grad(x, gamma):
    # 使用 torch.where 實作 Huber 梯度
    return torch.where(torch.abs(x) < gamma, x / gamma, torch.sign(x))

def huber(x, gamma):
    # 使用 torch.where 實作 Huber 損失
    B = x.shape[0]
    y = torch.where(torch.abs(x) < gamma, 
                    0.5 * x**2 / gamma, 
                    torch.abs(x) - 0.5 * gamma)
    y_flat = y.view(B, -1)
    return torch.mean(torch.sum(y_flat, dim=-1))

# =====================================================================
# 學習型最佳化演算法 (Learned Optimization)
# =====================================================================

def learned_opt_smooth(x, y, T, W, network, lam, n_iter):
    h_norm = 8.0 / Huber_param
    
    xn = x.clone()
    xn_prev = x.clone()
    dx = torch.zeros_like(x)
    
    losses = []

    for i in range(n_iter):
        # 1. 計算梯度
        grad_f = 2 * T.adjoint(T.forward(xn) - y)
        grad_h = lam * W.adjoint(huber_grad(W.forward(xn), Huber_param))
        grad = grad_f + grad_h
        
        # 2. 網路預測步長
        # 注意：PyTorch 的 Concat 是在 Channel 維度 (dim=1)
        inp = torch.cat([xn, grad_f, grad_h, dx], dim=1)
        dx = network(inp)
        dx_n = safely_normalize(dx) * alpha * norm(grad)
        
        # 3. 狀態更新
        xn_prev = xn.clone()
        xn = xn - (1.0 / (2.0 + lam * h_norm)) * (grad + dx_n)
        
        # 4. 記錄 Loss
        loss_f = l2(T.forward(xn), y)
        loss_g = lam * huber(W.forward(xn), Huber_param)
        loss_step = loss_f + loss_g
        losses.append(loss_step.item())

    # 回傳最終影像與損失
    final_loss = l2(T.forward(xn), y) + lam * huber(W.forward(xn), Huber_param)
    return xn, final_loss, losses


def learned_opt_nonsmooth(x, y, T, W, network1, network2, lam, mu, n_iter):
    xn = x.clone()
    xn_prev = x.clone()
    grad_f = 2 * T.adjoint(T.forward(x) - y)
    
    dx1 = torch.zeros_like(x)
    dx2 = torch.zeros_like(x)
    
    losses = []

    for i in range(n_iter):
        dx1_prev = dx1.clone()
        dx2_prev = dx2.clone()
        
        # 1. 網路預測
        inp1 = torch.cat([xn, grad_f, dx1], dim=1)
        dx1 = network1(inp1)
        
        if use_dx2 and network2 is not None:
            inp2 = torch.cat([xn, grad_f, dx2, dx1], dim=1)
            dx2 = network2(inp2)
            
        # 2. 正規化網路輸出 (完全保留原版的係數計算)
        c1 = np.sqrt(2 * beta * alpha * (2 * beta - gamma_n) / (2 * beta * gamma_n))
        c1 *= norm(xn - xn_prev - beta / (2 * beta - gamma_n) * dx2_prev)
        dx1_n = safely_normalize(dx1) * c1
        
        yn = xn + dx1_n
        grad_f_prev = grad_f.clone()
        grad_f = 2 * T.adjoint(T.forward(yn) - y)
        
        if use_dx2 and network2 is not None:
            c2 = np.sqrt(2 * gamma_n * (2 * beta - gamma_n) / (beta * beta * omega / 2))
            c2 *= norm(grad_f - grad_f_prev - 1.0 / beta * (xn - xn_prev - dx1_prev))
            dx2_n = safely_normalize(dx2) * c2
        else:
            dx2_n = dx2_prev

        # 3. 更新 xn
        xn_prev = xn.clone()
        xn = xn - gamma_n * grad_f
        xn = xn + (gamma_n / beta) * dx1_n
        xn = xn + dx2_n
        
        # 套用近端算子 (Proximal Operator)
        xn = prox(xn, W, W, gamma_n, lam, mu) # 注意 W 的 adjoint 內建在 W_adj 中，這裡呼叫統一的 W
        
        # 4. 記錄 Loss
        loss_f = l2(T.forward(xn), y)
        loss_g = lam * l1(W.forward(xn))
        losses.append((loss_f + loss_g).item())

    final_loss = l2(T.forward(xn), y) + lam * l1(W.forward(xn))
    return xn, final_loss, losses

# =====================================================================
# 基線演算法 (Baseline Optimization) - ISTA / FISTA / Steep Descent
# =====================================================================
# 為了篇幅精簡，此處示範最經典的 FISTA。其餘基線演算法邏輯相同，皆改成 for 迴圈。

def fista_opt(x, y, T, W, lam, mu, n_iter):
    xn = x.clone()
    yn = x.clone()
    tn = torch.tensor(1.0, dtype=torch.float32, device=x.device)
    losses = []

    for i in range(n_iter):
        xn_prev = xn.clone()
        grad_f = 2 * T.adjoint(T.forward(yn) - y)
        xn = yn - gamma_n * grad_f
        
        # 套用近端算子
        xn = prox(xn, W, W, gamma_n, lam, mu)
        
        tn_prev = tn.clone()
        tn = (1.0 + torch.sqrt(1.0 + 4.0 * tn**2)) / 2.0
        yn = xn + ((tn_prev - 1.0) / tn) * (xn - xn_prev)
        
        losses.append((l2(T.forward(xn), y) + lam * l1(W.forward(xn))).item())

    final_loss = l2(T.forward(xn), y) + lam * l1(W.forward(xn))
    return xn, final_loss, losses

# =====================================================================
# 統一入口函數
# =====================================================================

def optimize(algorithm, x, y, T, W, networks=None, n_iter=10):
    """
    PyTorch 版的總接口。
    :param networks: 對於 learned_smooth 需傳入單一 model，
                     對於 learned_nonsmooth 需傳入 (net1, net2) 的 tuple。
    """
    # 對應 TF 中的 mu = W_odl.norm()**2，在 PyTorch 中如果算子有 scale 或是正交的，這裡 mu 可視為 1.0 或由 Power Iteration 取得
    # 為簡單起見，我們假設 mu = 1.0 (如果 W 是 Sym5 小波)
    mu = 1.0 
    
    if algorithm == "learned_smooth":
        net = networks
        return learned_opt_smooth(x, y, T, W, net, lam_smooth, n_iter)
        
    elif algorithm == "learned_nonsmooth":
        net1, net2 = networks
        return learned_opt_nonsmooth(x, y, T, W, net1, net2, lam_nonsmooth, mu, n_iter)
        
    elif algorithm == "fista":
        return fista_opt(x, y, T, W, lam_nonsmooth, mu, n_iter)
        
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")