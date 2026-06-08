import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # 或使用 torch.utils.tensorboard
from tqdm import tqdm

from src.dataloader import LoDoPaBDataset, get_fastmri_dataloader
from src.network import LearnedOptimizerNet
from src.operators import get_operators  # 假設你上一步將算子封裝進 get_operators
from src.optimization import optimize

def get_args():
    parser = argparse.ArgumentParser(description="Unrolled Optimization Training & Testing Framework")
    # 實驗設定
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='執行模式: train 或 test')
    parser.add_argument('--train_task', type=str, default='CT', choices=['CT', 'MRI'], help='訓練使用的任務/資料集')
    parser.add_argument('--test_task', type=str, default='MRI', choices=['CT', 'MRI'], help='測試使用的任務/資料集 (可用於跨資料集實驗)')
    
    # 路徑設定
    parser.add_argument('--ct_dir', type=str, default='./data/lodopab', help='LoDoPaB CT 資料夾路徑')
    parser.add_argument('--mri_dir', type=str, default='./data/fastmri', help='fastMRI 資料夾路徑')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='模型儲存夾')
    parser.add_argument('--log_dir', type=str, default='./runs/experiment_1', help='TensorBoard 紀錄夾')
    parser.add_argument('--resume_weights', type=str, default=None, help='要載入測試或繼續訓練的模型權重路徑 (.pth)')
    
    # 超參數
    parser.add_argument('--epochs', type=int, default=20, help='訓練總 Epoch 數')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch 大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='學習率')
    parser.add_argument('--n_iter', type=int, default=10, help='展開式演算法內部迭代次數 (Unrolled steps)')
    parser.add_argument('--algo_type', type=str, default='learned_smooth', choices=['learned_smooth', 'learned_nonsmooth'], help='演算法類型')
    
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    # -----------------------------------------------------------------
    # 1. 初始化神經網路 (通道數與架構自動對齊)
    # -----------------------------------------------------------------
    # learned_smooth 需要 1 個網路；learned_nonsmooth 論文中需要 2 個網路
    if args.algo_type == 'learned_smooth':
        # 輸入: x(1) + grad_f(1) + grad_h(1) + dx(1) = 4 Channels
        net = LearnedOptimizerNet(in_channels=4, n_layers=3).to(device)
        networks_param = net
    else:
        # learned_nonsmooth 邏輯
        net1 = LearnedOptimizerNet(in_channels=3, n_layers=3).to(device)
        net2 = LearnedOptimizerNet(in_channels=4, n_layers=3).to(device)
        networks_param = (net1, net2)

    # 如果有指定權重，則載入 (不論是 test 還是 resume train 都要用)
    if args.resume_weights:
        print(f"Loading weights from {args.resume_weights}")
        if args.algo_type == 'learned_smooth':
            net.load_state_dict(torch.load(args.resume_weights, map_location=device))
        else:
            # 假設 nonsmooth 儲存時是用 dict 存兩個網路
            checkpoint = torch.load(args.resume_weights, map_location=device)
            net1.load_state_dict(checkpoint['net1'])
            net2.load_state_dict(checkpoint['net2'])

    # -----------------------------------------------------------------
    # 執行模式 [TRAIN] 
    # -----------------------------------------------------------------
    if args.mode == 'train':
        print(f"=== Starting Training on {args.train_task} ===")
        
        # 載入訓練用的物理算子與資料集
        # 這裡假設你的 operators.py 裡面有 get_operators(task_type, regularizer_type, device)
        T_train, W_train = get_operators(task_type=args.train_task, regularizer_type="smooth", device=device)
        
        if args.train_task == 'CT':
            train_dataset = LoDoPaBDataset(data_dir=args.ct_dir, mode='train')
            val_dataset = LoDoPaBDataset(data_dir=args.ct_dir, mode='validation')
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        else: # MRI
            train_loader = get_fastmri_dataloader(data_dir=os.path.join(args.mri_dir, 'train'), batch_size=args.batch_size, mode='train')
            val_loader = get_fastmri_dataloader(data_dir=os.path.join(args.mri_dir, 'val'), batch_size=args.batch_size, mode='val')

        # 最佳化器（更新網路權重用）
        params = net.parameters() if args.algo_type == 'learned_smooth' else list(net1.parameters()) + list(net2.parameters())
        optimizer = optim.Adam(params, lr=args.lr)
        
        best_val_loss = float('inf')
        global_step = 0
        
        for epoch in range(args.epochs):
            if args.algo_type == 'learned_smooth': net.train()
            else: net1.train(); net2.train()
                
            train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")
            for x0, y, x_true in train_loop:
                x0, y, x_true = x0.to(device), y.to(device), x_true.to(device)
                
                optimizer.zero_grad()
                
                # 執行展開式優化演算法
                x_recon, obj_loss, _ = optimize(
                    algorithm=args.algo_type,
                    x=x0, y=y, T=T_train, W=W_train,
                    networks=networks_param,
                    n_iter=args.n_iter
                )
                
                # 類神經網路的訓練損失：重建圖與真實圖的 MSE
                loss = torch.mean((x_recon - x_true) ** 2)
                
                loss.backward()
                optimizer.step()
                
                writer.add_scalar("Loss/Train_Step", loss.item(), global_step)
                train_loop.set_postfix(loss=loss.item())
                global_step += 1
            
            # --- Validation 驗證 ---
            if args.algo_type == 'learned_smooth': net.eval()
            else: net1.eval(); net2.eval()
                
            val_loss_cum = 0.0
            with torch.no_grad():
                for x0, y, x_true in val_loader:
                    x0, y, x_true = x0.to(device), y.to(device), x_true.to(device)
                    x_recon, _, _ = optimize(
                        algorithm=args.algo_type,
                        x=x0, y=y, T=T_train, W=W_train,
                        networks=networks_param,
                        n_iter=args.n_iter
                    )
                    val_loss_cum += torch.mean((x_recon - x_true) ** 2).item()
            
            avg_val_loss = val_loss_cum / len(val_loader)
            writer.add_scalar("Loss/Validation_Epoch", avg_val_loss, epoch)
            print(f"--> Epoch {epoch+1} Done. Val Loss: {avg_val_loss:.6f}")
            
            # 儲存最佳模型 (Best Checkpoint)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = os.path.join(args.checkpoint_dir, "best_model.pth")
                if args.algo_type == 'learned_smooth':
                    torch.save(net.state_dict(), save_path)
                else:
                    torch.save({'net1': net1.state_dict(), 'net2': net2.state_dict()}, save_path)
                print(f"Saved new best model to {save_path}")

    # -----------------------------------------------------------------
    # 執行模式 [TEST] (支援跨資料集 Zero-shot 測試)
    # -----------------------------------------------------------------
    elif args.mode == 'test':
        print(f"=== Starting Testing on {args.test_task} ===")
        if not args.resume_weights:
            print("Warning: Running test mode without loading pre-trained weights!")

        # 關鍵：載入測試任務的物理算子（即使跟訓練不同也沒關係）
        T_test, W_test = get_operators(task_type=args.test_task, regularizer_type="smooth", device=device)
        
        if args.test_task == 'CT':
            test_dataset = LoDoPaBDataset(data_dir=args.ct_dir, mode='test')
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        else: # MRI
            test_loader = get_fastmri_dataloader(data_dir=os.path.join(args.mri_dir, 'test'), batch_size=args.batch_size, mode='test')

        if args.algo_type == 'learned_smooth': net.eval()
        else: net1.eval(); net2.eval()
            
        test_loss_cum = 0.0
        with torch.no_grad():
            for idx, (x0, y, x_true) in enumerate(tqdm(test_loader, desc="Testing")):
                x0, y, x_true = x0.to(device), y.to(device), x_true.to(device)
                
                # 直接套用載入的模型去解測試任務的優化問題
                x_recon, _, _ = optimize(
                    algorithm=args.algo_type,
                    x=x0, y=y, T=T_test, W=W_test,
                    networks=networks_param,
                    n_iter=args.n_iter
                )
                
                batch_loss = torch.mean((x_recon - x_true) ** 2).item()
                test_loss_cum += batch_loss
                
                # 可以在這裡加入儲存重建影像圖片的程式碼 (例如使用 matplotlib 或 torchvision.utils.save_image)
                
        avg_test_loss = test_loss_cum / len(test_loader)
        print(f"\n=========================================")
        print(f"Final Test Loss on {args.test_task}: {avg_test_loss:.6f}")
        print(f"=========================================")

if __name__ == '__main__':
    main()