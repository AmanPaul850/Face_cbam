import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import os
from vgg16_cbam import VGG16_CBAM_IQA
import numpy as np
from scipy.stats import spearmanr, pearsonr
import time
import argparse
from datasets import get_dataset
from torch.utils.tensorboard import SummaryWriter
import datetime
from plot_metrics import MetricsPlotter

class EMDLoss(nn.Module):
    def __init__(self):
        super(EMDLoss, self).__init__()
        
    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        
        pred_safe = torch.clamp(pred, min=1e-7, max=1e7)
        
        pred_sorted, _ = torch.sort(pred_safe)
        target_sorted, _ = torch.sort(target)
        
        return torch.abs(pred_sorted - target_sorted).mean()

def calculate_metrics(pred, target):
    pred = pred.flatten()
    target = target.flatten()
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    plcc = pearsonr(pred, target)[0]
    srcc = spearmanr(pred, target)[0]
    return rmse, plcc, srcc

def train_model(regression_type='ridge', dataset_name='faceiqa'):
    log_dir = 'training_logs'
    os.makedirs(log_dir, exist_ok=True)
    model_dir = f'models_{regression_type}'
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(log_dir, f'training_log_{regression_type}_{timestamp}.txt')
    
    def log_print(*args, **kwargs):
        print(*args, **kwargs)
        with open(log_filename, 'a') as f:
            print(*args, file=f, **kwargs)
    
    log_print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_print("="*50)
    
    dataset = get_dataset(dataset_name)
    log_print(f"Dataset: {dataset_name}")
    log_print(f"Total dataset size: {len(dataset)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"Using device: {device}")
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    log_print(f"Training set size: {train_size}")
    log_print(f"Validation set size: {val_size}")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    model = VGG16_CBAM_IQA(regression_type=regression_type).to(device)
    criterion = EMDLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    
    log_print("Starting training...")
    
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = f'runs/{dataset_name}_{regression_type}_{current_time}'
    writer = SummaryWriter(log_dir)
    log_print(f"TensorBoard logs directory: {log_dir}")
    
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    writer.add_graph(model, dummy_input)
    
    num_epochs = 100
    best_val_loss = float('inf')
    
    # Initialize MetricsPlotter
    metrics_plotter = MetricsPlotter(regression_type)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        log_print(f"Starting Epoch {epoch + 1}/{num_epochs}")
        
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                log_print(f'Batch: {batch_idx}/{len(train_loader)}')
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                val_predictions.extend(output.cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        rmse, plcc, srcc = calculate_metrics(
            np.array(val_predictions),
            np.array(val_targets)
        )
        
        writer.add_scalars('Loss', {
            'train': train_loss,
            'val': val_loss
        }, epoch)
        
        writer.add_scalars('Metrics', {
            'RMSE': rmse,
            'PLCC': plcc,
            'SRCC': srcc
        }, epoch)
        
        epoch_time = time.time() - epoch_start_time
        log_print(f'Epoch {epoch + 1} Results (Time: {epoch_time:.2f}s):')
        log_print(f'Train Loss: {train_loss:.6f}')
        log_print(f'Val Loss: {val_loss:.6f}')
        log_print(f'RMSE: {rmse:.4f}')
        log_print(f'PLCC: {plcc:.4f}')
        log_print(f'SRCC: {srcc:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'rmse': rmse,
                'plcc': plcc,
                'srcc': srcc,
            }, os.path.join(model_dir, f'best_model_{regression_type}.pth'))
            log_print("Saved best model!")
        
        scheduler.step(val_loss)

        # Log epoch-level metrics
        metrics_plotter.update_metrics(train_loss, val_loss, rmse, plcc, srcc, epoch)

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CBAM model')
    parser.add_argument('--regression_type', 
                       type=str, 
                       default='simple',
                       choices=['simple', 'elastic', 'ridge'],
                       help='Type of regression to use')
    parser.add_argument('--dataset', 
                       type=str, 
                       default='koniq10k',
                       choices=['koniq10k', 'faceiqa' ,'kadid10k'],
                       help='Dataset to use for training')
    
    args = parser.parse_args()
    
    try:
        train_model(regression_type=args.regression_type, 
                   dataset_name=args.dataset)
    except Exception as e:
        print(f"Error during training: {e}") 
