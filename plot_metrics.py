import matplotlib.pyplot as plt
import os

class MetricsPlotter:
    def __init__(self, regression_type):
        self.regression_type = regression_type
        self.train_losses = []
        self.val_losses = []
        self.rmse_values = []
        self.plcc_values = []
        self.srcc_values = []
        self.epochs = []
        
        # Create directory for plots
        os.makedirs('training_plots', exist_ok=True)
        
    def update_metrics(self, train_loss, val_loss, rmse, plcc, srcc, epoch):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.rmse_values.append(rmse)
        self.plcc_values.append(plcc)
        self.srcc_values.append(srcc)
        self.epochs.append(epoch)
        
        self.plot_metrics()
        
    def plot_metrics(self):
        plt.figure(figsize=(15, 10))
        
        # Loss plot
        plt.subplot(2, 2, 1)
        plt.plot(self.epochs, self.train_losses, label='Train Loss')
        plt.plot(self.epochs, self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{self.regression_type} Regression - Training and Validation Loss')
        plt.legend()
        
        # RMSE plot
        plt.subplot(2, 2, 2)
        plt.plot(self.epochs, self.rmse_values)
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('Root Mean Square Error')
        
        # PLCC plot
        plt.subplot(2, 2, 3)
        plt.plot(self.epochs, self.plcc_values)
        plt.xlabel('Epoch')
        plt.ylabel('PLCC')
        plt.title('Pearson Linear Correlation Coefficient')
        
        # SRCC plot
        plt.subplot(2, 2, 4)
        plt.plot(self.epochs, self.srcc_values)
        plt.xlabel('Epoch')
        plt.ylabel('SRCC')
        plt.title('Spearman Rank Correlation Coefficient')
        
        plt.tight_layout()
        plt.savefig(f'training_plots/metrics_{self.regression_type}.png')
        plt.close() 