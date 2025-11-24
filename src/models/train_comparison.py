import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np

from .baselines import MODEL_BUILDERS
from .train import WasteDataset

class ModelTrainer:
    def __init__(self, model_name, model_builder, num_classes, device):
        self.model_name = model_name
        self.model = model_builder(num_classes=num_classes, pretrained=True).to(device)
        self.device = device
        self.train_losses = []
        self.val_accuracies = []
        self.train_times = []
        self.best_val_acc = 0.0
        
    def train(self, train_loader, val_loader, num_epochs=10):
        from tqdm import tqdm
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training
            self.model.train()
            running_loss = 0.0
            correct, total = 0, 0
            
            # Progress bar for training
            train_pbar = tqdm(train_loader, 
                            desc=f'{self.model_name} Epoch {epoch+1}/{num_epochs} [Train]',
                            leave=False)
            
            for images, labels in train_pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                current_acc = correct / total
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.4f}'
                })
            
            train_pbar.close()
            
            train_loss = running_loss / total
            train_acc = correct / total
            self.train_losses.append(train_loss)
            
            # Validation
            val_acc = self.evaluate(val_loader)
            self.val_accuracies.append(val_acc)
            
            epoch_time = time.time() - start_time
            self.train_times.append(epoch_time)
            
            print(f"{self.model_name} - Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f} | Time: {epoch_time:.2f}s")
            
            # Save best model for this specific model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model()
                print(f"💾 New best model saved! Val Acc: {val_acc:.4f}")
        
        return self.best_val_acc
    
    def evaluate(self, data_loader):
        from tqdm import tqdm
        
        self.model.eval()
        correct, total = 0, 0
        
        # Progress bar for validation
        val_pbar = tqdm(data_loader, 
                    desc='Validating',
                    leave=False)
        
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                current_acc = correct / total
                val_pbar.set_postfix({
                    'Acc': f'{current_acc:.4f}'
                })
        
        val_pbar.close()
        return correct / total
    
    def save_model(self):
        save_dir = os.path.join("src", "models", "comparison")
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{self.model_name}.pth")
        torch.save(self.model.state_dict(), model_path)
    
    def load_model(self):
        model_path = os.path.join("src", "models", "comparison", f"{self.model_name}.pth")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))


def calculate_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

def save_best_overall_model(best_model_name, best_model_builder, num_classes, device):
    model_path = os.path.join("src", "models", "comparison", f"{best_model_name}.pth")
    
    model = best_model_builder(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    save_dir = os.path.join("src", "models")
    os.makedirs(save_dir, exist_ok=True)
    final_path = os.path.join(save_dir, "best_model.pth")
    torch.save({
        'state_dict': model.state_dict(),
        'model_name': best_model_name,
        'num_classes': num_classes
    }, final_path)
    
    print(f"🏆 Best model '{best_model_name}' saved to: {final_path}")
    
    import shutil
    temp_dir = os.path.join("src", "models", "comparison")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

def run_model_comparison():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 1
    batch_size = 32
    
    # Load data
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    manifest_path = os.path.join(project_root, "data", "unified", "manifest.csv")
    df = pd.read_csv(manifest_path)
    num_classes = len(df['unified_class'].unique())
    
    # Transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = WasteDataset(df, split="train", transform=train_transforms)
    val_dataset = WasteDataset(df, split="val", transform=val_transforms)
    test_dataset = WasteDataset(df, split="test", transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Models to compare
    models_to_compare = [
        'resnet18', 'resnet50', 'mobilenet_v3_large',
        'efficientnet_b0', 'efficientnet_b2', 'convnext_tiny'
    ]
    
    results = {}
    
    for model_name in models_to_compare:
        print(f"\n{'='*50}")
        print(f"Training {model_name}...")
        print(f"{'='*50}")
        
        # Initialize trainer
        trainer = ModelTrainer(
            model_name=model_name,
            model_builder=MODEL_BUILDERS[model_name],
            num_classes=num_classes,
            device=device
        )
        
        # Train model
        start_train_time = time.time()
        best_val_acc = trainer.train(train_loader, val_loader, num_epochs)
        total_train_time = time.time() - start_train_time
        
        # Load best model for evaluation
        trainer.load_model()
        
        # Calculate model size
        model_size = calculate_model_size(trainer.model)
        
        # Test accuracy
        test_acc = trainer.evaluate(test_loader)
        
        # Store results
        results[model_name] = {
            'best_val_accuracy': best_val_acc,
            'test_accuracy': test_acc,
            'total_train_time': total_train_time,
            'avg_epoch_time': np.mean(trainer.train_times),
            'model_size_mb': model_size,
            'train_losses': trainer.train_losses,
            'val_accuracies': trainer.val_accuracies
        }
        
        print(f"{model_name} - Test Accuracy: {test_acc:.4f} | "
              f"Model Size: {model_size:.2f}MB")
            
    return results, num_classes

def plot_comparison_results(results, save_dir):    
    models = list(results.keys())
    test_accuracies = [results[model]['test_accuracy'] for model in models]
    val_accuracies = [results[model]['best_val_accuracy'] for model in models]
    model_sizes = [results[model]['model_size_mb'] for model in models]
    train_times = [results[model]['total_train_time'] for model in models]
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Accuracy comparison - ИСПРАВЛЕННЫЙ
    plt.subplot(2, 2, 1)
    x_pos = np.arange(len(models))
    width = 0.35
    
    plt.bar(x_pos - width/2, val_accuracies, width, label='Validation', alpha=0.7, color='skyblue')
    plt.bar(x_pos + width/2, test_accuracies, width, label='Test', alpha=0.7, color='lightcoral')
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.xticks(x_pos, models, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Model size comparison
    plt.subplot(2, 2, 2)
    bars = plt.bar(models, model_sizes, alpha=0.7, color='lightgreen')
    
    for bar, size in zip(bars, model_sizes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{size:.1f}MB', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Models')
    plt.ylabel('Model Size (MB)')
    plt.title('Model Size Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Training time comparison
    plt.subplot(2, 2, 4)
    bars = plt.bar(models, train_times, alpha=0.7, color='violet')
    
    for bar, time_val in zip(bars, train_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{time_val:.0f}s', ha='center', va='bottom', fontsize=8)
        
    plt.xlabel('Models')
    plt.ylabel('Total Training Time (s)')
    plt.title('Training Time Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "model_comparison.png"), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"📈 Comparison plots saved to: {save_dir}")
    

if __name__ == "__main__":
    results, num_classes = run_model_comparison()
    
    # Print summary
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    smallest_model = min(results.items(), key=lambda x: x[1]['model_size_mb'])
    
    print(f"🏆 Best Accuracy: {best_model[0]} ({best_model[1]['test_accuracy']:.4f})")
    print(f"📦 Smallest Model: {smallest_model[0]} ({smallest_model[1]['model_size_mb']:.2f}MB)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model_name = best_model[0]
    best_model_builder = MODEL_BUILDERS[best_model_name]
    
    save_best_overall_model(best_model_name, best_model_builder, num_classes, device)

    results_dir = os.path.join("reports", "model_comparison")
    os.makedirs(results_dir, exist_ok=True)
    plot_comparison_results(results, results_dir)