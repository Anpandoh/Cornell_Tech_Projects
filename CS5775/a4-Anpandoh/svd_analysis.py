from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms


class SVDLinear(nn.Module):
    def __init__(self, in_features, out_features, weight_matrix=None, rank_ratio=1.0):
        """
        SVD-compressed linear layer
        
        Args:
            in_features: input dimension
            out_features: output dimension
            weight_matrix: pre-trained weight matrix to compress
            rank_ratio: ratio of singular values to keep (1.0 = no compression, 0.1 = keep 10% of values)
        """
        super(SVDLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Store compression info
        self.rank_ratio = rank_ratio
        self.full_rank = min(in_features, out_features)
        self.compressed_rank = max(1, int(self.full_rank * rank_ratio))
        
        if weight_matrix is not None:
            # Perform SVD on provided weight matrix
            U, S, V = torch.svd(weight_matrix)
            
            # Trim matrices according to rank
            U_reduced = U[:, :self.compressed_rank]
            S_reduced = S[:self.compressed_rank]
            V_reduced = V[:, :self.compressed_rank]
            
            # Create parameter matrices
            self.U = nn.Parameter(U_reduced)
            self.S = nn.Parameter(S_reduced)
            self.V = nn.Parameter(V_reduced)
            
            # Store original matrix for reference
            self.register_buffer('original_weight', weight_matrix)
        else:
            # Initialize randomly if no weight matrix is provided
            self.U = nn.Parameter(torch.randn(out_features, self.compressed_rank))
            self.S = nn.Parameter(torch.ones(self.compressed_rank))
            self.V = nn.Parameter(torch.randn(in_features, self.compressed_rank))
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # Matrix multiplication: x @ V @ diag(S) @ U.t() + bias
        output = x @ self.V @ torch.diag(self.S) @ self.U.t() + self.bias
        return output
    
    def get_compression_ratio(self):
        """Calculate compression ratio of parameters"""
        original_params = self.in_features * self.out_features + self.out_features
        compressed_params = self.compressed_rank * (self.in_features + self.out_features + 1)
        return compressed_params / original_params
    
    def get_reconstruction_error(self):
        """Calculate the Frobenius norm of the difference between original and reconstructed weights"""
        if hasattr(self, 'original_weight'):
            reconstructed = self.U @ torch.diag(self.S) @ self.V.t()
            error = torch.norm(self.original_weight - reconstructed, p='fro')
            relative_error = error / torch.norm(self.original_weight, p='fro')
            return relative_error.item()
        return None


class OriginalNet(nn.Module):
    def __init__(self):
        super(OriginalNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        bs = x.size(0)
        x = x.view(bs, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output


class CompressedNet(nn.Module):
    def __init__(self, original_model, rank_ratio=1.0):
        super(CompressedNet, self).__init__()
        
        # Create SVD layers from original model weights
        self.fc1 = SVDLinear(28 * 28, 512, original_model.fc1.weight.data, rank_ratio)
        self.fc1.bias.data = original_model.fc1.bias.data.clone()
        
        self.fc2 = SVDLinear(512, 128, original_model.fc2.weight.data, rank_ratio)
        self.fc2.bias.data = original_model.fc2.bias.data.clone()
        
        self.fc3 = SVDLinear(128, 10, original_model.fc3.weight.data, rank_ratio)
        self.fc3.bias.data = original_model.fc3.bias.data.clone()
        
        self.rank_ratio = rank_ratio

    def forward(self, x):
        bs = x.size(0)
        x = x.view(bs, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output
    
    def get_compression_ratio(self):
        fc1_ratio = self.fc1.get_compression_ratio()
        fc2_ratio = self.fc2.get_compression_ratio()
        fc3_ratio = self.fc3.get_compression_ratio()
        # Average compression ratio across layers
        return (fc1_ratio + fc2_ratio + fc3_ratio) / 3
    
    def get_reconstruction_errors(self):
        fc1_error = self.fc1.get_reconstruction_error()
        fc2_error = self.fc2.get_reconstruction_error()
        fc3_error = self.fc3.get_reconstruction_error()
        return {
            'fc1': fc1_error,
            'fc2': fc2_error,
            'fc3': fc3_error,
            'avg': (fc1_error + fc2_error + fc3_error) / 3
        }


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    
    return accuracy


def measure_runtime(model, device, test_loader, num_runs=10):
    model.eval()
    # Warm-up run
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            _ = model(data)
            break
    
    # Measure runtime
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            for data, _ in test_loader:
                data = data.to(device)
                _ = model(data)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time


def plot_results(compression_ratios, accuracies, runtimes, reconstruction_errors=None):
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Compression ratio vs. Accuracy
    plt.subplot(1, 3, 1)
    plt.plot(compression_ratios, accuracies, 'o-')
    plt.xlabel('Compression Ratio')
    plt.ylabel('Accuracy (%)')
    plt.title('Compression Ratio vs. Accuracy')
    plt.grid(True)
    
    # Plot 2: Compression ratio vs. Runtime
    plt.subplot(1, 3, 2)
    plt.plot(compression_ratios, runtimes, 'o-')
    plt.xlabel('Compression Ratio')
    plt.ylabel('Runtime (s)')
    plt.title('Compression Ratio vs. Runtime')
    plt.grid(True)
    
    # Plot 3: Runtime vs. Accuracy
    plt.subplot(1, 3, 3)
    plt.plot(runtimes, accuracies, 'o-')
    plt.xlabel('Runtime (s)')
    plt.ylabel('Accuracy (%)')
    plt.title('Runtime vs. Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('svd_analysis_results.png')
    
    # If reconstruction errors are provided, plot them as well
    if reconstruction_errors is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(compression_ratios, reconstruction_errors, 'o-')
        plt.xlabel('Compression Ratio')
        plt.ylabel('Reconstruction Error')
        plt.title('Compression Ratio vs. Reconstruction Error')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('svd_reconstruction_errors.png')
    
    plt.show()


def main():
    # Arguments
    parser = argparse.ArgumentParser(description='Analyze SVD Compression on Pre-trained MNIST Model')
    parser.add_argument('--model-path', type=str, default='mnist_fc.pt',
                        help='path to pre-trained model')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Load test data
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
        test_kwargs.update(cuda_kwargs)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    
    # Load the original model
    original_model = OriginalNet().to(device)
    try:
        original_model.load_state_dict(torch.load(args.model_path))
        print(f"Loaded pre-trained model from {args.model_path}")
    except FileNotFoundError:
        print(f"Model file {args.model_path} not found. Training a new model.")
        # Code to train a new model could be added here
        return
    
    # Verify original model accuracy
    print("Evaluating original model:")
    original_accuracy = test(original_model, device, test_loader)
    original_runtime = measure_runtime(original_model, device, test_loader)
    print(f"Original model runtime: {original_runtime:.4f}s")
    
    # Test different compression levels
    rank_ratios = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]
    compression_ratios = []
    accuracies = []
    runtimes = []
    reconstruction_errors = []
    
    for rank_ratio in rank_ratios:
        print(f"\nAnalyzing with rank ratio: {rank_ratio}")
        compressed_model = CompressedNet(original_model, rank_ratio).to(device)
        
        # Get actual compression ratio
        compression_ratio = compressed_model.get_compression_ratio()
        
        # Get reconstruction errors
        errors = compressed_model.get_reconstruction_errors()
        reconstruction_error = errors['avg']
        
        # Evaluate accuracy
        accuracy = test(compressed_model, device, test_loader)
        
        # Measure runtime
        runtime = measure_runtime(compressed_model, device, test_loader)
        
        print(f"Rank ratio: {rank_ratio}, Compression ratio: {compression_ratio:.4f}")
        print(f"Reconstruction error: {reconstruction_error:.4f}")
        print(f"Accuracy: {accuracy:.2f}%, Runtime: {runtime:.4f}s")
        
        compression_ratios.append(compression_ratio)
        accuracies.append(accuracy)
        runtimes.append(runtime)
        reconstruction_errors.append(reconstruction_error)
    
    # Plot results
    plot_results(compression_ratios, accuracies, runtimes, reconstruction_errors)
    
    # Print summary
    print("\nSummary of results:")
    print(f"Original model - Accuracy: {original_accuracy:.2f}%, Runtime: {original_runtime:.4f}s")
    for i, rank_ratio in enumerate(rank_ratios):
        print(f"Rank ratio: {rank_ratio}, Compression ratio: {compression_ratios[i]:.4f}, "
              f"Accuracy: {accuracies[i]:.2f}%, Runtime: {runtimes[i]:.4f}s, "
              f"Reconstruction error: {reconstruction_errors[i]:.4f}")


if __name__ == '__main__':
    main() 