from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time
import matplotlib.pyplot as plt
import numpy as np


class SVDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank_ratio=1.0):
        """
        SVD-compressed linear layer
        
        Args:
            in_features: input dimension
            out_features: output dimension
            rank_ratio: ratio of singular values to keep (1.0 = no compression, 0.1 = keep 10% of values)
        """
        super(SVDLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize a standard linear layer
        linear = nn.Linear(in_features, out_features)
        
        # Get the weight matrix and perform SVD
        weight = linear.weight.data.clone()
        U, S, V = torch.svd(weight)
        
        # Calculate rank
        full_rank = min(in_features, out_features)
        reduced_rank = max(1, int(full_rank * rank_ratio))
        
        # Store compression info
        self.rank_ratio = rank_ratio
        self.compressed_rank = reduced_rank
        self.full_rank = full_rank
        
        # Trim matrices according to rank
        U_reduced = U[:, :reduced_rank]
        S_reduced = S[:reduced_rank]
        V_reduced = V[:, :reduced_rank]
        
        # Create parameter matrices
        self.U = nn.Parameter(U_reduced)
        self.S = nn.Parameter(S_reduced)
        self.V = nn.Parameter(V_reduced)
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(out_features) if linear.bias is None else linear.bias.data.clone())

    def forward(self, x):
        # Option 1: Reconstruct weight matrix and use it (slower but clearer)
        # weight = self.U @ torch.diag(self.S) @ self.V.t()
        # return F.linear(x, weight, self.bias)
        
        # Option 2: Compute matrix multiplications directly (faster)
        # x @ V @ diag(S) @ U.t() + bias
        output = x @ self.V @ torch.diag(self.S) @ self.U.t() + self.bias
        return output
    
    def get_compression_ratio(self):
        """Calculate compression ratio of parameters"""
        original_params = self.in_features * self.out_features + self.out_features
        compressed_params = self.compressed_rank * (self.in_features + self.out_features + 1)
        return compressed_params / original_params


class Net(nn.Module):
    def __init__(self, rank_ratio=1.0):
        super(Net, self).__init__()
        self.fc1 = SVDLinear(28 * 28, 512, rank_ratio)
        self.fc2 = SVDLinear(512, 128, rank_ratio)
        self.fc3 = SVDLinear(128, 10, rank_ratio)
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


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


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

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    
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


def plot_results(compression_ratios, accuracies, runtimes):
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
    plt.savefig('svd_compression_results.png')
    plt.show()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example with SVD Compression')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # Define compression ratios to test
    rank_ratios = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05]
    compression_ratios = []
    accuracies = []
    runtimes = []
    
    for rank_ratio in rank_ratios:
        print(f"\nTraining with rank ratio: {rank_ratio}")
        model = Net(rank_ratio=rank_ratio).to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            scheduler.step()
        
        # Evaluate accuracy
        accuracy = test(model, device, test_loader)
        
        # Measure runtime
        runtime = measure_runtime(model, device, test_loader)
        
        # Get actual compression ratio
        compression_ratio = model.get_compression_ratio()
        
        print(f"Rank ratio: {rank_ratio}, Compression ratio: {compression_ratio:.4f}")
        print(f"Accuracy: {accuracy:.2f}%, Runtime: {runtime:.4f}s")
        
        compression_ratios.append(compression_ratio)
        accuracies.append(accuracy)
        runtimes.append(runtime)
        
        if args.save_model:
            torch.save(model.state_dict(), f"mnist_svd_{rank_ratio}.pt")
    
    # Plot results
    plot_results(compression_ratios, accuracies, runtimes)
    
    # Print summary
    print("\nSummary of results:")
    for i, rank_ratio in enumerate(rank_ratios):
        print(f"Rank ratio: {rank_ratio}, Compression ratio: {compression_ratios[i]:.4f}, "
              f"Accuracy: {accuracies[i]:.2f}%, Runtime: {runtimes[i]:.4f}s")


if __name__ == '__main__':
    main() 