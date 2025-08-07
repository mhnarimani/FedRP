# ### Required Libraries ###
# Importing standard libraries and PyTorch-related libraries for building and training deep learning models.

import time
import numpy as np
import torch
import logging
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from abc import ABCMeta, abstractmethod
import math

# ### Initial Setup ###

# Logging configuration to save training results.
# Only one of these lines should be active per run to save results to the corresponding file.
logging.basicConfig(
    filename='resnet18_cifar100_main.log',
    force=True,
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# A class to manage hyperparameters and execution settings.
class Arguments:
    """
    A class to hold arguments and hyperparameters for the experiment.
    This is used for simplicity in notebook environments instead of argparse.
    """
    def __init__(self):
        self.batch_size = 64          # Batch size for training
        self.test_batch_size = 16     # Batch size for testing
        self.epochs = 30              # Number of communication rounds
        self.lr = 0.1                 # Learning rate
        self.client_count = 10        # Number of clients
        self.E = 1                    # Number of local epochs
        self.alpha = 1.0              # Penalty parameter for the ADMM algorithm
        self.rp_dim = 10              # Dimension of the random projection space (for FedRP)

args = Arguments()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ### Data Preparation ###

def get_datasets():
    """
    Downloads and prepares the CIFAR-100 dataset with necessary transforms.
    Transforms include normalization and data augmentation like random horizontal flipping.
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=train_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=test_transform, download=True)
    return train_dataset, test_dataset

class FederatedDataset(Dataset):
    """
    A class to partition a standard dataset among multiple clients.
    This class ensures that data is distributed in a balanced (IID) manner,
    where each client receives a portion of data from all classes.
    """
    def __init__(self, dataset: Dataset, num_clients: int, client_id: int):
        super(FederatedDataset, self).__init__()
        self.dataset = dataset
        self.client_id = client_id
        self.num_clients = num_clients
        # Distribute data balancedly among clients
        client_data_indices = self._distribute_data_balanced()
        self.map = client_data_indices[client_id]
        np.random.shuffle(self.map)
        self.len = len(self.map)

    def _distribute_data_balanced(self):
        """
        Distributes data indices balancedly based on class labels among clients.
        """
        num_samples = len(self.dataset)
        targets = np.array([self.dataset[i][1] for i in range(num_samples)])
        classes, _ = np.unique(targets, return_counts=True)
        class_indices = {cls: np.where(targets == cls)[0] for cls in classes}
        
        client_data_indices = [[] for _ in range(self.num_clients)]
        for indices in class_indices.values():
            np.random.shuffle(indices)
            splits = np.array_split(indices, self.num_clients)
            for cid, split in enumerate(splits):
                client_data_indices[cid].extend(split)
        return client_data_indices

    def __getitem__(self, index):
        return self.dataset[self.map[index]]

    def __len__(self):
        return self.len

# ### Model Definition ###

class ResNet18(nn.Module):
    """
    ResNet-18 model optimized for the CIFAR-100 dataset.
    Modifications include:
    1. Using a ResNet-18 model without pre-trained weights.
    2. Changing the first convolutional layer to accept 32x32 images.
    3. Removing the MaxPool layer to preserve spatial feature dimensions.
    4. Adjusting the final fully connected layer for 100 output classes.
    """
    def __init__(self, num_classes=100):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# ### Helper Functions ###

def evaluate_model(device, model, dataloader, criterion):
    """
    Evaluates the model's performance on the test dataset.
    Calculates and returns the model's loss and accuracy.
    """
    model.eval()
    loss, total, correct = 0, 0, 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    avg_loss = loss / len(dataloader.dataset)
    return avg_loss, accuracy

# ### Base Classes for Federated Learning ###

class FederatedLearning(metaclass=ABCMeta):
    """
    Abstract Base Class for Federated Learning algorithms.
    Defines the main methods that every FL algorithm should implement.
    """
    @abstractmethod
    def __init__(self, Model, device, client_count, optimizer, criterion):
        pass
    
    @abstractmethod
    def _client_update(self, client_id, lr, E):
        pass

    @abstractmethod
    def _server_aggregate(self):
        pass

    @abstractmethod
    def global_update(self, state, lr, E):
        pass

class FLBase(FederatedLearning):
    """
    A base class for different Federated Learning implementations.
    Contains shared logic like sending the model to clients and setting up federated data.
    """
    def __init__(self, Model, device, client_count, optimizer, criterion):
        self.Model = Model
        self.device = device
        self.client_count = client_count
        # Create a model instance for each client
        self.models = [Model().to(self.device) for _ in range(self.client_count)]
        self.optimizer = optimizer
        self.criterion = criterion

    def _send_model(self, state):
        """
        Sends the global model state (parameters) to all clients.
        """
        for model in self.models:
            model.load_state_dict(state.copy())
        # Reset weights and metrics for each round
        self.weights = [0] * self.client_count
        self.losses = [0] * self.client_count
        self.accuracies = [0] * self.client_count

    def setup_federated_data(self, dataset, batch_size):
        """
        Creates a federated dataloader for each client.
        """
        self.client_dataloaders = [
            DataLoader(FederatedDataset(dataset, self.client_count, cid), batch_size=batch_size, shuffle=True)
            for cid in range(self.client_count)
        ]

# ### FedAvg Algorithm Implementation ###
# This algorithm is used as a baseline for comparison.

class FedAvg(FLBase):
    """
    Implementation of the Federated Averaging (FedAvg) algorithm.
    """
    def __init__(self, Model, device, client_count, optimizer, criterion):
        super(FedAvg, self).__init__(Model, device, client_count, optimizer, criterion)
        
    def _client_update(self, client_id, lr, E):
        """
        Trains the model on a client's local data.
        """
        model = self.models[client_id]
        optimizer = self.optimizer(model.parameters(), lr=lr)
        criterion = self.criterion()
        dataloader = self.client_dataloaders[client_id]
        
        weight, losses, total, correct = 0, 0, 0, 0
        for _ in range(E):
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                losses += loss.item()
                weight += len(data)
                optimizer.step()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
        self.weights[client_id] = weight / E
        self.losses[client_id] = losses / (E * weight) if weight > 0 else 0
        self.accuracies[client_id] = 100 * correct / total if total > 0 else 0

    def _server_aggregate(self):
        """
        Aggregates client model parameters on the server using a weighted average.
        """
        total_weight = sum(self.weights)
        self.normalized_weights = np.array(self.weights) / total_weight if total_weight > 0 else [1/self.client_count]*self.client_count
        
        with torch.no_grad():
            clients_updates = [model.state_dict() for model in self.models]
            aggregated_params = clients_updates[0].copy()
            
            for name in aggregated_params:
                aggregated_params[name] = torch.zeros_like(aggregated_params[name]).to(self.device)
            
            for client_id, params in enumerate(clients_updates):
                for name in aggregated_params:
                    aggregated_params[name] += params[name] * self.normalized_weights[client_id]
                    
        return aggregated_params.copy()

    def global_update(self, state, lr, E=1, epoch=None):
        """
        Executes one full round of federated learning (send model, local training, aggregate).
        """
        self._send_model(state)
        for i in range(self.client_count):
            self._client_update(i, lr, E)
        
        avg_loss = sum(self.losses) / self.client_count
        avg_acc = sum(self.accuracies) / self.client_count
        return self._server_aggregate(), avg_loss, avg_acc

# ### FedAvg+DP Algorithm Implementation ###
# This algorithm adds Gaussian noise to the model parameters to provide differential privacy.

def clip_weights(weights, C):
    """
    Clips gradients to limit the influence of a single data point.
    """
    norm = torch.norm(weights, p=2)
    scale = max(1.0, norm / C)
    return weights / scale

def add_dp_noise(weights, C, sigma):
    """
    Adds Gaussian noise to achieve differential privacy.
    """
    noise = torch.normal(0, sigma * C, size=weights.size(), device=weights.device)
    return weights + noise

class FedAvgDP(FedAvg):
    """
    Implements FedAvg with Differential Privacy (DP).
    After local training, parameters are clipped and noise is added.
    """
    def __init__(self, Model, device, client_count, optimizer, criterion):
        super(FedAvgDP, self).__init__(Model, device, client_count, optimizer, criterion)
        self.C = 1.0       # Clipping threshold
        self.sigma = 1e-4  # Noise level

    def _client_update(self, client_id, lr, E):
        # First, perform standard local training
        super()._client_update(client_id, lr, E)
        
        # Apply DP mechanisms (clipping and noise addition)
        model = self.models[client_id]
        for name, param in model.named_parameters():
            if 'weight' in name:
                param.data = clip_weights(param.data, self.C)
                param.data = add_dp_noise(param.data, self.C, self.sigma)

# ### ADMM (FedADMM) Algorithm Implementation ###
# This algorithm uses the ADMM method to achieve consensus among clients.

class FedADMM(FLBase):
    """
    Implementation of Federated Learning using the ADMM optimization framework.
    """
    def __init__(self, Model, device, client_count, optimizer, criterion, alpha):
        super(FedADMM, self).__init__(Model, device, client_count, optimizer, criterion)
        self.alpha = alpha  # Penalty parameter
        
        with torch.no_grad():
            self.server_model = Model().to(self.device)  # z variable or server model
        
        params = list(self.Model().parameters())
        # Dual variable y
        self.y = [[torch.zeros_like(p).to(self.device) for p in params] for _ in range(self.client_count)]
        self.epoch_tracker = [0] * self.client_count

    def _update_y(self, client_id):
        """
        Updates the dual variable y for each client.
        """
        if self.epoch_tracker[client_id] != 0:
            with torch.no_grad():
                client_params = list(self.models[client_id].parameters())
                server_params = list(self.server_model.parameters())
                for i in range(len(self.y[client_id])):
                    self.y[client_id][i] += self.alpha * (client_params[i] - server_params[i])
        else:
            self.epoch_tracker[client_id] = 1

    def _client_update(self, client_id, lr, E):
        """
        Performs local client training with the modified ADMM cost function.
        """
        self._update_y(client_id)
        
        model = self.models[client_id]
        optimizer = self.optimizer(model.parameters(), lr=lr)
        criterion = self.criterion()
        dataloader = self.client_dataloaders[client_id]
        
        weight, losses, total, correct = 0, 0, 0, 0
        for _ in range(E):
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                
                # Calculate the main loss
                loss = criterion(output, target)
                losses += loss.item()
                
                # Add ADMM penalty and dual terms to the loss
                # This corresponds to Equation (9) in the paper, but without projection yet.
                client_params_flat = torch.cat([p.flatten() for p in model.parameters()])
                server_params_flat = torch.cat([p.flatten() for p in self.server_model.parameters()]).detach()
                y_flat = torch.cat([y_i.flatten() for y_i in self.y[client_id]]).detach()
                
                loss += (self.alpha / 2) * torch.norm(client_params_flat - server_params_flat) ** 2
                loss += torch.dot(y_flat, client_params_flat - server_params_flat)
                
                loss.backward()
                optimizer.step()
                weight += len(data)

                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        self.weights[client_id] = weight / E
        self.losses[client_id] = losses / (E * weight) if weight > 0 else 0
        self.accuracies[client_id] = 100 * correct / total if total > 0 else 0

    def _server_aggregate(self):
        """
        Aggregates client parameters to update the global model (z).
        """
        total_weight = sum(self.weights)
        normalized_weights = np.array(self.weights) / total_weight if total_weight > 0 else [1/self.client_count]*self.client_count
        
        with torch.no_grad():
            clients_updates = [model.state_dict() for model in self.models]
            aggregated_params = clients_updates[0].copy()
            
            for name in aggregated_params:
                aggregated_params[name] = torch.zeros_like(aggregated_params[name]).to(self.device)
            
            for cid, params in enumerate(clients_updates):
                for name in aggregated_params:
                    aggregated_params[name] += params[name] * normalized_weights[cid]
                    
        return aggregated_params.copy()

    def global_update(self, state, lr, E=1, epoch=None):
        """
        Executes one full round of FedADMM.
        """
        self.server_model.load_state_dict(state.copy())
        # Reset metrics for the round
        self.weights = [0] * self.client_count
        self.losses = [0] * self.client_count
        self.accuracies = [0] * self.client_count
        
        for i in range(self.client_count):
            self._client_update(i, lr, E)
            
        avg_loss = sum(self.losses) / self.client_count
        avg_acc = sum(self.accuracies) / self.client_count
        aggregated_state = self._server_aggregate()
        self.server_model.load_state_dict(aggregated_state)
        
        return aggregated_state, avg_loss, avg_acc
        
# ### FedRP Algorithm Implementation (Main method from the paper) ###
# This algorithm combines ADMM with random projection to simultaneously improve privacy
# and reduce communication cost.

class FedRP(FLBase):
    """
    Implementation of the FedRP algorithm combining ADMM and random projection.
    """
    def __init__(self, Model, device, client_count, optimizer, criterion, alpha, rp_dim):
        super(FedRP, self).__init__(Model, device, client_count, optimizer, criterion)
        self.alpha = alpha
        self.rp_dim = rp_dim
        self.num_params = sum(p.numel() for p in Model().parameters())
        
        # ADMM variables in the projected space
        self.server_z = torch.zeros(self.rp_dim).to(self.device)
        self.y = [torch.zeros(self.rp_dim).to(self.device) for _ in range(self.client_count)]
        self.epoch_tracker = [0] * self.client_count

    def _generate_projection_matrix(self, seed):
        """
        Creates a shared random projection matrix for all clients in a round.
        This matrix is unknown to the server.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        # Using a variance of 1/n for computational stability
        return (torch.randn(self.rp_dim, self.num_params) / math.sqrt(self.num_params)).to(self.device).detach()

    def _update_y(self, client_id, proj_matrix):
        """
        Updates the dual variable y in the projected space.
        """
        if self.epoch_tracker[client_id] != 0:
            with torch.no_grad():
                client_params_flat = torch.cat([p.flatten() for p in self.models[client_id].parameters()])
                client_z = torch.matmul(proj_matrix, client_params_flat)
                self.y[client_id] += self.alpha * (client_z - self.server_z)
        else:
            self.epoch_tracker[client_id] = 1

    def _client_update(self, client_id, lr, E, proj_matrix):
        """
        Performs local client training with the modified FedRP cost function.
        The loss is calculated based on projected parameters.
        """
        self._update_y(client_id, proj_matrix)
        
        model = self.models[client_id]
        optimizer = self.optimizer(model.parameters(), lr=lr)
        criterion = self.criterion()
        dataloader = self.client_dataloaders[client_id]
        
        weight, losses, total, correct = 0, 0, 0, 0
        for _ in range(E):
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                
                loss = criterion(output, target)
                losses += loss.item()
                
                # Add ADMM penalty and dual terms in the projected space (Equation 9 from the paper)
                client_params_flat = torch.cat([p.flatten() for p in model.parameters()])
                client_z = torch.matmul(proj_matrix, client_params_flat)
                
                loss += (self.alpha / 2) * torch.norm(client_z - self.server_z.detach()) ** 2
                loss += torch.dot(self.y[client_id].detach(), client_z - self.server_z.detach())
                
                loss.backward()
                optimizer.step()
                weight += len(data)

                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        # Store projected parameters for aggregation
        client_params_flat = torch.cat([p.flatten() for p in self.models[client_id].parameters()])
        self.projected_params[client_id] = torch.matmul(proj_matrix, client_params_flat)
        self.weights[client_id] = weight / E
        self.losses[client_id] = losses / (E * weight) if weight > 0 else 0
        self.accuracies[client_id] = 100 * correct / total if total > 0 else 0

    def _server_aggregate(self):
        """
        Aggregates the projected vectors z_i to update the global z_bar.
        """
        total_weight = sum(self.weights)
        normalized_weights = np.array(self.weights) / total_weight if total_weight > 0 else [1/self.client_count]*self.client_count
        
        with torch.no_grad():
            aggregated_z = torch.zeros(self.rp_dim).to(self.device)
            for cid in range(self.client_count):
                aggregated_z += self.projected_params[cid] * normalized_weights[cid]
        return aggregated_z.clone()

    def global_update(self, state, lr, E, epoch):
        """
        Executes one full round of the FedRP algorithm.
        """
        self._send_model(state) # Send the model w to clients
        # Reset metrics for the round
        self.weights = [0] * self.client_count
        self.losses = [0] * self.client_count
        self.accuracies = [0] * self.client_count
        self.projected_params = [None] * self.client_count
        
        # Generate a new projection matrix for each round
        proj_matrix = self._generate_projection_matrix(seed=42 + epoch)
        
        for i in range(self.client_count):
            self._client_update(i, lr, E, proj_matrix)
        
        # Aggregate and update z_bar
        self.server_z = self._server_aggregate()
        
        # Since the server does not have access to the original parameters (w),
        # for evaluation, we consider one client's model (e.g., the last one) as the global model.
        # In practice, the average of models could also be used.
        final_state = self.models[-1].state_dict().copy()
        avg_loss = sum(self.losses) / self.client_count
        avg_acc = sum(self.accuracies) / self.client_count
        
        return final_state, avg_loss, avg_acc

# ### Main Experiment Runner Function ###

def run_experiment(algorithm_class, train_dataset, test_dataset, args, **kwargs):
    """
    Runs a complete experiment for a specified Federated Learning algorithm.
    """
    print(f"\n--- Running Experiment: {algorithm_class.__name__} with {args.client_count} clients ---")
    
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    
    # Instantiate the federated algorithm
    fl_instance = algorithm_class(ResNet18, device, args.client_count, optim.SGD, nn.CrossEntropyLoss, **kwargs)
    fl_instance.setup_federated_data(train_dataset, args.batch_size)
    
    # Initial global model
    global_model = ResNet18().to(device)
    global_state = global_model.state_dict().copy()
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    for epoch in range(args.epochs):
        # Perform one global update round
        global_state, train_loss, train_acc = fl_instance.global_update(global_state, lr=args.lr, E=args.E, epoch=epoch)
        
        # Evaluate the global model
        global_model.load_state_dict(global_state.copy())
        test_loss, test_acc = evaluate_model(device, global_model, test_loader, criterion)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        logging.info(f"Epoch {epoch+1}, client_count {args.client_count}, Algo: {algorithm_class.__name__}: train_accuracy={train_acc:.4f}, val_accuracy={test_acc:.4f}, train_loss={train_loss:.4f}, val_loss={test_loss:.4f}")

    print(f"Training time: {time.time() - start_time:.2f} seconds")

# ### Program Entry Point ###

if __name__ == '__main__':
    train_data, test_data = get_datasets()
    
    # Run experiments for the different algorithms
    run_experiment(FedAvg, train_data, test_data, args)
    run_experiment(FedAvgDP, train_data, test_data, args)
    run_experiment(FedADMM, train_data, test_data, args, alpha=args.alpha)
    run_experiment(FedRP, train_data, test_data, args, alpha=args.alpha, rp_dim=args.rp_dim)
