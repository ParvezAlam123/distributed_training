import torch
import torch.distributed as dist
import torch.multiprocessing as mp 
import os 
from torch.utils.data import Dataset, DataLoader

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'  # Change this to the master node's IP address if using multiple machines
    os.environ['MASTER_PORT'] = '12355'  # Pick a free port on the master node
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group() 
    
    
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP 


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(20, 10)
        self.relu = nn.ReLU()
        #self.fc2 = nn.Linear(10, 10)
        #self.fc3 = nn.Linear(10, 10)

    def forward(self, x):
        #x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        #x = self.fc2(x)
        #x = self.relu(x)
        #x = self.fc3(x)
        return x

def create_model():
    return SimpleModel() 


from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
 
 
 
class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.files = [] 
        for i in range(size):
           sample = {}
           sample["data"] = torch.rand(20) 
           sample["target"] = torch.rand(10)
           self.files.append(sample) 
           

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        return {"data":self.files[index]["data"], "target":self.files[index]["target"]}
        
 
def create_dataloader(rank, world_size, batch_size=32):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = MyTrainDataset(32000)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return dataloader  
    
    
    
def train(rank, world_size, epochs=5):
    setup(rank, world_size) 
    print("after setup")
    
    dataloader = create_dataloader(rank, world_size) 
    print("after dataloader")
    model = create_model().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        ddp_model.train()
        print("in training")
        for batch_idx, data, in enumerate(dataloader):
            #print(data.keys())
            data_ = data["data"].to(rank) 
            target = data["target"].to(rank).requires_grad_(True)
            
            output = ddp_model(data_)
            
            loss = ((output - target)**2).sum()
            #print(loss) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if rank == 0:
            print(f"Epoch {epoch} complete")
    
    cleanup() 
    
    
    
def main():
    world_size = 2  # Number of GPUs
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True) 
    



if __name__ == "__main__":
    main() 



