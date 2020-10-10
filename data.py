import torch

### Speficy Data Attributes

C = 3 # No of Channels
H = 64 # Height
W = 64 # Width

def get_random_data(n_batches, batch_size, T, N):

    random_data_sample = torch.rand((n_batches, batch_size, C, T + N, H, W))
    random_data_target = torch.stack((torch.zeros(n_batches,batch_size,1+N), torch.ones(n_batches,batch_size,1+N)), dim=2)

    data = [(x,y) for x,y in zip(random_data_sample,random_data_target)]
    
    return data