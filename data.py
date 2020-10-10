import torch

### Speficy Data Attributes

C = 3 # No of Channels
H = 64 # Height
W = 64 # Width

def get_random_data(n_batches, batch_size, T, N):
    r"""Returns a random dataset according to 
         - the data attributes specified: C, H, W 
         - no of batches: n_batches
         - batch size: batch_size
         - length of base sequence: T
         - additional frames added to base sequence: N

    Args:
         - no of batches: n_batches
         - batch size: batch_size
         - length of base sequence: T
         - additional frames added to base sequence: N
    """
    random_data_sample = torch.rand((n_batches, batch_size, C, T + N, H, W)) ## Random samples to emulate video sequence
    
    ## Random targets is of shape n_batches*batch_size*2*(1+N). 
    ## Each 2*1 tensor is [0,1]. It represents that the target is True i.e there is a shot boundary there. 
    ## Currently, we assume all the frames have shot boundaries. So all targets are True

    random_data_target = torch.stack((torch.zeros(n_batches,batch_size,1+N), torch.ones(n_batches,batch_size,1+N)), dim=2) 

    data = [(x,y) for x,y in zip(random_data_sample,random_data_target)]
    
    return data