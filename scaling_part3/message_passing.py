import torch.distributed as dist
import os
import torch

LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_RANK = int(os.environ['RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])

def run():
    tensor = torch.zeros(1)
    
    # Send tensor to GPU device
    device = torch.device("cuda:{}".format(LOCAL_RANK))
    tensor = tensor.to(device)

    if WORLD_RANK == 0:
        for rank_recv in range(1, WORLD_SIZE):
            dist.send(tensor=tensor, dst=rank_recv)
            print('worker_{} sent data to Rank {}\n'.format(0, rank_recv))
    else:
        dist.recv(tensor=tensor, src=0)
        print('worker_{} has received data from rank {}\n'.format(WORLD_RANK, 0))

def init_processes():
     dist.init_process_group(backend="nccl", #"nccl" for using GPUs, "gloo" for using CPUs
                          world_size=WORLD_SIZE, 
                          rank=WORLD_RANK)
     run()

if __name__ == "__main__":
    init_processes()