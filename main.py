import torch

if __name__ == '__main__':
    a = torch.tensor([2, 3]).fill_(0)
    print(a)