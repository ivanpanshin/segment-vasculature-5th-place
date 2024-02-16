from torch.utils.data.distributed import DistributedSampler


def distributed_sampler(dataset, shuffle=True):
    return DistributedSampler(dataset=dataset, shuffle=shuffle)
