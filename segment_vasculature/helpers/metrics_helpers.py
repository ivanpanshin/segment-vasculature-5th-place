from hydra.utils import instantiate


def build_metric(cfg):
    return instantiate(cfg.metric)
