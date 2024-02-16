from hydra.utils import instantiate


def build_callbacks(cfg):
    return [instantiate(callback) for callback in cfg.callbacks]
