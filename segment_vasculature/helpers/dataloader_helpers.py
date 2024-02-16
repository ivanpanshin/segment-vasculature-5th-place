import logging

from hydra.utils import instantiate


def build_transforms(cfg):
    train_transforms = instantiate(cfg.transform.train)
    val_transforms = instantiate(cfg.transform.val)

    return {
        "train": train_transforms,
        "val": val_transforms,
    }


def build_loaders(cfg):
    logging.info("Building transforms")
    transforms = build_transforms(cfg)
    # print(f"transforms in loaders: {transforms}")

    logging.info("Building datasets")

    train_datasets = val_datasets = test_datasets = None
    train_loaders = val_loaders = test_loaders = {}
    train_sampler = collator = None

    if hasattr(cfg.dataset, "train"):
        logging.info("Building train dataset")
        train_datasets = {k: instantiate(v, transform=transforms["train"]) for (k, v) in cfg.dataset.train.items()}
        assert len(train_datasets) == 1, "There should be only 1 train dataset"

    if hasattr(cfg.dataset, "val"):
        logging.info("Building val dataset")
        val_datasets = {k: instantiate(v, transform=transforms["val"]) for (k, v) in cfg.dataset.val.items()}
        logging.info(f"Num of val datasets: {len(val_datasets)}")
    if hasattr(cfg.dataset, "test"):
        logging.info("Building test dataset")
        test_datasets = {k: instantiate(v, transform=transforms["val"]) for (k, v) in cfg.dataset.test.items()}
        assert len(test_datasets) == 1, "There should be only 1 test dataset"

    assert train_datasets or val_datasets or test_datasets, "At least train/val/test dataset should be present"

    if hasattr(cfg, "collate_fn"):
        logging.info("Building collator")
        collator = instantiate(cfg.collate_fn)

    if train_datasets:
        logging.info("Building train loader")
        if hasattr(cfg, "sampler"):
            logging.info("Building train sampler")
            cfg.loader.train.shuffle = False
            train_sampler = instantiate(cfg.sampler, dataset=train_datasets["train"])

        train_loaders = {
            k: instantiate(cfg.loader.train, dataset=v, sampler=train_sampler, collate_fn=collator)
            for (k, v) in train_datasets.items()
        }

    if val_datasets:
        if hasattr(cfg, "sampler"):
            logging.info("Building val sampler")
            cfg.loader.val.shuffle = False
            if cfg.trainer.trainer_hyps.ddp_validation:
                val_samplers = {
                    k: instantiate(cfg.sampler, shuffle=False, dataset=v) for (k, v) in val_datasets.items()
                }
            else:
                val_samplers = {k: None for (k, v) in val_datasets.items()}

        val_loaders = {}
        for k in val_datasets.keys():
            if hasattr(cfg, "sampler"):
                val_loaders[k] = instantiate(
                    cfg.loader.val, dataset=val_datasets[k], sampler=val_samplers[k], collate_fn=collator
                )
            else:
                val_loaders[k] = instantiate(cfg.loader.val, dataset=val_datasets[k], collate_fn=collator)

    if test_datasets:
        logging.info("Building test loader")
        test_loaders = {
            k: instantiate(cfg.loader.test, dataset=v, collate_fn=collator) for (k, v) in test_datasets.items()
        }

    return {**train_loaders, **val_loaders, **test_loaders}
