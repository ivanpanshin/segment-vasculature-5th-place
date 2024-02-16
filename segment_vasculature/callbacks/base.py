"""Callback class for main pipeline functionality: all manipulations (with data, models, logging, etc) are
done via Callbacks
"""


class Callback:
    """
    Abstract base class used to build new callbacks.
    Subclass this class and override any of the relevant hooks
    """

    def on_init_start(
        self,
        trainer,
    ) -> None:
        """Run callback at the start of trainer init.
        Args:
            trainer: Trainer of the pipeline.
        Returns:
            None
        """

    def on_init_end(
        self,
        trainer,
    ):
        """Run callback at the end of trainer init.
        Args:
            trainer: Trainer of the pipeline.
        Returns:
            None
        """

    def on_fit_start(
        self,
        trainer,
    ) -> None:
        """Run callback at the start of fit.
        Args:
            trainer: Trainer of the pipeline.
        Returns:
            None
        """

    def on_fit_end(
        self,
        trainer,
    ):
        """Run callback at the end of fit.
        Args:
            trainer: Trainer of the pipeline.
        Returns:
            None
        """

    def on_train_start(
        self,
        trainer,
    ):
        """Run callback at the start of train epoch.
        Args:
            trainer: Trainer of the pipeline.
        Returns:
            None
        """

    def on_train_end(
        self,
        trainer,
    ):
        """Run callback at the end of train epoch.
        Args:
            trainer: Trainer of the pipeline.
        Returns:
            None
        """

    def on_val_start(
        self,
        trainer,
    ):
        """Run callback at the start of validation.
        Args:
            trainer: Trainer of the pipeline.
        Returns:
            None
        """

    def on_val_end(self, trainer):
        """Run callback at the end of validation.
        Args:
            trainer: Trainer of the pipeline.
        Returns:
            None
        """

    def on_logging_by_iter(
        self,
        trainer,
    ):
        """Run callback once every N iterations (defined by config).
        Args:
            trainer: Trainer of the pipeline.
        Returns:
            None
        """

    def on_epoch_start(
        self,
        trainer,
    ):
        """Run callback at the start of each epoch.
        Args:
            trainer: Trainer of the pipeline.
        Returns:
            None
        """

    def on_epoch_end(
        self,
        trainer,
    ):
        """Run callback at the end of each epoch.
        Args:
            trainer: Trainer of the pipeline.
        Returns:
            None
        """

    def on_train_step_start(
        self,
        trainer,
    ):
        """Run callback at the start of each train step (forward -> backward pass).
        Args:
            trainer: Trainer of the pipeline.
        Returns:
            None
        """

    def on_train_step_end(
        self,
        trainer,
    ):
        """Run callback at the end of each train step (forward -> backward pass).
        Args:
            trainer: Trainer of the pipeline.
        Returns:
            None
        """

    def on_val_step_start(
        self,
        trainer,
    ):
        """Run callback at the start of each train step (forward -> backward pass).
        Args:
            trainer: Trainer of the pipeline.
        Returns:
            None
        """

    def on_val_step_end(
        self,
        trainer,
    ):
        """Run callback at the end of each train step (forward -> backward pass).
        Args:
            trainer: Trainer of the pipeline.
        Returns:
            None
        """
