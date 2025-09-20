import sys,datetime
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import pandas as pd
import torch

"""
@author : lyhue1991, zhangyu
@description : keras core code
"""
class StepRunner:
    def __init__(self, net, loss_fn, accelerator=None, stage="train", metrics_dict=None,
                 optimizer=None, lr_scheduler=None, **kwargs):
        """
        Initialize the StepRunner object

        Args:
            net: Neural network model
            loss_fn: Loss function
            accelerator: Hardware accelerator (e.g., GPU，CPU)
            stage: Training or evaluation stage
            metrics_dict: Dictionary of metrics functions
            optimizer: Optimizer for training
            lr_scheduler: Learning rate scheduler
            **kwargs: Additional keyword arguments
        """
        self.net, self.loss_fn, self.metrics_dict, self.stage = net, loss_fn, metrics_dict, stage
        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
        self.kwargs = kwargs
        self.accelerator = accelerator

        # Set the network to training mode during the training stage, and evaluation mode otherwise
        if self.stage == 'train':
            self.net.train()
        else:
            self.net.eval()

    def __call__(self, batch):
        """
        Perform a training or evaluation step.

        Args:
            batch: Input data batch.

        Returns:
            Tuple of dictionaries containing step losses and step metrics.
        """
        features, labels = batch

        # Compute loss
        with self.accelerator.autocast():
            preds = self.net(features)
            loss = self.loss_fn(preds, labels)

        # Backward pass and optimization (only during training)
        if self.stage == "train" and self.optimizer is not None:
            self.accelerator.backward(loss)

            # Clip gradients if synchronization is enabled
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.net.parameters(), 1.0)

            self.optimizer.step()

            # Adjust learning rate if a scheduler is provided
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Zero gradients for the next iteration
            self.optimizer.zero_grad()

        # Gather loss, predictions, and labels using the accelerator
        all_loss = self.accelerator.gather(loss).sum()
        all_preds = self.accelerator.gather(preds)
        all_labels = self.accelerator.gather(labels)

        # Compute and gather additional metrics
        step_losses = {self.stage + "_loss": all_loss.item()}
        step_metrics = {self.stage + "_" + name: metric_fn(all_preds, all_labels).item()
                        for name, metric_fn in self.metrics_dict.items()}

        # Include learning rate in metrics if available
        if self.stage=="train":
            if self.optimizer is not None:
                step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            else:
                step_metrics['lr'] = 0.0
                
        return step_losses, step_metrics


class EpochRunner:
    def __init__(self, step_runner, quiet=False):
        """
        Initialize the EpochRunner object

        Args:
            step_runner: StepRunner object for handling individual training or evaluation steps
            quiet: Flag to control whether to display progress bar and logs
        """
        self.step_runner = step_runner
        self.stage = step_runner.stage
        self.accelerator = step_runner.accelerator
        self.net = step_runner.net
        self.quiet = quiet

    def __call__(self, dataloader):
        """
        Perform a complete epoch of training or evaluation

        Args:
            dataloader: DataLoader providing batches of data for the epoch

        Returns:
            Dictionary containing aggregated epoch losses and metrics
        """
        # Determine the size of the dataset
        n = dataloader.size if hasattr(dataloader, 'size') else len(dataloader)

        # Initialize tqdm progress bar
        loop = tqdm(enumerate(dataloader, start=1),
                    total=n,
                    file=sys.stdout,
                    disable=not self.accelerator.is_local_main_process or self.quiet,
                    ncols=100
                    )
        epoch_losses = {}

        for step, batch in loop:
            # Perform a step with the provided StepRunner
            with self.accelerator.accumulate(self.net):
                step_losses, step_metrics = self.step_runner(batch)
                step_log = dict(step_losses, **step_metrics)

                # Accumulate step losses for computing epoch losses
                for k, v in step_losses.items():
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v

                # Update progress bar during the epoch
                if step < n:
                    loop.set_postfix(**step_log)

                    if hasattr(self, 'progress') and self.accelerator.is_local_main_process:
                        post_log = dict(**{'i': step, 'n': n}, **step_log)
                        self.progress.set_postfix(**post_log)

                # Compute and display epoch-level metrics at the end of the epoch
                elif step == n:
                    epoch_metrics = step_metrics
                    epoch_metrics.update({self.stage + "_" + name: metric_fn.compute().item()
                                          for name, metric_fn in self.step_runner.metrics_dict.items()})
                    epoch_losses = {k: v / step for k, v in epoch_losses.items()}
                    epoch_log = dict(epoch_losses, **epoch_metrics)
                    loop.set_postfix(**epoch_log)

                    # Update progress bar if available
                    if hasattr(self, 'progress') and self.accelerator.is_local_main_process:
                        post_log = dict(**{'i': step, 'n': n}, **epoch_log)
                        self.progress.set_postfix(**post_log)

                    # Reset stateful metrics for the next epoch
                    for name, metric_fn in self.step_runner.metrics_dict.items():
                        metric_fn.reset()
                else:
                    break

        return epoch_log

class KerasModel(torch.nn.Module):
    StepRunner, EpochRunner = StepRunner, EpochRunner

    def __init__(self, net, loss_fn, metrics_dict=None, optimizer=None, lr_scheduler=None, **kwargs):
        """
        Initialize the KerasModel.

        Args:
            net: Neural network model
            loss_fn: Loss function
            metrics_dict: Dictionary of metrics functions
            optimizer: Optimizer for training
            lr_scheduler: Learning rate scheduler
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.net, self.loss_fn, self.metrics_dict = net, loss_fn, torch.nn.ModuleDict(metrics_dict)
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(
            self.net.parameters(), lr=3e-4)
        self.lr_scheduler = lr_scheduler
        self.kwargs = kwargs
        self.from_scratch = True

    def save_ckpt(self, ckpt_path=None, accelerator=None):
        """
        Save the model checkpoint

        Args:
            ckpt_path: Path to save the checkpoint
            accelerator: Hardware accelerator (e.g., GPU)
        """
        accelerator = accelerator if accelerator is not None else self.accelerator
        net_dict = accelerator.get_state_dict(self.net)
        accelerator.save(net_dict, ckpt_path if ckpt_path is not None else self.ckpt_path)

    def load_ckpt(self, ckpt_path=None):
        """
        Load the model checkpoint

        Args:
            ckpt_path: Path to the checkpoint
        """
        net_dict = torch.load(ckpt_path if ckpt_path is not None else self.ckpt_path,
                               map_location='cpu', weights_only=True)
        self.net.load_state_dict(net_dict)
        self.from_scratch = False

    def forward(self, x):
        """
        Forward pass through the model

        Args:
            x: Input data

        Returns:
            Model predictions
        """
        return self.net.forward(x)

    def fit(self, train_data, val_data=None, epochs=10, ckpt_path='checkpoint',
            patience=5, monitor="val_loss", mode="min", 
            callbacks=None, plot=True, wandb=False, 
            mixed_precision='no', cpu=False, gradient_accumulation_steps=1):
        """
        Train the model.

        Args:
            train_data: Training data
            val_data: Validation data
            epochs: Number of training epochs
            ckpt_path: Path to save model checkpoints
            patience: Number of epochs with no improvement after which training will be stopped
            monitor: Metric to monitor for early stopping
            mode: 'min' for minimizing the monitor metric, 'max' for maximizing
            callbacks: List of callback functions
            plot: Whether to plot training progress
            wandb: Whether to use WandB for logging
            mixed_precision: Mixed precision training ('no', 'O1', 'O2', 'O3')
            cpu: Use CPU for training
            gradient_accumulation_steps: Number of steps to accumulate gradients before optimizer step

        Returns:
            DataFrame containing training history.
        """
        self.__dict__.update(locals())
        from accelerate import Accelerator
        from torchkeras.utils import colorful, is_jupyter

        self.accelerator = Accelerator(mixed_precision=mixed_precision, cpu=cpu,
                                       gradient_accumulation_steps=gradient_accumulation_steps)

        device = str(self.accelerator.device)
        device_type = '🐌' if 'cpu' in device else ('⚡️' if 'cuda' in device else '🚀')
        self.accelerator.print(
            colorful("<<<<<< " + device_type + " " + device + " is used >>>>>>"))

        self.net, self.loss_fn, self.metrics_dict, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.net, self.loss_fn, self.metrics_dict, self.optimizer, self.lr_scheduler)

        for key in self.kwargs:
            self.kwargs[key] = self.accelerator.prepare(self.kwargs[key])

        train_dataloader, val_dataloader = self.accelerator.prepare(train_data, val_data)
        train_dataloader.size = train_data.size if hasattr(train_data, 'size') else len(train_data)
        train_dataloader.size = min(train_dataloader.size, len(train_dataloader))

        if val_data:
            val_dataloader.size = val_data.size if hasattr(val_data, 'size') else len(val_data)
            val_dataloader.size = min(val_dataloader.size, len(val_dataloader))

        self.history = {}
        callbacks = callbacks if callbacks is not None else []

        
        if bool(plot):
            from torchkeras.kerascallbacks import VisProgress, VisMetric
            callbacks = [VisMetric(), VisProgress()] + callbacks

        if wandb != False:
            from torchkeras.kerascallbacks import WandbCallback
            project = wandb if isinstance(wandb, str) else 'torchkeras'
            callbacks.append(WandbCallback(project=project))

        self.callbacks = [self.accelerator.prepare(x) for x in callbacks]

        if self.accelerator.is_local_main_process:
            [cb.on_fit_start(model=self) for cb in self.callbacks if hasattr(cb, 'on_fit_start')]

        start_epoch = 1 if self.from_scratch else 0
        quiet = bool(plot)
        
        for epoch in range(start_epoch, epochs + 1):
            if not quiet:
                nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.accelerator.print("\n" + "==========" * 8 + "%s" % nowtime)
                self.accelerator.print("Epoch {0} / {1}".format(epoch, epochs) + "\n")

            # 1，train -------------------------------------------------
            train_step_runner = self.StepRunner(
                net=self.net,
                loss_fn=self.loss_fn,
                accelerator=self.accelerator,
                stage="train",
                metrics_dict=deepcopy(self.metrics_dict),
                optimizer=self.optimizer if epoch > 0 else None,
                lr_scheduler=self.lr_scheduler if epoch > 0 else None,
                **self.kwargs
            )

            train_epoch_runner = self.EpochRunner(train_step_runner, quiet)
            train_metrics = {'epoch': epoch}
            train_metrics.update(train_epoch_runner(train_dataloader))

            for name, metric in train_metrics.items():
                self.history[name] = self.history.get(name, []) + [metric]

            if self.accelerator.is_local_main_process:
                [cb.on_train_epoch_end(model=self) for cb in self.callbacks
                 if hasattr(cb, 'on_train_epoch_end')]
            # 2，validate -------------------------------------------------
            if val_dataloader is not None:
                val_step_runner = self.StepRunner(
                    net = self.net,
                    loss_fn = self.loss_fn,
                    accelerator = self.accelerator,
                    stage="val",
                    metrics_dict= deepcopy(self.metrics_dict),
                    **self.kwargs
                )
                val_epoch_runner = self.EpochRunner(val_step_runner,quiet)
                with torch.no_grad():
                    val_metrics = val_epoch_runner(val_dataloader)

                for name, metric in val_metrics.items():
                    self.history[name] = self.history.get(name, []) + [metric]
                
            if self.accelerator.is_local_main_process:
                [cb.on_validation_epoch_end(model = self) for cb in self.callbacks 
                 if hasattr(cb,'on_validation_epoch_end')]

            # 3，early-stopping -------------------------------------------------
            self.accelerator.wait_for_everyone()
            arr_scores = self.history[monitor]
            best_score_idx = np.argmax(arr_scores) if mode=="max" else np.argmin(arr_scores)

            if best_score_idx==len(arr_scores)-1 and self.accelerator.is_local_main_process:
                self.save_ckpt(ckpt_path,accelerator = self.accelerator)
                if not quiet:
                    self.accelerator.print(colorful("<<<<<< reach best {0} : {1} >>>>>>".format(
                        monitor,arr_scores[best_score_idx])))

            if len(arr_scores)-best_score_idx>patience:
                break
                
        if self.accelerator.is_local_main_process:   
            dfhistory = pd.DataFrame(self.history)
            [cb.on_fit_end(model = self) for cb in self.callbacks 
                 if hasattr(cb,'on_fit_end')]
            if epoch<epochs:
                self.accelerator.print(colorful(
                        "<<<<<< {} without improvement in {} epoch,""early stopping >>>>>> \n"
                    ).format(monitor,patience))
            self.net = self.accelerator.unwrap_model(self.net)
            self.net.cpu()
            self.load_ckpt(ckpt_path)
            return dfhistory
        
    def evaluate(self, val_data, quiet=False):
        """
        Evaluate the model on validation data

        Args:
            val_data: Validation data
            quiet: Whether to suppress evaluation progress logs
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Ensure accelerator is available or create a new one
        from accelerate import Accelerator
        accelerator = Accelerator() if not hasattr(self, 'accelerator') else self.accelerator

        # Prepare model, loss function, and metrics for evaluation
        self.net, self.loss_fn, self.metrics_dict = accelerator.prepare(
            self.net, self.loss_fn, self.metrics_dict)

        val_data = accelerator.prepare(val_data)

        # Initialize StepRunner for validation
        val_step_runner = self.StepRunner(net=self.net, stage="val",
                                          loss_fn=self.loss_fn, metrics_dict=deepcopy(self.metrics_dict),
                                          accelerator=accelerator)

        # Initialize EpochRunner for validation
        val_epoch_runner = self.EpochRunner(val_step_runner, quiet=quiet)

        # Evaluate on validation data without gradient computation
        with torch.no_grad():
            val_metrics = val_epoch_runner(val_data)

        return val_metrics
    
    def fit_ddp(self, num_processes, train_data,
                val_data=None, epochs=10, ckpt_path='checkpoint',
                patience=5, monitor="val_loss", mode="min", callbacks=None,
                plot=True, wandb=False, mixed_precision='no', 
                cpu=False, gradient_accumulation_steps=1):
        """
        Distributed Data Parallel (DDP) training for the model.

        Args:
            num_processes: Number of processes for DDP
            train_data: Training data
            val_data: Validation data
            epochs: Number of training epochs
            ckpt_path: Path to save model checkpoints
            patience: Number of epochs with no improvement after which training will be stopped
            monitor: Metric to monitor for early stopping
            mode: 'min' for minimizing the monitor metric, 'max' for maximizing
            callbacks: List of callback functions
            plot: Whether to plot training progress
            wandb: Whether to use WandB for logging
            mixed_precision: Mixed precision training ('no', 'O1', 'O2', 'O3')
            cpu: Use CPU for training
            gradient_accumulation_steps: Number of steps to accumulate gradients before optimizer step
        """
        # Import notebook_launcher from accelerate
        from accelerate import notebook_launcher

        # Create a tuple of arguments for the fit method
        args = (train_data, val_data, epochs, ckpt_path, patience, monitor, mode,
                callbacks, plot, wandb, mixed_precision, cpu, gradient_accumulation_steps)

        # Launch the fit method using notebook_launcher
        notebook_launcher(self.fit, args, num_processes=num_processes)
    
    def evaluate_ddp(self, num_processes, val_data, quiet=False):
        """
        Distributed Data Parallel (DDP) evaluation for the model

        Args:
            num_processes: Number of processes for DDP
            val_data: Validation data.
            quiet: Whether to suppress evaluation progress logs

        Returns:
            Dictionary of evaluation metrics
        """
        # Import notebook_launcher from accelerate
        from accelerate import notebook_launcher

        # Create a tuple of arguments for the evaluate method
        args = (val_data, quiet)

        # Launch the evaluate method using notebook_launcher
        notebook_launcher(self.evaluate, args, num_processes=num_processes)