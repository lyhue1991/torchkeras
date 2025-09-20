import torch
from transformers import Trainer,TrainingArguments,EarlyStoppingCallback,TrainerCallback
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import is_peft_available
import numpy as np
import pandas as pd 
import os 
import matplotlib.pyplot as plt 
from torchkeras.utils import is_jupyter

"""
@author : lyhue1991
@description : hug model code
"""

class HugModel(torch.nn.Module):
    """Custom PyTorch module for training and evaluating models using the Hugging Face Transformers library."""

    def __init__(self, net=None, loss_fn=None, metrics_dict=None,
                 label_names=None, feature_names=None):
        """
        Args:
            net (torch.nn.Module): The neural network model
            loss_fn: The loss function for training
            metrics_dict (dict): A dictionary of evaluation metrics
            label_names (list): List of label names
            feature_names (list): List of feature names
        """
        super().__init__()
        self.net, self.loss_fn = net, loss_fn
        self.metrics_dict = metrics_dict if metrics_dict is not None else {}

        # Attention here!
        self.feature_names = ['features'] if feature_names is None else feature_names
        self.label_names = ['labels'] if label_names is None else label_names

        self.training_args = None
        self.trainer = None

    def forward(self, **batch):
        """Forward pass of the model.

        Args:
            **batch: Keyword arguments representing the input batch

        Returns:
            dict: Output dictionary containing logits and loss
        """
        # Model forward pass logic
        if self.loss_fn is None:
            return self.net(**batch)
        
        elif set(batch.keys())=={'features','labels'}:
            features = batch['features']
            labels = batch['labels']
            logits = self.net.forward(features)
            loss = self.loss_fn(logits,labels)
            out = {'logits':logits,'loss':loss}
            return out 
        
        else:
            if len(self.feature_names)==1:
                features = batch[self.feature_names[0]]
                logits = self.net.forward(features)
            else:
                features = {k:batch[k] for k in self.feature_names}
                logits = self.net.forward(**features)
            
            if len(self.label_names)==1:
                labels = batch[self.label_names[0]]
            else:
                labels = {k:batch[k] for k in self.label_names}
                
            loss = self.loss_fn(logits,labels)
            out = {'logits':logits,'loss':loss}
            return out 

    def compute_metrics(self, eval_preds):
        """Compute evaluation metrics.

        Args:
            eval_preds: Tuple of logits and labels

        Returns:
            dict: Dictionary of computed metrics
        """
        logits, labels = eval_preds
        for k, m in self.metrics_dict.items():
            m(torch.from_numpy(logits), torch.from_numpy(labels))
        result = {k: m.compute() for k, m in self.metrics_dict.items()}
        for k, m in self.metrics_dict.items():
            m.reset()
        return result

    def get_collate_fn(self, default_fn):
        """Get custom collate function for the data loader

        Args:
            default_fn: Default collate function

        Returns:
            function: Custom collate function
        """
        def collate_fn(examples):
            batch = default_fn(examples)
            if hasattr(batch, 'keys') and hasattr(batch, 'pop'):
                return batch
            elif isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    return {'features': batch[0], 'labels': batch[1]}
            else:
                raise Exception('dataset format error!')
        return collate_fn

    def fit(self, train_data, val_data=None, output_dir='output_dir',
            epochs=10, learning_rate=5e-5, logging_steps=20,
            monitor="val_loss", patience=20, mode="min",
            plot=True, wandb=False,
            no_cuda=False, use_mps_device=False,
            **kwargs):
        """Train the model.

        Args:
            train_data: Training data
            val_data: Validation data
            output_dir: Output directory for saving checkpoints
            epochs: Number of training epochs
            learning_rate: Learning rate for training
            logging_steps: Number of steps to log information
            monitor: Metric to monitor for early stopping
            patience: Patience for early stopping
            mode: Early stopping mode ('min' or 'max')
            plot: Whether to plot training progress
            wandb: Whether to use Weights & Biases for logging
            no_cuda: Whether to use CUDA
            use_mps_device: Whether to use Mixed Precision Training
            **kwargs: Additional keyword arguments
        """
        self.train_data,self.val_data,self.monitor,self.mode = train_data,val_data,monitor,mode
        
        if wandb ==False:
            os.environ["WANDB_DISABLED"] = "true"
        else:
            os.environ["WANDB_DISABLED"] = "false"
            import wandb as wb
            import datetime
            name =datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            wb.init(
                    project=wandb if isinstance(wandb,str) else "HugModel",
                    name = name,save_code=True
                )
            self.run_id  = wb.run.id
            self.project_name = wb.run.project_name()
            
        self.prefix = 'eval' if monitor.startswith('eval') else 'val'
        monitor_metric = 'e'+monitor if monitor.startswith('val_') else monitor
        
        self.training_args = TrainingArguments(
            output_dir = output_dir,
            num_train_epochs = epochs,
            learning_rate = learning_rate,
            logging_steps = logging_steps,
            
            evaluation_strategy="steps", #epoch
            metric_for_best_model= monitor_metric,
            greater_is_better= False if mode=='min' else True,
            report_to='wandb' if wandb else [],
            
            load_best_model_at_end= True,
            remove_unused_columns = False,
            label_names = self.label_names,
            
            per_device_train_batch_size = train_data.batch_size,
            per_device_eval_batch_size = val_data.batch_size,
            dataloader_drop_last=train_data.drop_last,
            dataloader_num_workers = train_data.num_workers,
            dataloader_pin_memory = train_data.pin_memory,
            
            no_cuda = no_cuda, 
            use_mps_device = use_mps_device,
            **kwargs
        )
        
        callbacks = [EarlyStoppingCallback(early_stopping_patience= patience)]
        
        if plot and is_jupyter():
            callbacks.append(VisCallback(monitor=monitor))
      
        self.trainer = Trainer(
            self,
            self.training_args,
            train_dataset=train_data.dataset,
            eval_dataset=val_data.dataset,
            compute_metrics=self.compute_metrics,
            callbacks = callbacks,
            data_collator=self.get_collate_fn(train_data.collate_fn)
        )
        
        self.trainer.train()
        
        #save best ckpt
        self.ckpt_path = os.path.join(output_dir,'best')
        self.save_ckpt(self.ckpt_path)
                
        if wandb:
            arti_model = wb.Artifact('checkpoint', type='model')
            arti_model.add_file(self.ckpt_path)
            wb.log_artifact(arti_model)
            wb.finish()
    

    def evaluate(self, val_data, **kwargs):
        """Evaluate the model on validation data

        Args:
            val_data: Validation data.
            **kwargs: Additional keyword arguments

        Returns:
            dict: Dictionary of evaluation metrics
        """
        dl_val = self.trainer.get_eval_dataloader(val_data.dataset)
        out = self.trainer.evaluation_loop(dl_val,
                                           description=self.prefix, prediction_loss_only=False,
                                           metric_key_prefix=self.prefix).metrics
        return out

    def save_ckpt(self, ckpt_path):
        """Save model checkpoint

        Args:
            ckpt_path: Path to save the checkpoint
        """
        supported_classes = [PreTrainedModel]
        if is_peft_available():
            from peft import PeftModel
            supported_classes.append(PeftModel)
        supported_classes = tuple(supported_classes)
        if isinstance(self.net, supported_classes) and self.trainer is not None:
            self.trainer.save_model(ckpt_path)
        else:
            torch.save(self.net.state_dict(), ckpt_path)

    def load_ckpt(self, ckpt_path):
        """Load model checkpoint

        Args:
            ckpt_path: Path to load the checkpoint from
        """
        if is_peft_available():
            from peft import PeftModel
            if isinstance(self.net, PeftModel):
                self.net = self.net.from_pretrained(self.net.base_model.model, ckpt_path)
        elif isinstance(self.net, PreTrainedModel):
            self.net = self.net.from_pretrained(ckpt_path)
        else:
            self.net.load_state_dict(torch.load(ckpt_path))
    def debug(self):
        """Debugging method to identify issues during model fitting."""
        # Step 1: Check label_names
        trainer = self.trainer
        print('Step 1: Check label_names.')
        print('label_names = ', trainer.label_names, '\n')

        # Step 2: Check dataset keys
        print('Step 2: Check dataset keys.')
        dl_val = trainer.get_eval_dataloader()
        for step, inputs in enumerate(dl_val):
            break
        print('batch.keys() = ', inputs.keys(), '\n')

        # Step 3: Check labels not None
        print('Step 3: Check labels not None.')
        labels_check = {k + '!=None': inputs.get(k) is not None for k in trainer.label_names}
        print(labels_check)
        has_labels = False if len(trainer.label_names) == 0 else all(
            inputs.get(k) is not None for k in trainer.label_names)
        print('has_labels = ', has_labels, '\n')

        # Step 4: Check trainer.prediction_step
        print('Step 4: Check trainer.prediction_step.')
        loss, logits, labels = trainer.prediction_step(self, inputs, False)
        print('loss = ', loss, '\n')

        # Step 5: Check trainer.evaluation_loop
        print('Step 5: Check trainer.evaluation_loop.')
        out = trainer.evaluation_loop(dl_val, description=self.prefix, metric_key_prefix=self.prefix)
        print('metrics:', out.metrics)

class VisCallback(TrainerCallback):
    def __init__(self, figsize=(6, 4), update_freq=1, monitor=None):
        """
        Visualization callback for Hugging Face Transformers Trainer

        Args:
            figsize (tuple, optional): Figure size. Defaults to (6, 4)
            update_freq (int, optional): Frequency of updating the visualization. Defaults to 1
            monitor (str, optional): Metric to monitor. Defaults to None
        """
        self.figsize = figsize
        self.update_freq = update_freq
        self.monitor = monitor

    def on_train_begin(self, args, state, control, **kwargs):
        """
        Called at the beginning of training

        Args:
            args (TrainingArguments): Training arguments
            state (TrainerState): Trainer state
            control (TrainerControl): Trainer control
        """
        metric = args.metric_for_best_model
        self.greater_is_better = args.greater_is_better
        self.prefix = 'val_' if self.monitor is None or self.monitor.startswith('val_') else 'eval_'
        self.metric = metric if self.monitor is None else self.monitor

        dfhistory = pd.DataFrame()
        x_bounds = [0, args.logging_steps * 10]
        self.update_graph(dfhistory, self.metric.replace(self.prefix, ''), x_bounds=x_bounds, figsize=self.figsize)

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """
        Called after each evaluation

        Args:
            args (TrainingArguments): Training arguments
            state (TrainerState): Trainer state
            control (TrainerControl): Trainer control
            metrics (Dict): Evaluation metrics
        """
        dfhistory = self.get_history(state)
        n = dfhistory['step'].max()
        if n % self.update_freq == 0:
            x_bounds = [dfhistory['step'].min(),
                        10 * args.logging_steps + (n // (args.logging_steps * 10)) * args.logging_steps * 10]
            self.update_graph(dfhistory, self.metric.replace(self.prefix, ''), x_bounds=x_bounds, figsize=self.figsize)

    def on_train_end(self, args, state, control, **kwargs):
        """
        Called at the end of training

        Args:
            args (TrainingArguments): Training arguments
            state (TrainerState): Trainer state
            control (TrainerControl): Trainer control
        """
        dfhistory = self.get_history(state)
        self.update_graph(dfhistory, self.metric.replace(self.prefix, ''), figsize=self.figsize)

    def get_history(self, state):
        """
        Get training and evaluation history from TrainerState

        Args:
            state (TrainerState): Trainer state

        Returns:
            DataFrame: Combined training and evaluation history
        """
        log_history = state.log_history
        train_history = [x for x in log_history if 'loss' in x.keys()]
        eval_history = [x for x in log_history if 'eval_loss' in x.keys()]

        dfhistory_train = pd.DataFrame(train_history)
        dfhistory_eval = pd.DataFrame(eval_history)
        dfhistory = dfhistory_train.merge(dfhistory_eval, on=['step', 'epoch'])
        if self.prefix == 'val_':
            dfhistory.columns = [x.replace('eval_', 'val_') for x in dfhistory.columns]
        return dfhistory

    def get_best_score(self, dfhistory):
        """
        Get the best score and corresponding step

        Args:
            dfhistory (DataFrame): Combined training and evaluation history

        Returns:
            Tuple: (best_step, best_score)
        """
        arr_scores = dfhistory[self.metric]
        best_score = np.max(arr_scores) if self.greater_is_better else np.min(arr_scores)
        best_step = dfhistory.loc[arr_scores == best_score, 'step'].tolist()[0]
        return best_step, best_score

    def update_graph(self, dfhistory, metric, x_bounds=None, y_bounds=None, figsize=(6, 4)):
        """
        Update the visualization graph

        Args:
            dfhistory (DataFrame): Combined training and evaluation history
            metric (str): Metric to visualize
            x_bounds (list, optional): X-axis bounds. Defaults to None
            y_bounds (list, optional): Y-axis bounds. Defaults to None
            figsize (tuple, optional): Figure size. Defaults to (6, 4)
        """
        from IPython.display import display
        if not hasattr(self, 'graph_fig'):
            self.graph_fig, self.graph_ax = plt.subplots(1, figsize=figsize)
            self.graph_out = display(self.graph_ax.figure, display_id=True)

        self.graph_ax.clear()
        steps = dfhistory['step'] if 'step' in dfhistory.columns else []

        m1 = metric
        if m1 in dfhistory.columns:
            train_metrics = dfhistory[m1]
            self.graph_ax.plot(steps, train_metrics, 'bo--', label=m1, clip_on=False)

        m2 = self.prefix + metric
        if m2 in dfhistory.columns:
            val_metrics = dfhistory[m2]
            self.graph_ax.plot(steps, val_metrics, 'co-', label=m2, clip_on=False)

        self.graph_ax.set_xlabel("step")
        self.graph_ax.set_ylabel(metric)

        if m1 in dfhistory.columns or m2 in dfhistory.columns or metric in dfhistory.columns:
            self.graph_ax.legend(loc='best')

        if len(steps) > 0:
            best_step, best_score = self.get_best_score(dfhistory)
            self.graph_ax.plot(best_step, best_score, 'r*', markersize=15, clip_on=False)
            title = f'best {self.metric} = {best_score:.4f} (@step {best_step})'
            self.graph_ax.set_title(title)
        else:
            title = f'best {self.metric} = ?'
            self.graph_ax.set_title(title)

        if x_bounds is not None:
            self.graph_ax.set_xlim(*x_bounds)
        if y_bounds is not None:
            self.graph_ax.set_ylim(*y_bounds)
        self.graph_out.update(self.graph_ax.figure)
        plt.close()
