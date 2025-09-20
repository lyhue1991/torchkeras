"""
StepRunner for tabular models
"""

class StepRunner:
    def __init__(self, net, loss_fn, accelerator=None, stage="train", metrics_dict=None,
                 optimizer=None, lr_scheduler=None, **kwargs):
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
        # Compute loss
        if self.loss_fn is not None:
            self.net.loss = self.loss_fn
            
        with self.accelerator.autocast():
            output = self.net(batch)
            preds = output['logits'] if 'yhat' not in output.keys() else output['yhat']
            labels = batch['target']
            loss = self.net.compute_loss(output, labels)

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