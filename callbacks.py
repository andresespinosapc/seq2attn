from machine.util.callbacks import Callback


class CometLogger(Callback):
    def __init__(self, comet_experiment):
        super().__init__()

        self.experiment = comet_experiment

    def on_epoch_end(self, info=None):
        metrics = {
            'epoch': info['epoch'],
        }
        for split in ['train', 'eval']:
            for metric in info['%s_metrics' % (split)]:
                metrics.update({
                    '%s_%s' % (split, metric.log_name): metric.get_val(),
                })
            for loss in info['%s_losses' % (split)]:
                metrics.update({
                    '%s_%s' % (split, loss.log_name): loss.get_loss(),
                })
        self.experiment.log_metrics(metrics)
