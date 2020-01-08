from machine.loss import Loss
import torch


class L1Loss(Loss):
    def __init__(self, input_name='encoder_hidden'):
        self.name = 'L1 %s Loss' % (input_name)
        self.log_name = 'l1_%s_loss' % (input_name)
        self.inputs = input_name
        self.acc_loss = 0
        self.norm_term = 0
        self.criterion = torch.tensor([])

    def get_loss(self):
        if isinstance(self.acc_loss, int):
            return 0

        return self.acc_loss.item() / self.norm_term

    def eval_batch(self, decoder_outputs, other, target_variable):
        outputs = other[self.inputs]
        if self.inputs == 'model_parameters':
            for parameter in outputs:
                self.acc_loss += parameter.abs().sum()
                self.norm_term += 1
        else:
            batch_size = outputs.size(0)
            self.acc_loss += outputs.abs().sum()
            self.norm_term += batch_size
