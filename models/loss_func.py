from torch import from_numpy, zeros, exp, FloatTensor
from torch.nn import Module, CrossEntropyLoss, Parameter

from models import run_device


class MultiTaskLossWrapper(Module):
    def __init__(self, num_tasks, weights=None, log_vars=None):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = num_tasks
        self.loss_func = [CrossEntropyLoss(weight=weights[i] if weights else None) for i in range(num_tasks)]
        self.log_vars = Parameter(from_numpy(log_vars) if log_vars else zeros(num_tasks))

    def forward(self, predictions, targets):
        running_loss = 0.0
        for idx in range(self.task_num):
            loss = self.loss_func[idx](predictions[idx], targets[idx])
            running_loss += (exp(-self.log_vars[idx]) * loss) + self.log_vars[idx]
        return running_loss


def calculate_class_weights(config, normalized_counts):
    # Loss weighting proposed in https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html
    class_weights = (1 - config.beta) / (1 - (config.beta ** normalized_counts))
    return FloatTensor(class_weights).to(run_device)


def get_loss_function(config, features):
    if config.use_class_weights:
        gender_class_weights = calculate_class_weights(config, features['mappings']['gender']['weights'])
        accent_class_weights = calculate_class_weights(config, features['mappings']['accent']['weights'])
        loss_func = MultiTaskLossWrapper(2, (gender_class_weights, accent_class_weights))
    else:
        loss_func = MultiTaskLossWrapper(2)

    return loss_func.to(run_device)