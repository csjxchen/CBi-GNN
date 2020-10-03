
import torch.nn as nn

classs RPN(nn.Module):
    def __init__(self, model_cfg, train_cfg, test_cfg, is_train=True):
        super(RPN, self).__init__()
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.is_train = is_train
        # for k, m in self.model_cfg.items():
            # eval_model_str = f"self.{k} = {m.type}(**m)"
            # eval(eval_model_str)
    @abstractmethod
    def init_architecture(self):
        pass

    @abstractmethod
    def forward(self, data):
        if self.is_train:
            self.forward_train(data)
        else:
            self.forward_test(data)

    @abstractmethod
    def forward_train(self, data):
        pass
    
    @abstractmethod
    def forward_test(self, data):
        pass