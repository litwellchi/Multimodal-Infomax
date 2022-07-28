from solver import Solver

class fed_solver(Solver):
    def __init__(self, hyp_params, train_loader, dev_loader, test_loader, is_train=True, model=None, pretrained_emb=None):
        super().__init__(hyp_params, train_loader, dev_loader, test_loader, is_train, model, pretrained_emb)

    def train_and_eval(self):
        return super().train_and_eval()