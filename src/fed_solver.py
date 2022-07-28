import time
import copy
import torch
import numpy as np
import torch.nn as nn

from solver import Solver
from utils.eval_metrics import *
from utils.tools import *

class FedSolver(Solver):
    def __init__(self, hyp_params, train_loader, dev_loader, test_loader, is_train=True, model=None, pretrained_emb=None):
        super().__init__(hyp_params, train_loader, dev_loader, test_loader, is_train, model, pretrained_emb)

    def train_and_eval(self):
        for epoch in range(self.hp.num_epochs):
            start_time = time.time()
            w_locals = []
            self.model.train()
            net_glob = self.model.state_dict()

            idxs_users = np.random.choice(
                self.hp.clients, self.hp.samples, replace=False)

            for idx in idxs_users:
                print(f'training client {idx}...')
                client_config = self.hp
                client_config.num_epochs = 2
                lc_model = copy.deepcopy(self.model)
                train_data_loader = self.train_loader[idx]
                local_train = Solver(client_config, 
                    train_loader=train_data_loader, 
                    dev_loader=self.dev_loader,
                    test_loader = self.test_loader,
                    is_train=True,
                    model=lc_model)
                local_train.train_and_eval()

                local_w = local_train.model.state_dict()
                w_locals.append(copy.deepcopy(local_w))
            
            net_glob = self.FedAvg(w_locals)
            self.model.load_state_dict(net_glob)
            self.evaluate(self.model, nn.L1Loss())

            val_loss, _, _ = self.evaluate(self.model, self.criterion, test=False)
            test_loss, results, truths = self.evaluate(self.model, self.criterion, test=True)
            
            self.scheduler_main.step(val_loss)    # Decay learning rate by validation loss

            # validation F1
            print("-"*50)
            print('Server Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, time.time()-start_time, val_loss, test_loss))
            print("-"*50)

        if self.hp.dataset in ["mosei_senti", "mosei"]:
            eval_mosei_senti(results, truths, True)
        elif self.hp.dataset == 'mosi':
            self.best_dict = eval_mosi(results, truths, True)
        elif self.hp.dataset == 'iemocap':
            eval_iemocap(results, truths)      


    def evaluate(self, model, criterion, test=False):
            model.eval()
            loader = self.test_loader if test else self.dev_loader
            total_loss = 0.0
            total_l1_loss = 0.0
        
            results = []
            truths = []

            with torch.no_grad():
                for batch in loader:
                    text, vision, vlens, audio, alens, y, lengths, bert_sent, bert_sent_type, bert_sent_mask, ids = batch

                    with torch.cuda.device(0):
                        text, audio, vision, y = text.cuda(), audio.cuda(), vision.cuda(), y.cuda()
                        lengths = lengths.cuda()
                        bert_sent, bert_sent_type, bert_sent_mask = bert_sent.cuda(), bert_sent_type.cuda(), bert_sent_mask.cuda()
                        if self.hp.dataset == 'iemocap':
                            y = y.long()
                    
                        if self.hp.dataset == 'ur_funny':
                            y = y.squeeze()

                    batch_size = lengths.size(0) # bert_sent in size (bs, seq_len, emb_size)

                    # we don't need lld and bound anymore
                    _, _, preds, _, _ = model(text, vision, audio, vlens, alens, bert_sent, bert_sent_type, bert_sent_mask)

                    if self.hp.dataset in ['mosi', 'mosei', 'mosei_senti'] and test:
                        criterion = nn.L1Loss()

                    total_loss += criterion(preds, y).item() * batch_size

                    # Collect the results into ntest if test else self.hp.n_valid)
                    results.append(preds)
                    truths.append(y)
            
            avg_loss = total_loss / (self.hp.n_test if test else self.hp.n_valid)

            results = torch.cat(results)
            truths = torch.cat(truths)
            return avg_loss, results, truths

    def FedAvg(self, w):
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
        return w_avg