import sys

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np


class ReverseFCNet(nn.Module):
    def __init__(self, cfg):
        super(ReverseFCNet, self).__init__()
        input_size = cfg['problems']['input_size']
        output_size = cfg['problems']['output_size']

        activation_type = cfg['model']['activation']
        if activation_type == "ReLU":
            activation_cls = nn.ReLU
        elif activation_type == "ELU":
            activation_cls = nn.ELU
        elif activation_type == "LeakyReLU":
            activation_cls = nn.LeakyReLU

        fc_sizes = cfg['model']['fc_sizes'] + [output_size]

        net = []
        last_fc_size = input_size
        for size in fc_sizes:
            net.append(nn.Linear(last_fc_size, size))
            net.append(activation_cls())
            last_fc_size = size

        net.pop(-1)

        # net = [
        #   nn.Linear(1, 100),
        #   nn.ReLU(),
        #   nn.Linear(100, 2)
        # ]
        self.fc_net = nn.Sequential(*net)
        print(self.fc_net)

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], 1))
        return self.fc_net(x)


class RegressionOptimizer:
    def __init__(self, cfg, train_data_loader, test_data_loader, logger):
        self.cfg = cfg
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.logger = logger

        # Load cfg variables.
        lr = cfg['model']['lr']
        sgd_momentum = cfg['model']['optimizer_sgd_momentum']
        self.batch_size = cfg['model']['batch_size']
        self.n_epochs = cfg['model']['n_epochs']
        self.train_eval_split = cfg['model']['train_eval_split']

        # Set it all up.
        # TODO 1 is hardcoded.
        self.net = ReverseFCNet(cfg)
        self.criterion = nn.MSELoss()
        if cfg['model']['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                self.net.parameters(), lr=lr, momentum=sgd_momentum
            )
        elif cfg['model']['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(
                self.net.parameters(), lr=lr
            )

    def train(self):
        self.net.train()
        data_len = len(self.train_data_loader)
        for epoch in range(self.n_epochs):
            print(f'\n\nEPOCH #{epoch}')
            batch_loss = 0.
            all_res_x = []
            all_res_y = []

            for i, data in enumerate(self.train_data_loader):
                inputs, labels = data

                self.optimizer.zero_grad()

                outputs = self.net(inputs.float())

                outputs_temp = outputs.detach().numpy()
                labels_temp = labels.numpy()
                res = np.abs(outputs_temp-labels_temp)
                avg_res = np.mean(res, axis=0)
                all_res_x.append(avg_res[0])
                all_res_y.append(avg_res[1])
                
                loss = self.criterion(outputs, labels.float())
                loss.backward()
                self.optimizer.step()

                # TODO!!! What if batch_size is not a factor of total size.
                # Then the last term will be wrong.
                batch_loss += loss.item() * self.batch_size
                if i % 1000 == 0:
                    print('outputs = \n', outputs)
                    print('labels = \n', labels)

                    avg_loss = batch_loss / (i + 1)
                    msg = '[%d, %5d] loss: %.3f' % (epoch + 1, i, avg_loss)
                    sys.stdout.write('\r' + msg)
                    sys.stdout.flush()

                    self.logger.append_loss(avg_loss)

                # if i % 1000 == 0:
                #     for param in self.net.parameters():
                #         print(param.data)
                #         # print(param.shape)
                #     print('')

                # if i % 1000 == 0:
                #     data = {
                #         "loss_train": batch_loss / (i + 1)
                #     }
                #     self.logger.log_train(data, data_len * epoch + i)
            res_x = np.mean(all_res_x)
            res_y = np.mean(all_res_y)
            res_x = float(res_x)
            res_y = float(res_y)
            self.logger.append_res_x(res_x)
            self.logger.append_res_y(res_y)

            self.net.eval()
            data = {}
            test_loss = self.eval(epoch, do_print=False, debug=epoch % 10 == 0)
            data['loss_eval'] = test_loss
            # self.logger.log_eval_reverse(data, epoch)
            self.net.train()
        print('')

    def eval(self, epoch, do_print=True, debug=False):
        sse = 0
        ssm_mean = None
        n = 0

        self.net.eval()
        total_loss = 0.0
        for i, data in enumerate(self.test_data_loader):
            inputs, labels = data
            outputs = self.net(inputs.float())

            loss = self.criterion(outputs, labels)

            sse += ((labels.numpy() - outputs[0].detach().numpy()) ** 2).sum()
            if ssm_mean is None:
                ssm_mean = labels.numpy()
            else:
                ssm_mean += labels.numpy()
            n += 1

            total_loss += loss.item()
            if do_print and i % 1000 == 0:
                msg = '[%d] loss: %.3f' % (i, total_loss / (i + 1))
                sys.stdout.write('\r' + msg)
                sys.stdout.flush()

        ssm_mean /= n
        ssm = 0
        for i, data in enumerate(self.test_data_loader):
            inputs, labels = data
            ssm += ((labels.numpy() - ssm_mean) ** 2).sum()
        R2 = 1 - (sse / ssm)
        print(" ", sse, ssm)
        print("R2", R2)

        # self.logger.log_custom_reverse_kpi("R2", R2, epoch)
        self.logger.append_r2(R2)

        data_len = len(self.test_data_loader)

        return total_loss / data_len

    def save(self, model_fname):
        torch.save(self.net.state_dict(), 'models/' + model_fname)

    def load(self, model_fname, output_size):
        self.net = ReverseFCNet(self.cfg, output_size)
        self.net.load_state_dict(torch.load(model_fname))
