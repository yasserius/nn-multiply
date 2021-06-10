import os
import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl

import datetime

from torch.utils.tensorboard import SummaryWriter


# mpl.font_manager._rebuild()
# plt.rc('font', family='Raleway')


def truncate_colormap(cmapIn='jet', minval=0.0, maxval=1.0, n=100):
    cmapIn = plt.get_cmap(cmapIn)

    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmapIn.name, a=minval, b=maxval),
        cmapIn(np.linspace(minval, maxval, n)))

    return new_cmap

class Logger:
    def __init__(self, model_fname, cfg):
        self.cfg = cfg
        self.model_fname = model_fname
        self.log = {
            'config': self.cfg,
            'config_name': self.model_fname,
            'loss': [],
            'r2': [],
            'res_x': [],
            'res_y': [],
        }

    def save_log(self):
        now = datetime.datetime.now().strftime('%m-%d-%Y-%H-%M-%S')
        dest_dir = f"runs/{self.model_fname}/"
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        fn = dest_dir + f"{now}_{self.model_fname}.json"
        with open(fn, 'w') as f:
            f.write(json.dumps(self.log))
            print("Saved log " + fn)

    def append_loss(self, loss):
        self.log['loss'].append(loss)
    
    def append_r2(self, r2):
        self.log['r2'].append(r2)
        
    def append_res_x(self, res_x):
        self.log['res_x'].append(res_x)
        
    def append_res_y(self, res_y):
        self.log['res_y'].append(res_y)

# class Logger:
#     def __init__(self, model_fname, cfg):
#         self.cfg = cfg
#         self.model_fname = model_fname
#         self.writer = SummaryWriter(log_dir='runs/' + model_fname)

#     def log_config(self):
#         self.writer.add_text('Info/Config', json.dumps(self.cfg), 0)

#     def log_train(self, data, n_iter):
#         self.writer.add_scalar('Loss/Train', data['loss_train'], n_iter)

#     def log_eval(self, data, n_iter):
#         self.writer.add_scalar('Loss/Eval', data['loss_eval'], n_iter)
#         for k, v in data['Problem_Misclassifications'].items():
#             self.writer.add_scalar(
#                 'Problem_Misclassifications/' + k,
#                 v,
#                 n_iter
#             )
#         self.writer.add_scalar(
#             'Total_Misclassifications',
#             data['Total_Misclassifications'],
#             n_iter
#         )

#     def log_eval_reverse(self, data, n_iter):
#         self.writer.add_scalar('Loss/Eval', data['loss_eval'], n_iter)

#     def log_custom_reverse_kpi(self, kpi, data, n_iter):
#         self.writer.add_scalar('Custom/' + kpi, data, n_iter)

#     def close(self):
#         self.writer.close()


if __name__ == '__main__':
    l = Logger(
        'model_1',
        {'optimizer': 'Adam'}
    )
    l.append_loss(1)
    l.append_loss(2)
    l.append_r2(50)
    l.append_r2(60)
    l.save_log()
