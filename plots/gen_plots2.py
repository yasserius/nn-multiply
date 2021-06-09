import os
import glob
import json

import matplotlib.pyplot as plt

def load_json(f):
    with open(f, 'r') as json_f:
        return json.loads(json_f.read())

def get_json_logs(model_name):
    files = []
    for f in glob.glob(f"../runs/*-{model_name}/*.json"):
        files.append(load_json(f))
    return files

def plot_graph(y):
    fig = plt.figure(figsize=(15,10))
    plt.plot(y)
    plt.savefig('fig.png')

if __name__ == '__main__':
    fs = get_json_logs('2')
    f = fs[0]
    plot_graph(f['r2'])