import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
pbandit_path = 'tb_logs_reduced/gat_flickr_poisson-bandit_steps_1000_bs_2000_layers_2_lr_0.001_eta_0.001_10.csv'
pladies_path = 'tb_logs_reduced/gat_flickr_poisson-ladies_steps_1000_bs_2000_layers_2_lr_0.001_eta_0.001_10.csv'
df_bandit = pd.read_csv(pbandit_path, header=[0, 1])
df_ladies = pd.read_csv(pladies_path, header=[0, 1])

y_pbandit = df_bandit['val_acc'].bfill()['mean']
std_pbandit = df_bandit['val_acc'].bfill()['std']

y_pladies = df_ladies['val_acc'].bfill()['mean']
std_pladies = df_ladies['val_acc'].bfill()['std']

plt.xlabel('Step')
plt.ylabel('Average Validation Accuracy')
plt.plot(y_pbandit, label='Poisson-Bandit')
plt.plot(y_pladies, label='Poisson-Ladies')
plt.fill_between(range(len(y_pbandit)), y_pbandit+std_pbandit, y_pbandit-std_pbandit, alpha=0.2)
plt.fill_between(range(len(y_pladies)), y_pladies+std_pladies, y_pladies-std_pladies, alpha=0.2)
plt.legend()
plt.savefig('plot_result_flickr_dataset.png')

