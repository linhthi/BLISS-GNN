import glob
import os
import matplotlib.pyplot as plt
import tensorboard_reducer as tbr
import pandas as pd
k_runs = 10
# print how many tb logs are there to get the mean, max, min, std on.
logdir = "tb_logs"
subdir = "gat_flickr_poisson-bandit_steps_1000_bs_2000_layers_2_lr_0.001_eta_0.0001"
input_event_dirs = sorted(glob.glob(f"{os.path.join(logdir, subdir)}/*"))
print(f"Found {len(input_event_dirs)}")

events_out_dir = f"{logdir}_reduced/{subdir}_{len(input_event_dirs)}"
csv_out_path = f"{logdir}_reduced/{subdir}_{len(input_event_dirs)}.csv"
overwrite = True
reduce_ops = ("mean", "min", "max", "std")

events_dict = tbr.load_tb_events(
    input_event_dirs, verbose=True, handle_dup_steps='mean')

reduced_events = tbr.reduce_events(events_dict, reduce_ops, verbose=True)

for op in reduce_ops:
    print(f"Writing '{op}' reduction to '{events_out_dir}-{op}'")

tbr.write_tb_events(reduced_events, events_out_dir, overwrite, verbose=True)
print(f"Writing results to '{csv_out_path}'")
tbr.write_data_file(reduced_events, csv_out_path, overwrite, verbose=True)
print("✓ Reduction complete")

df_reduced_results = pd.read_csv(csv_out_path, header=[0, 1])
y = df_reduced_results['val_acc'].bfill()['mean']
std = df_reduced_results['val_acc'].bfill()['std']

plt.xlabel('Step')
plt.ylabel('Average Validation Accuracy')
plt.plot(y)
plt.fill_between(range(len(y)), y+std, y-std, alpha=0.2)
# Show the plot
plt.grid()
plt.savefig(f"{logdir}_reduced/{subdir}_{len(input_event_dirs)}.png")
# plt.show(block=True)