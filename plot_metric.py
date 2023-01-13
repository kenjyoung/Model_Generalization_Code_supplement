import argparse
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir
import itertools
import matplotlib.patches as mpatches
import jax.numpy as jnp
from jax import vmap, jit
from functools import partial

convolve = jit(vmap(partial(jnp.convolve, mode='valid'), in_axes=(1, None)))

palette = itertools.cycle(sns.color_palette())

parser = argparse.ArgumentParser()
parser.add_argument("--data", "-d", type=str, nargs='+')
parser.add_argument("--window", "-w", type=int, default=1)
parser.add_argument("--frequency", "-f", type=int, default=1)
parser.add_argument("--metric", "-m", type=str, default="reward_rates")
args = parser.parse_args()

handles = []

for d in args.data:
	color = next(palette)
	values = None
	with open(d, 'rb') as f:
		data = pkl.load(f)["metrics"]
	values = np.asarray(data[args.metric])[::args.frequency,:]
	times = jnp.transpose(np.asarray([data["eval_times"][::args.frequency]]*values.shape[1]))

	values = convolve(values, np.ones(args.window)/args.window)
	times = convolve(times, np.ones(args.window)/args.window)

	handles+=[mpatches.Patch(color=color, label=d)]

	values = values.flatten()
	times = times.flatten()
	data_frame = pd.DataFrame({args.metric:values,"time":times})
	sns.lineplot(x="time", y=args.metric, data=data_frame)
plt.legend(handles=handles)
plt.grid()
plt.show()
