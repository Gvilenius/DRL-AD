from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt
import json
import numpy as np
from scipy import interpolate
with open("results/DDPG", "r") as f:
    DDPG = json.load(f)

with open("results/TD3", "r") as f:
    TD3 = json.load(f)

with open("results/SAC", "r") as f:
    SAC = json.load(f)

def plot(a, method, label, smooth_step=1):
    x = []
    y = []
    for item in (a[method]):
        x.append(item[0] * 100)
        y.append(item[1])
    

    for i in range(len(y)-1, smooth_step-2, -1):
        for j in range(i-1, i-smooth_step, -1):
            y[i] += y[j]
        y[i] /= smooth_step
    x = x[smooth_step-1:]
    y = y[smooth_step-1:]
    plt.plot(x, y, label=label)

algo = [TD3, DDPG, SAC]
# method = ["fgsm", "i-fgsm", "pgd", "random"]
method = ["fgsm", "pgd"]
plt.xlabel("epsilon x 10^(-2)")
plt.ylabel("rewards")
for m in method:
    plot(DDPG, m, "DDPG-" + m)
    # plot(TD3, m, "TD3-" + m)
    # plot(SAC, m, "SAC-" + m)

ax=plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(0.1))
plt.legend()
plt.savefig("results/TD3.png")
plt.show()