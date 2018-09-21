# -*- coding:utf-8 -*-
# Imports
import numpy as np
import matplotlib.pyplot as plt

# Create a new figure of size 8x6 points, using 100 dots per inch
plt.figure(figsize=(8,6), dpi=80)

# Create a new subplot from a grid of 1x1
ax = plt.subplot(111)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))


X = np.linspace(-np.pi, np.pi, 256,endpoint=True)
C,S = np.cos(X), np.sin(X)

# Plot cosine using blue color with a continuous line of width 1 (pixels)
plt.plot(X, C, color="blue", linewidth=2.5, linestyle="-", label='cosine')

# Plot sine using green color with a continuous line of width 1 (pixels)
plt.plot(X, S, color="green", linewidth=2.5, linestyle="-", label='sine')

# Set x limits
plt.xlim(np.min(X)*1.1,np.max(X)*1.1)

# Set x ticks  range from -4 to 4 with 9 points
# plt.xticks(np.linspace(-4,4,9,endpoint=True))
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])

# Set y limits
plt.ylim(np.min(C)*1.1, np.max(C)*1.1)

# Set y ticks  range from -4 to 4 with 9 points
# plt.yticks(np.linspace(-1,1,5,endpoint=True))
plt.yticks([-1, +1], [r'$-1$', r'$+1$'])

# Save figure using 72 dots per inch
# savefig("../figures/exercice_2.png",dpi=72)



plt.legend(loc='upper left', frameon=False)  # 设置图标的位置


for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(16)
    label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.15))

# Show result on screen
plt.show()
