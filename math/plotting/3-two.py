#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

# Complete code for plotting x ↦ y1 and x ↦ y2 as line graphs
plt.plot(x, y1, 'r--', label='C-14')  # 'r--' specifies a dashed red line
plt.plot(x, y2, 'g-', label='Ra-226')  # 'g-' specifies a solid green line
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title("Exponential Decay of Radioactive Elements")
plt.xlim(0, 20000)  # Set the x-axis limits
plt.ylim(0, 1)      # Set the y-axis limits
plt.legend(loc='upper right')  # Add a legend in the upper right corner

# Display the plot
plt.show()
