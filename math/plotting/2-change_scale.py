#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

# Complete code for plotting x ↦ y as a line graph
plt.plot(x, y, 'b-')  # 'b-' specifies a blue line
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title("Exponential Decay of C-14")
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.xlim(0, 28650)  # Set the x-axis limits

# Display the plot
plt.show()
