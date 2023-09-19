#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

# Complete code for plotting y as a solid red line with x-axis ranging from 0 to 10
x = np.arange(0, 11)

plt.plot(x, y, 'r-')  # 'r-' specifies a solid red line
plt.xlabel('x')
plt.ylabel('y')
plt.title('y = x^3')
plt.xlim(0, 10)  # Set the x-axis limits
plt.grid(True)   # Add a grid for better readability

# Display the plot
plt.show()
