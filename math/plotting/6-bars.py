#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

# Complete code for plotting a stacked bar graph
fig, ax = plt.subplots()

colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
fruits = ['Apples', 'Bananas', 'Oranges', 'Peaches']
people = ['Farrah', 'Fred', 'Felicia']

bottom = np.zeros(len(people))

for i, fruit_row in enumerate(fruit):
    ax.bar(people, fruit_row, color=colors[i], bottom=bottom, label=fruits[i])
    bottom += fruit_row

plt.xlabel('People')
plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.legend()
plt.ylim(0, 80)
plt.yticks(np.arange(0, 81, 10))

# Display the plot
plt.show()