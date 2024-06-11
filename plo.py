import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Define the grid size
grid_size = (10, 10)

# Initialize the grid for the number of transitions
transitions = np.full(grid_size, 4, dtype=int)  # Default to 4 transitions

# Define wall positions (blocked positions)
blocked_positions = [
    (0, 4), (0, 5), (0, 6), (0, 7), (1, 6), (1, 7), (2, 5),
    (2, 6), (3, 5), (3, 6), (4, 4), (4, 5), (5, 4), (5, 5),
    (9,5), (9,7), (8,6), (6,0)
]

# Update transitions for blocked positions
for row, col in blocked_positions:
    transitions[row, col] = 5  # Use 5 to indicate a wall

# Adjust transitions based on neighboring walls
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        if transitions[i, j] != 5:
            walls_count = 0
            # Check the four possible directions
            if i > 0 and transitions[i - 1, j] == 5:  # Up
                walls_count += 1
            if i < grid_size[0] - 1 and transitions[i + 1, j] == 5:  # Down
                walls_count += 1
            if j > 0 and transitions[i, j - 1] == 5:  # Left
                walls_count += 1
            if j < grid_size[1] - 1 and transitions[i, j + 1] == 5:  # Right
                walls_count += 1

            # Adjust for edges and corners of the grid
            if i == 0 or i == grid_size[0] - 1:
                walls_count += 1
            if j == 0 or j == grid_size[1] - 1:
                walls_count += 1

            # Adjust the number of transitions based on the count of neighboring walls
            if walls_count == 2:
                transitions[i, j] = 3  # 3 transitions (corner)
            elif walls_count == 3:
                transitions[i, j] = 2  # 2 transitions (edge)
            elif walls_count >= 4:
                transitions[i, j] = 1  # 1 transition (surrounded by walls)

# Initialize the plot
fig, ax = plt.subplots()

# Mask the walls for the color mapping
masked_transitions = np.ma.masked_equal(transitions, 5)

# Create a custom colormap using RGB values
cmap = mcolors.LinearSegmentedColormap.from_list("", [
    (1, 1, 1),  # White
    (0.85, 0.85, 0.85),  # Light grey
    (0.7, 0.7, 0.7),  # Grey
    (0.5, 0.5, 0.5),  # Dark grey
    (0.25, 0.25, 0.25)   # Dim grey
])
norm = mcolors.Normalize(vmin=0, vmax=4)

# Plot the grid
for (i, j), value in np.ndenumerate(transitions):
    if value == 5:
        color = 'black'
    else:
        color = cmap(norm(value))
    ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color))

# Set the limits and grid
ax.set_xlim(0, grid_size[1])
ax.set_ylim(0, grid_size[0])
ax.set_xticks(np.arange(0, grid_size[1] + 1, 1))
ax.set_yticks(np.arange(0, grid_size[0] + 1, 1))
ax.grid(which='both', color='black')

# Hide the axes
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.tick_params(left=False, bottom=False)

# Invert Y-axis to match typical matrix layout
plt.gca().invert_yaxis()

# Add color bar (legend) for non-wall cells
sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=1, vmax=4))
sm.set_array(masked_transitions)
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Number of Transitions')

# Set the format of the color bar ticks to integers and limit to 1-4
cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
cbar.set_ticks([1, 2, 3, 4])

plt.show()
