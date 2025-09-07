# Active Contour Model Parameters

# The number of points to initialize the snake with
N_POINTS = 100

# The alpha parameter, controlling the elasticity of the snake.
# Higher values make the snake more resistant to stretching.
ALPHA = 0.1

# The beta parameter, controlling the stiffness of the snake.
# Higher values make the snake more resistant to bending.
BETA = 0.1

# The gamma parameter, which is the step size for each iteration.
# It controls how far the snake moves in each step.
GAMMA = 0.1

# The weight of the image energy term.
# Higher values make the snake more attracted to edges.
W_EDGE = 1.0

# The number of iterations to run the algorithm.
N_ITERATIONS = 500

# The size of the neighborhood to search for the minimum energy.
# The search is performed in a (2 * N_SEARCH + 1) x (2 * N_SEARCH + 1) window.
N_SEARCH = 1
