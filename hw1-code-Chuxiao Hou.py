# First Solution
# Implement BFS
# check if a given state is valid
def is_valid_state(state):
    # get the number of missionaries and cannibals on both sides, and the boat's position
    (can_left, miss_left, _, can_right, miss_right) = state
    # check if both sides meet the safety condition: the number of missionaries is 0 or greater than or equal to the number of cannibals
    return (miss_left == 0 or miss_left >= can_left) and (miss_right == 0 or miss_right >= can_right)


# generate all valid successor states for a given state
def generate_successors(state):
    # get detailed information about the current state
    (can_left, miss_left, boat, can_right, miss_right) = state
    # an empty list for successor states
    successors = []
    # all possible move operations
    actions = [(2, 0), (1, 1), (0, 2), (1, 0), (0, 1)]

    # iterate through all possible move operations
    for (can, miss) in actions:
        if boat:  # the boat is on the left side
            # moving the boat to the right side
            new_state = (can_left - can, miss_left - miss, not boat, can_right + can, miss_right + miss)
        else:  # the boat is on the right side
            # moving the boat to the left side
            new_state = (can_left + can, miss_left + miss, not boat, can_right - can, miss_right - miss)

        # check if the new state is valid and ensure no negative numbers of people
        if is_valid_state(new_state) and min(new_state[:2]) >= 0 and min(new_state[3:5]) >= 0:
            # add it to the list of successor states if the new state is valid,
            successors.append(new_state)

    # return the list of all valid successor states
    return successors


# define the BFS function with initial and goal states as parameters
def bfs(initial_state, goal_state):
    # initialize the queue with the initial state and an empty path
    queue = [(initial_state, [])]
    # initialize a set to keep track of seen states to avoid revisiting
    seen = set()
    # initialize a counter to keep track of the number of visited nodes
    visited_nodes_count = 0

    # looping as long as there are states in the queue
    while queue:
        # pop the first state from the queue and increment the visited nodes counter
        current_state, path = queue.pop(0)
        visited_nodes_count += 1  # increment the counter for each visited node

        # check if the current state is the goal state
        if current_state == goal_state:
            # print the number of visited nodes and return the path including the current (goal) state
            print(f"Visited nodes: {visited_nodes_count}")
            return path + [current_state]

        # if the current state has been seen before, skip it to avoid cycles
        if current_state in seen:
            continue
        # add the current state to the set of seen states
        seen.add(current_state)

        # generate all valid successors of the current state
        for successor in generate_successors(current_state):
            # if the successor has not been seen, add it to the queue with the updated path
            if successor not in seen:
                queue.append((successor, path + [current_state]))

    # print the visited nodes count and return None if the loop ends without finding the goal state
    print(f"Visited nodes: {visited_nodes_count}")
    return None


# print the solution path
def print_solution(solution_path):
    # a solution path was found
    if solution_path:
        # iterate and print each step in the path
        for step in solution_path:
            print(step)
    else:
        # Print No solution found if no solution was found
        print("No solution found.")


# define the initial and goal states
initial_state = (3, 3, True, 0, 0)
goal_state = (0, 0, False, 3, 3)

# search for a solution path using the BFS algorithm
solution_path = bfs(initial_state, goal_state)

# print the solution path
print_solution(solution_path)


# Second Solution
# Implement DFS
# check if a given state is valid
def is_valid_state(state):
    # get the numbers of missionaries and cannibals on both sides, and the boat's position
    (can_left, miss_left, _, can_right, miss_right) = state
    # the number of missionaries is less than the number of cannibals and the number of missionaries is not zero
    return (miss_left == 0 or miss_left >= can_left) and (miss_right == 0 or miss_right >= can_right)


# generate all valid successor states from a given state
def generate_successors(state):
    # get the current state details
    (can_left, miss_left, boat, can_right, miss_right) = state
    # initialize a list to hold successor states
    successors = []
    # all possible actions (moving people across the river)
    actions = [(2, 0), (1, 1), (0, 2), (1, 0), (0, 1)]

    # iterate over all possible actions
    for (can, miss) in actions:
        if boat:  # the boat is on the left side
            # generate a new state based on the action, moving the boat to the right side
            new_state = (can_left - can, miss_left - miss, not boat, can_right + can, miss_right + miss)
        else:  # the boat is on the right side
            # generate a new state based on the action, moving the boat to the left side
            new_state = (can_left + can, miss_left + miss, not boat, can_right - can, miss_right - miss)

        # add it to the list of successors if the new state is valid
        if is_valid_state(new_state) and min(new_state[:2]) >= 0 and min(new_state[3:5]) >= 0:
            successors.append(new_state)

    # return the list of successor states
    return successors


# define DFS function to find a path from the initial state to the goal state
def dfs(initial_state, goal_state):
    # initialize a stack with the initial state and an empty path
    stack = [(initial_state, [])]
    # initialize a set to keep track of visited states to avoid revisiting them
    seen = set()
    # initialize a counter to track the number of nodes visited during the search
    visited_nodes_count = 0

    # continue the search as long as there are states in the stack to explore
    while stack:
        # pop the last state from the stack to explore it
        current_state, path = stack.pop()
        # process the state if it has not been visited before
        if current_state not in seen:
            # increment the visited nodes counter as we are exploring a new state
            visited_nodes_count += 1

            # check if the current state is the goal state
            if current_state == goal_state:
                # print the number of visited nodes and return the path to the goal if the goal state is found
                print(f"Visited nodes: {visited_nodes_count}")
                return path + [current_state]

            # add the current state to the set of visited states
            seen.add(current_state)

            # generate all valid successor states from the current state
            for successor in generate_successors(current_state):
                # for each successor that has not been visited, add it to the stack with the updated path
                if successor not in seen:
                    stack.append((successor, path + [current_state]))

    # print the number of visited nodes and return None if the search completes without finding the goal state
    print(f"Visited nodes: {visited_nodes_count}")
    return None


# print the solution path
def print_solution(solution_path):
    # if a solution path exists
    if solution_path:
        # print each step in the solution path
        for step in solution_path:
            print(step)
    else:
        # print No solution found if no solution was found
        print("No solution found.")


# define the initial and goal states
initial_state = (3, 3, True, 0, 0)
goal_state = (0, 0, False, 3, 3)

# execute the DFS search
solution_path_dfs = dfs(initial_state, goal_state)

# print the solution
print_solution(solution_path_dfs)


# Third Solution
# Implement IDS
# check if a given state is valid
def is_valid_state(state):
    # get the numbers of missionaries and cannibals on both sides, and the boat's position
    (can_left, miss_left, _, can_right, miss_right) = state
    # return True if missionaries are not outnumbered by cannibals on both sides, else False
    return (miss_left == 0 or miss_left >= can_left) and (miss_right == 0 or miss_right >= can_right)


# generate all valid successor states from a given state
def generate_successors(state):
    # get the current state details
    (can_left, miss_left, boat, can_right, miss_right) = state
    # initialize a list to hold successor states
    successors = []
    # all possible actions
    actions = [(2, 0), (1, 1), (0, 2), (1, 0), (0, 1)]

    # all possible actions
    for (can, miss) in actions:
        if boat:  # the boat is on the left side
            # generate a new state based on the action, moving the boat to the right side
            new_state = (can_left - can, miss_left - miss, not boat, can_right + can, miss_right + miss)
        else:  # the boat is on the right side
            # generate a new state based on the action, moving the boat to the left side
            new_state = (can_left + can, miss_left + miss, not boat, can_right - can, miss_right - miss)

        # add it to the list of successors if the new state is valid and does not result in negative numbers
        if is_valid_state(new_state) and min(new_state[:2]) >= 0 and min(new_state[3:5]) >= 0:
            successors.append(new_state)

    # return the list of successor states
    return successors


# initialize a global counter to track the number of visited nodes across all depths
visited_nodes_count = 0


# DLS function
def dls(state, goal_state, depth):
    global visited_nodes_count
    visited_nodes_count += 1  # increment the counter as each state is visited

    # check if the current depth is 0 and the state is the goal state
    if depth == 0 and state == goal_state:
        return [state]
    # explore further if depth is greater than 0
    if depth > 0:
        for successor in generate_successors(state):
            # recursively call dls on each successor with decreased depth
            path = dls(successor, goal_state, depth - 1)
            # return it if a path to the goal is found
            if path:
                return [state] + path
    # return None if no path is found
    return None


# IDS function
def ids(initial_state, goal_state, max_depth):
    for depth in range(max_depth):
        # reset the visited nodes count for each depth level
        global visited_nodes_count
        visited_nodes_count = 0

        # call Depth-Limited Search with increasing depth limits
        result = dls(initial_state, goal_state, depth)
        # print the number of visited nodes and return the result if a result is found
        if result:
            print(f"Depth: {depth}, Visited Nodes: {visited_nodes_count}")
            return result
    # return None if no solution is found within the max depth
    return None


# print the solution path
def print_solution(solution_path):
    # if a solution path exists
    if solution_path:
        # print each step in the solution path
        for step in solution_path:
            print(step)
    else:
        # print No solution found if no solution was found
        print("No solution found.")


# set the initial and goal states
initial_state = (3, 3, True, 0, 0)
goal_state = (0, 0, False, 3, 3)

# set a maximum depth limit
max_depth = 20

# execute Iterative Deepening Search
solution_path_ids = ids(initial_state, goal_state, max_depth)

# print the solution
print_solution(solution_path_ids)

# Forth Solution
# Implement A*
from queue import PriorityQueue


# estimating the cost to reach the goal from the current state
def heuristic(state):
    # based on the number of people (missionaries and cannibals) left on the left bank.
    # each boat trip can move up to two people towards the goal.
    can_left, miss_left, _, _, _ = state
    return (can_left + miss_left) / 2


# check if the current state is valid according to the problem's rules
def is_valid_state(state):
    can_left, miss_left, _, can_right, miss_right = state
    # conditions to ensure missionaries are not outnumbered by cannibals on either side
    if can_left < miss_left and can_left > 0: return False
    if can_right < miss_right and can_right > 0: return False
    # the number of people does not become negative
    return can_left >= 0 and miss_left >= 0 and can_right >= 0 and miss_right >= 0


# generate all possible valid successor states from the current state
def generate_successors(state):
    successors = []
    can_left, miss_left, boat, can_right, miss_right = state
    # possible actions
    actions = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]

    for can, miss in actions:
        # apply each action based on the boat's current position
        if boat:  # boat on the left side, move to the right
            new_state = (can_left - can, miss_left - miss, 0, can_right + can, miss_right + miss)
        else:  # boat on the right side, move to the left
            new_state = (can_left + can, miss_left + miss, 1, can_right - can, miss_right - miss)
        # add the state to successors if it's valid
        if is_valid_state(new_state):
            successors.append(new_state)

    return successors


# A* search algorithm
def a_star_search(initial_state, goal_state):
    visited_nodes_count = 0  # initialize counter for visited nodes
    frontier = PriorityQueue()  # priority queue for states to explore
    frontier.put((0, initial_state))  # start with the initial state
    came_from = {initial_state: None}  # track path taken to reach each state
    cost_so_far = {initial_state: 0}  # track the cost to reach each state

    # Loop until no more states to explore
    while not frontier.empty():
        current = frontier.get()[1]  # dequeue the state with the lowest cost estimate
        visited_nodes_count += 1  # increment the counter for each state visited

        # check if the current state is the goal
        if current == goal_state:
            print(f"Visited Nodes: {visited_nodes_count}")  # the count of visited nodes
            return came_from, cost_so_far

        # successors of the current state
        for next_state in generate_successors(current):
            new_cost = cost_so_far[current] + 1  # Assume each move costs 1
            # update cost and add to frontier if this state is new or offers a cheaper path
            if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                cost_so_far[next_state] = new_cost
                priority = new_cost + heuristic(next_state)  # priority with heuristic
                frontier.put((priority, next_state))  # add new state to the frontier
                came_from[next_state] = current  # track how we got to this state

    print(f"Visited Nodes: {visited_nodes_count}")  # visited nodes if goal not reached
    return None, None


# the path from the initial state to the goal state
def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:  # the path from goal to start
        path.append(current)
        current = came_from.get(current, None)
    path.append(start)  # add the start state to the path
    path.reverse()  # reverse the path to get it in the correct order
    return path


# the initial state and the goal state
initial_state = (3, 3, 1, 0, 0)
goal_state = (0, 0, 0, 3, 3)

# find the solution path
came_from, cost_so_far = a_star_search(initial_state, goal_state)

# reconstruct and print the solution path if a solution was found
if came_from and cost_so_far:
    solution_path = reconstruct_path(came_from, initial_state, goal_state)
    print("Solution Path:", solution_path)
    # the cost of reaching the goal state
    print("Solution Cost:", cost_so_far[goal_state])
else:
    print("No solution found.")


