# agent.py
import random
import math
from collections import deque
import heapq
 
 
class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""
 
    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']
 
    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept['agent_pos']
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)
 
 
# LAB 3/4: Goal-Based / Planning Agent using uninformed search (BFS, DFS, UCS)
# and informed search (A*) with heuristics.
class SearchAgent:
    """A Goal-Based Agent that uses the exposed World Model (grid_size, walls,
    all_food from get_percept) to plan a complete sequence of actions to the
    nearest food pellet BEFORE acting, instead of reacting one step at a time.
 
    BFS/DFS/UCS search directly over states of the form (x, y, facing), since
    those algorithms were built in LAB 3 around the environment's turn-based
    action model ('turn_left', 'turn_right', 'move_forward', 'suck').
 
    A* (LAB 4) follows the lab sheet exactly and searches over plain (x, y)
    positions using Up/Down/Left/Right neighbour expansion. Its resulting
    directional path is then converted into turn_left/turn_right/move_forward
    actions so it can still be executed by VisualGridHuntGame.execute_action.
    """
 
    def __init__(self):
        self.plan = []                 # LAB 3 - Step 1.3: the current sequence of planned actions
        self.active_algo = 'AStar'     # 'BFS', 'DFS', 'UCS', or 'AStar' (LAB 4) - change this to compare strategies
        self.heuristic_type = 'manhattan'  # LAB 4: 'manhattan' or 'euclidean' - used only when active_algo == 'AStar'
 
    # ---------- movement helpers (mirrors VisualGridHuntGame's facing model) ----------
    def _delta(self, facing):
        return {'Up': (0, 1), 'Down': (0, -1), 'Left': (-1, 0), 'Right': (1, 0)}[facing]
 
    def _turn(self, facing, direction):
        order = ['Up', 'Right', 'Down', 'Left']
        idx = order.index(facing)
        if direction == 'left':
            return order[(idx - 1) % 4]
        return order[(idx + 1) % 4]
 
    def _successors(self, state, walls, width, height):
        """Given a state (x, y, facing), return a list of (action, next_state, step_cost)."""
        x, y, facing = state
        successors = []
 
        # move_forward: only valid if the cell ahead is inside the grid and not a wall
        dx, dy = self._delta(facing)
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in walls:
            successors.append(('move_forward', (nx, ny, facing), 1))
 
        # turn_left / turn_right: always valid, cost 1, position stays the same
        successors.append(('turn_left', (x, y, self._turn(facing, 'left')), 1))
        successors.append(('turn_right', (x, y, self._turn(facing, 'right')), 1))
 
        return successors
 
    def _closest_food(self, start_pos, all_food):
        """Pick the nearest food pellet using Manhattan distance, so the agent
        always plans towards a single, well-defined goal state."""
        sx, sy = start_pos
        return min(all_food, key=lambda f: abs(f[0] - sx) + abs(f[1] - sy))
 
    def _directions_to_turn_actions(self, directions, start_facing):
        """Converts a list of ['Up','Down','Left','Right'] moves (as produced by
        astar_search, which reasons over plain grid directions per the lab sheet)
        into the turn_left / turn_right / move_forward sequence the environment
        actually understands, given the agent's current facing direction."""
        actions = []
        facing = start_facing
        for direction in directions:
            while facing != direction:
                # Rotate right until facing matches the desired direction (at most 3 turns)
                facing = self._turn(facing, 'right')
                actions.append('turn_right')
            actions.append('move_forward')
        return actions
 
    # ---------- LAB 4 - Step 1.1: heuristic functions ----------
    def manhattan_distance(self, pos, goal):
        """h(n) = |x1 - x2| + |y1 - y2|  -  sum of horizontal and vertical distance."""
        x1, y1 = pos
        x2, y2 = goal
        return abs(x1 - x2) + abs(y1 - y2)
 
    def euclidean_distance(self, pos, goal):
        """h(n) = sqrt((x1 - x2)^2 + (y1 - y2)^2)  -  straight-line distance."""
        x1, y1 = pos
        x2, y2 = goal
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
 
    # ---------- LAB 3 - Step 1.2: the three uninformed search strategies ----------
    def bfs_search(self, start_state, goal_pos, walls, width, height):
        """Breadth-First Search: FIFO queue -> explores the shallowest nodes first."""
        frontier = deque([(start_state, [])])
        reached = {start_state}
 
        while frontier:
            state, path = frontier.popleft()          # FIFO: pop from the front
            if (state[0], state[1]) == goal_pos:
                return path
 
            for action, next_state, cost in self._successors(state, walls, width, height):
                if next_state not in reached:          # reached set -> Graph Search, not Tree Search
                    reached.add(next_state)
                    frontier.append((next_state, path + [action]))
 
        return []  # no path found
 
    def dfs_search(self, start_state, goal_pos, walls, width, height):
        """Depth-First Search: LIFO stack -> explores the deepest nodes first."""
        frontier = [(start_state, [])]
        reached = {start_state}
 
        while frontier:
            state, path = frontier.pop()               # LIFO: pop from the end
            if (state[0], state[1]) == goal_pos:
                return path
 
            for action, next_state, cost in self._successors(state, walls, width, height):
                if next_state not in reached:
                    reached.add(next_state)
                    frontier.append((next_state, path + [action]))
 
        return []  # no path found
 
    def ucs_search(self, start_state, goal_pos, walls, width, height):
        """Uniform-Cost Search: Priority Queue ordered by total path cost g(n)."""
        counter = 0  # tie-breaker so heapq never tries to compare states directly
        frontier = [(0, counter, start_state, [])]      # (g(n), tie-breaker, state, path)
        reached = {start_state: 0}
 
        while frontier:
            cost_so_far, _, state, path = heapq.heappop(frontier)  # lowest g(n) first
            if (state[0], state[1]) == goal_pos:
                return path
 
            for action, next_state, step_cost in self._successors(state, walls, width, height):
                new_cost = cost_so_far + step_cost
                if next_state not in reached or new_cost < reached[next_state]:
                    reached[next_state] = new_cost
                    counter += 1
                    heapq.heappush(frontier, (new_cost, counter, next_state, path + [action]))
 
        return []  # no path found
 
    # ---------- LAB 4 - Step 1.2: A* Search (informed search) ----------
    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan'):
        """A* Search: Priority Queue ordered by f(n) = g(n) + h(n).
        Unlike UCS (which only considers g(n)), A* also uses a heuristic h(n)
        estimating the remaining distance to the goal, which lets it explore
        far fewer nodes than the uninformed strategies.
 
        Searches directly over (x, y) grid positions using Up/Down/Left/Right
        moves, matching the lab sheet's 4-way adjacent-cell expansion exactly.
        Returns a list of directions (e.g. ['Up', 'Up', 'Right']); use
        _directions_to_turn_actions() to convert this into actions the
        environment can execute.
        """
        width, height = grid_size
 
        def heuristic(pos):
            if heuristic_type == 'euclidean':
                return self.euclidean_distance(pos, goal_pos)
            return self.manhattan_distance(pos, goal_pos)
 
        counter = 0  # tie-breaker so heapq never tries to compare tuples with equal (f, g) directly
        g_start = 0
        h_start = heuristic(start_pos)
        f_start = g_start + h_start
 
        # LAB 4 - Step 1.2.3: tuple format is (f_cost, g_cost, current_pos, path_taken)
        frontier = [(f_start, g_start, start_pos, [])]
        reached_states = set()
 
        while frontier:
            f_cost, g_cost, current_pos, path_taken = heapq.heappop(frontier)  # lowest f_cost first
 
            if current_pos == goal_pos:
                return path_taken
 
            if current_pos in reached_states:
                continue
            reached_states.add(current_pos)
 
            # LAB 4 - Step 1.2.5: expand the four adjacent cells (Up, Down, Left, Right)
            x, y = current_pos
            neighbors = [
                ('Up', (x, y + 1)),
                ('Down', (x, y - 1)),
                ('Left', (x - 1, y)),
                ('Right', (x + 1, y)),
            ]
 
            for direction, next_pos in neighbors:
                nx, ny = next_pos
                in_bounds = 0 <= nx < width and 0 <= ny < height
                if not in_bounds or next_pos in walls or next_pos in reached_states:
                    continue
 
                g_new = g_cost + 1
                h_new = heuristic(next_pos)
                f_new = g_new + h_new
                counter += 1
                heapq.heappush(frontier, (f_new, g_new, next_pos, path_taken + [direction]))
 
        return []  # no path found
 
    # ---------- LAB 3 - Step 1.3 / LAB 4 - Step 1.3: forming and executing the plan ----------
    def sense_and_act(self, percept):
        if not self.plan:
            if not percept['all_food']:
                return 'turn_left'  # nothing left to plan towards, just idle-turn
 
            start_pos = tuple(percept['agent_pos'])
            goal_pos = self._closest_food(start_pos, percept['all_food'])
            walls = set(tuple(w) for w in percept['walls'])
            width, height = percept['grid_size']
 
            if self.active_algo == 'BFS':
                start_state = (start_pos[0], start_pos[1], percept['facing'])
                self.plan = self.bfs_search(start_state, goal_pos, walls, width, height)
                self.plan.append('suck')
 
            elif self.active_algo == 'DFS':
                start_state = (start_pos[0], start_pos[1], percept['facing'])
                self.plan = self.dfs_search(start_state, goal_pos, walls, width, height)
                self.plan.append('suck')
 
            elif self.active_algo == 'UCS':
                start_state = (start_pos[0], start_pos[1], percept['facing'])
                self.plan = self.ucs_search(start_state, goal_pos, walls, width, height)
                self.plan.append('suck')
 
            elif self.active_algo == 'AStar':
                # LAB 4 - Step 1.3: A* returns a list of directions (Up/Down/Left/Right),
                # so convert it into turn_left/turn_right/move_forward before storing the plan.
                directions = self.astar_search(start_pos, goal_pos, walls, (width, height),
                                               heuristic_type=self.heuristic_type)
                self.plan = self._directions_to_turn_actions(directions, percept['facing'])
                self.plan.append('suck')
 
            if not self.plan:
                return 'turn_left'  # safety fallback if no path was found
 
        return self.plan.pop(0)
 
 
if __name__ == "__main__":
    # LAB 4 - Step 1.1: Testing Checkpoint
    # Verify manhattan_distance((0,0), (3,4)) == 7 and euclidean_distance((0,0), (3,4)) == 5.0
    test_agent = SearchAgent()
    print("Manhattan distance (0,0) -> (3,4):", test_agent.manhattan_distance((0, 0), (3, 4)))
    print("Euclidean distance (0,0) -> (3,4):", test_agent.euclidean_distance((0, 0), (3, 4)))
 