# agent.py
import random
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
 
 
# LAB 3: Goal-Based / Planning Agent using uninformed search (BFS, DFS, UCS)
class SearchAgent:
    """A Goal-Based Agent that uses the exposed World Model (grid_size, walls,
    all_food from get_percept) to plan a complete sequence of actions to the
    nearest food pellet BEFORE acting, instead of reacting one step at a time.
 
    Search is performed over states of the form (x, y, facing), because the
    environment only accepts 'turn_left', 'turn_right', 'move_forward', and
    'suck' as valid actions (see VisualGridHuntGame.execute_action).
    """
 
    def __init__(self):
        self.plan = []                 # LAB 3 - Step 1.3: the current sequence of planned actions
        self.active_algo = 'BFS'       # LAB 3 - Step 1.3: 'BFS', 'DFS', or 'UCS' - change this to compare strategies
 
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
 
    # ---------- LAB 3 - Step 1.3: forming and executing the plan ----------
    def sense_and_act(self, percept):
        if not self.plan:
            if not percept['all_food']:
                return 'turn_left'  # nothing left to plan towards, just idle-turn
 
            start_pos = tuple(percept['agent_pos'])
            start_state = (start_pos[0], start_pos[1], percept['facing'])
            goal_pos = self._closest_food(start_pos, percept['all_food'])
            walls = set(tuple(w) for w in percept['walls'])
            width, height = percept['grid_size']
 
            if self.active_algo == 'BFS':
                self.plan = self.bfs_search(start_state, goal_pos, walls, width, height)
            elif self.active_algo == 'DFS':
                self.plan = self.dfs_search(start_state, goal_pos, walls, width, height)
            elif self.active_algo == 'UCS':
                self.plan = self.ucs_search(start_state, goal_pos, walls, width, height)
 
            self.plan.append('suck')  # once the agent reaches the food cell, eat it
 
            if not self.plan:
                return 'turn_left'  # safety fallback if no path was found
 
        return self.plan.pop(0)
 