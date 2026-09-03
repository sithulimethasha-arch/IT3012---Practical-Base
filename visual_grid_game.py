import random
import tkinter as tk


class VisualGridHuntGame:
    """A flexible Pacman-style grid environment with support for configurable opponents and larger scales."""

    def __init__(self, width=10, height=10, num_food=10, num_opponents=2, custom_walls=None):
        self.width = width
        self.height = height
        self.agent_pos = [0, 0]  # Starting position (x, y)
        self.facing = 'Up'  # LAB 2: the direction the agent is currently facing

        if custom_walls is not None:
            self.walls = set(custom_walls)
        else:
            # Generate some default scattered walls for a larger grid
            self.walls = {(2, 2), (2, 3), (5, 5), (6, 5), (3, 7)}

        # Dynamically generate random food positions avoiding walls and agent start
        self.food_positions = set()
        while len(self.food_positions) < num_food:
            fx = random.randint(0, self.width - 1)
            fy = random.randint(0, self.height - 1)
            pos_tuple = (fx, fy)
            if pos_tuple != (0, 0) and pos_tuple not in self.walls:
                self.food_positions.add(pos_tuple)

        # Generate adversarial opponents
        self.opponents = []
        while len(self.opponents) < num_opponents:
            ox = random.randint(0, self.width - 1)
            oy = random.randint(0, self.height - 1)
            op_pos = [ox, oy]
            if tuple(op_pos) != (0, 0) and tuple(op_pos) not in self.walls and tuple(op_pos) not in self.food_positions:
                self.opponents.append(op_pos)

        # Toxic traps - hidden hazards not revealed directly to the agent's sensors
        self.toxic_traps = set()
        while len(self.toxic_traps) < 5:
            tx = random.randint(0, self.width - 1)
            ty = random.randint(0, self.height - 1)
            trap_tuple = (tx, ty)
            if (trap_tuple != (0, 0)
                    and trap_tuple not in self.walls
                    and trap_tuple not in self.food_positions
                    and trap_tuple not in [tuple(op) for op in self.opponents]):
                self.toxic_traps.add(trap_tuple)

        self.score = 0
        self.steps = 0
        self.collision = False

    # LAB 2: helper to get the (dx, dy) offset for a facing direction
    def _delta(self, facing):
        return {'Up': (0, 1), 'Down': (0, -1), 'Left': (-1, 0), 'Right': (1, 0)}[facing]

    # LAB 2: helper to rotate the facing direction left or right
    def _turn(self, facing, direction):
        order = ['Up', 'Right', 'Down', 'Left']
        idx = order.index(facing)
        if direction == 'left':
            return order[(idx - 1) % 4]
        return order[(idx + 1) % 4]

    def get_percept(self) -> dict:
        # LAB 2 - Step 1.1: No longer returns exact global coordinates (agent_pos).
        # Instead, only local booleans based on the cell directly ahead of the agent.
        dx, dy = self._delta(self.facing)
        ax, ay = self.agent_pos
        nx, ny = ax + dx, ay + dy

        out_of_bounds = not (0 <= nx < self.width and 0 <= ny < self.height)
        wall_ahead = out_of_bounds or (nx, ny) in self.walls
        food_here = tuple(self.agent_pos) in self.food_positions
        toxin_here = tuple(self.agent_pos) in self.toxic_traps

        return {
            'wall_ahead': wall_ahead,
            'food_here': food_here,
            'smells_toxin': toxin_here,
            'collision': self.collision,
            'score': self.score,
            'remaining_food': len(self.food_positions)
        }

    def execute_action(self, action: str):
        self.steps += 1

        if action == 'turn_left':
            self.facing = self._turn(self.facing, 'left')

        elif action == 'turn_right':
            self.facing = self._turn(self.facing, 'right')

        elif action == 'suck':
            tuple_pos = tuple(self.agent_pos)
            if tuple_pos in self.food_positions:
                self.food_positions.remove(tuple_pos)
                self.score += 20

        elif action == 'move_forward':
            dx, dy = self._delta(self.facing)
            new_pos = [self.agent_pos[0] + dx, self.agent_pos[1] + dy]
            out_of_bounds = not (0 <= new_pos[0] < self.width and 0 <= new_pos[1] < self.height)
            if out_of_bounds or tuple(new_pos) in self.walls:
                self.score -= 5  # Bumped into a wall or the grid boundary
            else:
                self.agent_pos = new_pos

        # Toxic trap penalty applies wherever the agent currently stands
        tuple_pos = tuple(self.agent_pos)
        if tuple_pos in self.toxic_traps:
            self.score -= 15

        # Opponents move independently and randomly (Multi-Agent behaviour from Lab 01)
        for op in self.opponents:
            move = random.choice(['Up', 'Down', 'Left', 'Right', 'Stay'])
            if move == 'Up' and op[1] < self.height - 1:
                op[1] += 1
            elif move == 'Down' and op[1] > 0:
                op[1] -= 1
            elif move == 'Left' and op[0] > 0:
                op[0] -= 1
            elif move == 'Right' and op[0] < self.width - 1:
                op[0] += 1

            if op == self.agent_pos:
                self.score -= 50
                self.collision = True

    def is_done(self) -> bool:
        return len(self.food_positions) == 0 or self.steps >= 60 or self.collision


# LAB 2 - Step 1.2: Simple Reflex Agent
class SimpleReflexAgent:
    """A purely reactive agent using strict Condition-Action (IF-THEN) rules.
    It has NO memory of the past - every decision is based only on the current percept."""

    def sense_and_act(self, percept):
        if percept['food_here']:
            return 'suck'
        elif percept['wall_ahead']:
            return 'turn_left'
        else:
            return 'move_forward'


# LAB 2 - Step 1.3: Model-Based Agent
class ModelBasedAgent:
    """A reflex agent enhanced with an internal state. Since get_percept() no longer
    exposes the agent's true global position, this agent tracks its own position and
    facing direction internally (dead-reckoning) using the Transition Model, based on
    the actions it has taken."""

    def __init__(self):
        self.visited_cells = set()
        self.rel_pos = (0, 0)      # Internally-tracked relative position estimate
        self.rel_facing = 'Up'     # Internally-tracked facing direction
        self.visited_cells.add(self.rel_pos)
        self.last_action = None

    def _delta(self, facing):
        return {'Up': (0, 1), 'Down': (0, -1), 'Left': (-1, 0), 'Right': (1, 0)}[facing]

    def _turn(self, facing, direction):
        order = ['Up', 'Right', 'Down', 'Left']
        idx = order.index(facing)
        if direction == 'left':
            return order[(idx - 1) % 4]
        return order[(idx + 1) % 4]

    def sense_and_act(self, percept):
        # Update internal state (Transition Model) based on the LAST action taken
        if self.last_action == 'move_forward' and not percept['wall_ahead']:
            dx, dy = self._delta(self.rel_facing)
            self.rel_pos = (self.rel_pos[0] + dx, self.rel_pos[1] + dy)
            self.visited_cells.add(self.rel_pos)
        elif self.last_action == 'turn_left':
            self.rel_facing = self._turn(self.rel_facing, 'left')
        elif self.last_action == 'turn_right':
            self.rel_facing = self._turn(self.rel_facing, 'right')

        # Decide the next action using memory-aware rules
        if percept['food_here']:
            action = 'suck'
        elif percept['wall_ahead']:
            # Check whether turning left leads back to an already-visited cell
            left_facing = self._turn(self.rel_facing, 'left')
            dx, dy = self._delta(left_facing)
            left_cell = (self.rel_pos[0] + dx, self.rel_pos[1] + dy)
            if left_cell in self.visited_cells:
                action = 'turn_right'  # avoid re-entering a known loop
            else:
                action = 'turn_left'
        else:
            action = 'move_forward'

        self.last_action = action
        return action


class GridGameGUI:
    """Tkinter wrapper that dynamically scales cell sizes to keep larger grids on screen."""

    def __init__(self, root, width=10, height=10, num_food=12, num_opponents=2, walls=None, agent_type='model'):
        self.root = root
        self.root.title("IT3012 - Scalable Multi-Agent Grid Hunt")

        self.env = VisualGridHuntGame(width=width, height=height, num_food=num_food, num_opponents=num_opponents,
                                      custom_walls=walls)

        # LAB 2: choose which agent architecture drives the simulation
        # 'simple' = SimpleReflexAgent (will get stuck in loops)
        # 'model'  = ModelBasedAgent (remembers visited cells, escapes loops)
        if agent_type == 'simple':
            self.agent = SimpleReflexAgent()
        else:
            self.agent = ModelBasedAgent()

        # Dynamically calculate cell size so the total canvas fits nicely within a 600x600 window ceiling
        max_canvas_dim = 600
        self.cell_size = max(20, min(max_canvas_dim // self.env.width, max_canvas_dim // self.env.height))

        canvas_w = self.env.width * self.cell_size
        canvas_h = self.env.height * self.cell_size

        self.canvas = tk.Canvas(root, width=canvas_w, height=canvas_h, bg="white")
        self.canvas.pack()

        self.label = tk.Label(root, text="Score: 0 | Steps: 0", font=("Arial", 14))
        self.label.pack(pady=10)

        self.btn = tk.Button(root, text="Start Simulation", command=self.run_loop, font=("Arial", 12), bg="#000066",
                             fg="white")
        self.btn.pack(pady=5)

        self.draw_grid()

    def draw_grid(self):
        self.canvas.delete("all")

        for x in range(self.env.width):
            for y in range(self.env.height):
                x1 = x * self.cell_size
                y1 = (self.env.height - 1 - y) * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                color = "#f1f5f9" if (x, y) not in self.env.walls else "#64748b"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="#cbd5e1")

                if self.cell_size >= 40 and (x, y) in self.env.walls:
                    self.canvas.create_text(x1 + self.cell_size / 2, y1 + self.cell_size / 2, text="W", fill="white",
                                            font=("Arial", 8, "bold"))

        for fx, fy in self.env.food_positions:
            offset = self.cell_size * 0.25
            x1 = fx * self.cell_size + offset
            y1 = (self.env.height - 1 - fy) * self.cell_size + offset
            self.canvas.create_oval(x1, y1, x1 + self.cell_size * 0.5, y1 + self.cell_size * 0.5, fill="#f59e0b",
                                    outline="#d97706")

        for tx, ty in self.env.toxic_traps:
            x1 = tx * self.cell_size
            y1 = (self.env.height - 1 - ty) * self.cell_size
            cx = x1 + self.cell_size / 2
            cy = y1 + self.cell_size / 2
            r = self.cell_size * 0.35
            self.canvas.create_polygon(
                cx, cy - r,
                cx + r, cy,
                cx, cy + r,
                cx - r, cy,
                fill="#7c3aed", outline="#5b21b6"
            )

        for ox, oy in self.env.opponents:
            offset = self.cell_size * 0.2
            x1 = ox * self.cell_size + offset
            y1 = (self.env.height - 1 - oy) * self.cell_size + offset
            self.canvas.create_rectangle(x1, y1, x1 + self.cell_size * 0.6, y1 + self.cell_size * 0.6, fill="#990000",
                                         outline="#7a0000")

        ax, ay = self.env.agent_pos
        offset = self.cell_size * 0.15
        x1 = ax * self.cell_size + offset
        y1 = (self.env.height - 1 - ay) * self.cell_size + offset
        self.canvas.create_oval(x1, y1, x1 + self.cell_size * 0.7, y1 + self.cell_size * 0.7, fill="#000066",
                                outline="#1e3a8a")

    def run_loop(self):
        self.btn.config(state="disabled")

        def step():
            if not self.env.is_done():
                # LAB 2: the agent senses, decides, then the environment executes the action
                percept = self.env.get_percept()
                action = self.agent.sense_and_act(percept)
                self.env.execute_action(action)

                self.draw_grid()
                self.label.config(text=f"Score: {self.env.score} | Steps: {self.env.steps} | Action: {action}")
                self.root.after(250, step)
            else:
                end_text = f"Collision! Game Over! Final Score: {self.env.score}" if self.env.collision else f"Finished! Final Score: {self.env.score}"
                self.label.config(text=end_text)
                self.btn.config(state="normal")

        step()


if __name__ == "__main__":
    root = tk.Tk()
    # LAB 2: change agent_type to 'simple' to observe the SimpleReflexAgent getting
    # stuck in an infinite loop, or 'model' to see the ModelBasedAgent escape it.
    app = GridGameGUI(root, width=12, height=12, num_food=15, num_opponents=0, agent_type='simple')
    root.mainloop()