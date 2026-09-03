import random


# =============================================================
# LAB 01 AGENT
# =============================================================

class GreedyGridAgent:
    """
    Original Lab 01 agent.

    This agent is kept here so the previous work
    is not lost.
    """

    def __init__(self):

        self.actions_pool = [
            'Up',
            'Down',
            'Left',
            'Right'
        ]

    def sense_and_act(self, percept: dict) -> str:

        return random.choice(
            self.actions_pool
        )


# =============================================================
# LAB 02 - SIMPLE REFLEX AGENT
# =============================================================

class SimpleReflexAgent:
    """
    Simple Reflex Agent.

    The agent makes decisions using only
    the current percept.

    It does NOT maintain history.
    """

    def sense_and_act(self, percept: dict) -> str:

        # -----------------------------------------------------
        # CONDITION-ACTION RULE 1
        # IF food_here THEN move forward
        # -----------------------------------------------------

        if percept['food_here']:

            return 'Forward'

        # -----------------------------------------------------
        # CONDITION-ACTION RULE 2
        # IF wall_ahead THEN turn left
        # -----------------------------------------------------

        if percept['wall_ahead']:

            return 'TurnLeft'

        # -----------------------------------------------------
        # CONDITION-ACTION RULE 3
        # ELSE move forward
        # -----------------------------------------------------

        return 'Forward'


# =============================================================
# LAB 02 - MODEL BASED AGENT
# =============================================================

class ModelBasedAgent:
    """
    Model-Based Agent.

    Unlike the Simple Reflex Agent, this agent
    maintains internal state / memory.
    """

    def __init__(self):

        # Remember previously observed percept states
        self.visited_states = set()

        # Remember the previous action
        self.last_action = None

        # Keep track of how many decisions have been made
        self.step_count = 0

    def sense_and_act(self, percept: dict) -> str:

        # -----------------------------------------------------
        # STEP 1: UPDATE INTERNAL STATE
        # -----------------------------------------------------

        current_state = (
            percept['wall_ahead'],
            percept['food_here']
        )

        self.visited_states.add(
            current_state
        )

        self.step_count += 1

        # -----------------------------------------------------
        # STEP 2: CONDITION-ACTION RULES
        # -----------------------------------------------------

        # IF food is here
        if percept['food_here']:

            action = 'Forward'

        # IF wall is ahead
        elif percept['wall_ahead']:

            # If we previously turned left,
            # try turning right instead.
            if self.last_action == 'TurnLeft':

                action = 'TurnRight'

            else:

                action = 'TurnLeft'

        # ELSE move forward
        else:

            action = 'Forward'

        # -----------------------------------------------------
        # STEP 3: REMEMBER THE ACTION
        # -----------------------------------------------------

        self.last_action = action

        return action