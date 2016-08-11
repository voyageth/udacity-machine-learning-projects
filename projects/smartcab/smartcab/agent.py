import random

import numpy as np
import pandas as pd

from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(
            env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.available_actions = [None, 'forward', 'left', 'right']
        self.available_states = [
            'GREEN_CAN_LEFT_NEXT_NONE'
            , 'GREEN_CAN_LEFT_NEXT_FORWARD'
            , 'GREEN_CAN_LEFT_NEXT_LEFT'
            , 'GREEN_CAN_LEFT_NEXT_RIGHT'

            , 'GREEN_CANT_LEFT_NEXT_NONE'
            , 'GREEN_CANT_LEFT_NEXT_FORWARD'
            , 'GREEN_CANT_LEFT_NEXT_LEFT'
            , 'GREEN_CANT_LEFT_NEXT_RIGHT'

            , 'RED_CAN_RIGHT_NEXT_NONE'
            , 'RED_CAN_RIGHT_NEXT_FORWARD'
            , 'RED_CAN_RIGHT_NEXT_LEFT'
            , 'RED_CAN_RIGHT_NEXT_RIGHT'

            , 'RED_CANT_RIGHT_NEXT_NONE'
            , 'RED_CANT_RIGHT_NEXT_FORWARD'
            , 'RED_CANT_RIGHT_NEXT_LEFT'
            , 'RED_CANT_RIGHT_NEXT_RIGHT'
        ]
        self.epsilon = 0.1  # choose wrong action randomly by epsilon
        column_names = []
        column_names.extend(self.available_actions)
        self.q_values = pd.DataFrame(np.zeros((len(self.available_states), len(column_names))),
                                     columns=column_names)
        self.learning_count = 1
        self.gamma = 0.1

    def get_current_q_value(self, state, action):
        state_index = self.available_states.index(state)
        return self.q_values.get_value(state_index, action)

    def set_q_value(self, state, action, value):
        state_index = self.available_states.index(state)
        self.q_values.set_value(state_index, action, value)

    def find_max_q_value_in_state(self, state):
        max_value = 0
        for action in self.available_actions:
            action_value = self.get_current_q_value(state, action)
            if action_value > max_value:
                max_value = action_value

        return max_value

    def find_max_q_value_action_in_state(self, state):
        max_value = 0
        max_value_action = ""
        for action in self.available_actions:
            action_value = self.get_current_q_value(state, action)
            if action_value >= max_value:
                max_value = action_value
                max_value_action = action

        return max_value_action

    def calculate_state(self, inputs, next_waypoint):
        if next_waypoint == "forward":
            next_waypoint_postfix = "_NEXT_FORWARD"
        elif next_waypoint == "left":
            next_waypoint_postfix = "_NEXT_LEFT"
        elif next_waypoint == "right":
            next_waypoint_postfix = "_NEXT_RIGHT"
        else:
            next_waypoint_postfix = "_NEXT_NONE"

        if inputs.get('light') == 'green':
            oncoming = inputs.get('oncoming')
            if oncoming == None or oncoming == 'left':
                return 'GREEN_CAN_LEFT' + next_waypoint_postfix  # None, 'forward', 'right', 'left'
            else:
                return 'GREEN_CANT_LEFT' + next_waypoint_postfix  # None, 'forward', 'right'
        else:
            if inputs.get('left') != 'forward':
                return 'RED_CAN_RIGHT' + next_waypoint_postfix  # None, 'right'
            else:
                return 'RED_CANT_RIGHT' + next_waypoint_postfix  # None

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state
        self.state = self.calculate_state(inputs, self.next_waypoint)

        # Select action according to your policy
        # use q-learning
        if random.random() < self.epsilon:
            # random policy
            action = random.choice(self.available_actions)
        else:
            # use q_values
            action = self.find_max_q_value_action_in_state(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # Learn policy based on state, action, reward
        next_state_inputs = self.env.sense(self)
        next_waypoint_after_move = self.planner.next_waypoint()
        next_state = self.calculate_state(next_state_inputs, next_waypoint_after_move)
        learning_rate = 1 / float(self.learning_count)
        new_q_value = ((1 - learning_rate) * self.get_current_q_value(self.state, action)) + (
            learning_rate * (reward + self.gamma * self.find_max_q_value_in_state(next_state)))
        self.set_q_value(self.state, action, new_q_value)
        self.learning_count += 1

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}" \
            .format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
