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
        # TODO add next_waypoint to state.
        self.available_states = ['GREEN_CAN_LEFT', 'GREEN_CANT_LEFT', 'RED_CAN_RIGHT', 'RED_CANT_RIGHT']
        self.epsilon = 0.1  # choose wrong action randomly by epsilon
        self.q_values = pd.DataFrame(np.zeros((4, 4)), columns=self.available_actions)
        self.learning_count = 1
        self.gamma = 0.1

    def get_prev_q_value(self, state, action):
        state_index = self.available_states.index(state)
        return self.q_values.get_value(state_index, action)

    def set_q_value(self, state, action, value):
        state_index = self.available_states.index(state)
        self.q_values.set_value(state_index, action, value)

    def find_max_q_value_in_state(self, state):
        max_value = 0
        for action in self.available_actions:
            action_value = self.get_prev_q_value(state, action)
            if action_value > max_value:
                max_value = action_value

        return max_value

    def calculate_state(self, inputs):
        if inputs.get('light') == 'green':
            oncoming = inputs.get('oncoming')
            if oncoming == None or oncoming == 'left':
                return 'GREEN_CAN_LEFT'  # None, 'forward', 'right', 'left'
            else:
                return 'GREEN_CANT_LEFT'  # None, 'forward', 'right'
        else:
            if inputs.get('left') != 'forward':
                return 'RED_CAN_RIGHT'  # None, 'right'
            else:
                return 'RED_CANT_RIGHT'  # None

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state
        self.state = self.calculate_state(inputs)

        # TODO: Select action according to your policy

        # my-way policy
        # action = self.next_waypoint

        # random policy
        # action = random.choice(self.available_actions)

        # law-abiding policy
        if self.next_waypoint == 'forward':
            if self.state == 'GREEN_CAN_LEFT' or self.state == 'GREEN_CANT_LEFT':
                action = 'forward'
            else:
                action = None
        elif self.next_waypoint == 'left':
            if self.state == 'GREEN_CAN_LEFT':
                action = 'left'
            else:
                action = None
        elif self.next_waypoint == 'right':
            if self.state == 'GREEN_CAN_LEFT' or self.state == 'GREEN_CANT_LEFT' or self.state == 'RED_CAN_RIGHT':
                action = 'right'
            else:
                action = None
        else:
            action = None

        # TODO use q-learning

        # Execute action and get reward
        reward = self.env.act(self, action)

        # Learn policy based on state, action, reward
        next_state_inputs = self.env.sense(self)
        next_state = self.calculate_state(next_state_inputs)
        learning_rate = 1 / float(self.learning_count)
        new_q_value = ((1 - learning_rate) * self.get_prev_q_value(self.state, action)) + (
            learning_rate * (reward + self.gamma * self.find_max_q_value_in_state(next_state)))

        print "state : " + self.state + ", action : " + str(action) + ", next_state : " + str(
            next_state) + ", new_q_value : " + str(
            new_q_value)
        self.set_q_value(self.state, action, new_q_value)
        print self.q_values

        # print self.q_values

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
