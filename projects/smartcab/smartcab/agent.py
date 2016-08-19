import random

import numpy as np
import pandas as pd
from scipy.spatial import distance as dt

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
        self.epsilon = 0  # choose wrong action randomly by epsilon
        column_names = []
        column_names.extend(self.available_actions)
        self.q_values = pd.DataFrame(np.zeros((len(self.available_states), len(column_names))),
                                     columns=column_names)

        # big penalty when ignore rule of the road
        big_penalty_value = -100
        self.set_q_value('GREEN_CANT_LEFT_NEXT_NONE', 'left', big_penalty_value)
        self.set_q_value('GREEN_CANT_LEFT_NEXT_FORWARD', 'left', big_penalty_value)
        self.set_q_value('GREEN_CANT_LEFT_NEXT_LEFT', 'left', big_penalty_value)
        self.set_q_value('GREEN_CANT_LEFT_NEXT_RIGHT', 'left', big_penalty_value)
        self.set_q_value('RED_CAN_RIGHT_NEXT_NONE', 'forward', big_penalty_value)
        self.set_q_value('RED_CAN_RIGHT_NEXT_NONE', 'left', big_penalty_value)
        self.set_q_value('RED_CAN_RIGHT_NEXT_FORWARD', 'forward', big_penalty_value)
        self.set_q_value('RED_CAN_RIGHT_NEXT_FORWARD', 'left', big_penalty_value)
        self.set_q_value('RED_CAN_RIGHT_NEXT_LEFT', 'forward', big_penalty_value)
        self.set_q_value('RED_CAN_RIGHT_NEXT_LEFT', 'left', big_penalty_value)
        self.set_q_value('RED_CAN_RIGHT_NEXT_RIGHT', 'forward', big_penalty_value)
        self.set_q_value('RED_CAN_RIGHT_NEXT_RIGHT', 'left', big_penalty_value)
        self.set_q_value('RED_CANT_RIGHT_NEXT_NONE', 'forward', big_penalty_value)
        self.set_q_value('RED_CANT_RIGHT_NEXT_NONE', 'left', big_penalty_value)
        self.set_q_value('RED_CANT_RIGHT_NEXT_NONE', 'right', big_penalty_value)
        self.set_q_value('RED_CANT_RIGHT_NEXT_FORWARD', 'forward', big_penalty_value)
        self.set_q_value('RED_CANT_RIGHT_NEXT_FORWARD', 'left', big_penalty_value)
        self.set_q_value('RED_CANT_RIGHT_NEXT_FORWARD', 'right', big_penalty_value)
        self.set_q_value('RED_CANT_RIGHT_NEXT_LEFT', 'forward', big_penalty_value)
        self.set_q_value('RED_CANT_RIGHT_NEXT_LEFT', 'left', big_penalty_value)
        self.set_q_value('RED_CANT_RIGHT_NEXT_LEFT', 'right', big_penalty_value)
        self.set_q_value('RED_CANT_RIGHT_NEXT_RIGHT', 'forward', big_penalty_value)
        self.set_q_value('RED_CANT_RIGHT_NEXT_RIGHT', 'left', big_penalty_value)
        self.set_q_value('RED_CANT_RIGHT_NEXT_RIGHT', 'right', big_penalty_value)

        self.learning_count = 1
        self.learning_count_delta = 1
        self.gamma = 0.25

        self.deadline = 0
        self.current_deadline = 0
        self.distance = 0
        self.total_reward = 0
        self.penalty = 0

        self.trial = -1
        self.learn_results = pd.DataFrame(np.zeros((100, 6)),
                                          columns=["distance", "currentDeadline", "initialDeadline", "usedStep",
                                                   "totalReward", "penalty"])

        self.learn_results_without_random = pd.DataFrame(np.zeros((100, 6)),
                                                         columns=["distance", "currentDeadline", "initialDeadline",
                                                                  "usedStep",
                                                                  "totalReward", "penalty"])

        self.learn_results_without_random_last_10 = pd.DataFrame(np.zeros((100, 6)),
                                                                 columns=["distance", "currentDeadline",
                                                                          "initialDeadline", "usedStep",
                                                                          "totalReward", "penalty"])

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
        agent_state = self.env.agent_states[self.env.primary_agent]
        self.distance = dt.euclidean(agent_state.get("destination"), agent_state.get("location"))
        self.deadline = agent_state.get("deadline")
        self.current_deadline = self.deadline
        self.total_reward = 0
        self.penalty = 0
        self.trial += 1

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state
        self.state = self.calculate_state(inputs, self.next_waypoint)

        # Select action according to your policy
        # use q-learning
        if random.random() < self.epsilon and self.trial < 90:
            # random policy
            action = random.choice(self.available_actions)
            is_random_choose = True
        else:
            # use q_values
            action = self.find_max_q_value_action_in_state(self.state)
            is_random_choose = False
        # action = random.choice(self.available_actions)

        # Execute action and get reward
        reward = self.env.act(self, action)

        self.current_deadline = self.env.agent_states[self.env.primary_agent].get("deadline")
        if reward < -0.5:
            self.penalty += 1
            # big penalty!
            self.total_reward += (reward * 100)
        else:
            self.total_reward += reward

        # Learn policy based on state, action, reward
        next_state_inputs = self.env.sense(self)
        next_waypoint_after_move = self.planner.next_waypoint()
        next_state = self.calculate_state(next_state_inputs, next_waypoint_after_move)
        learning_rate = 1 / float(self.learning_count)
        new_reward = float(reward) / float(max(self.deadline - self.current_deadline, 1)) * 10
        new_q_value = ((1 - learning_rate) * self.get_current_q_value(self.state, action)) + (
            learning_rate * (new_reward + self.gamma * self.find_max_q_value_in_state(next_state)))
        self.set_q_value(self.state, action, new_q_value)
        self.learning_count += self.learning_count_delta
        if reward < -0.5 and self.trial >= 90:
            print "LearningAgent.update(): deadline = {}, inputs = {}, state = {}, action = {}, reward = {}".format(
                deadline, inputs, self.state, action, reward)  # [debug]

        self.learn_results.set_value(self.trial, "distance", self.distance)
        self.learn_results.set_value(self.trial, "currentDeadline", self.current_deadline)
        self.learn_results.set_value(self.trial, "initialDeadline", self.deadline)
        self.learn_results.set_value(self.trial, "usedStep", (self.deadline - self.current_deadline))
        self.learn_results.set_value(self.trial, "totalReward", self.total_reward)
        self.learn_results.set_value(self.trial, "penalty", self.penalty)

        if not is_random_choose:
            self.learn_results_without_random.set_value(self.trial, "distance", self.distance)
            self.learn_results_without_random.set_value(self.trial, "currentDeadline", self.current_deadline)
            self.learn_results_without_random.set_value(self.trial, "initialDeadline", self.deadline)
            self.learn_results_without_random.set_value(self.trial, "usedStep", (self.deadline - self.current_deadline))
            self.learn_results_without_random.set_value(self.trial, "totalReward", self.total_reward)
            self.learn_results_without_random.set_value(self.trial, "penalty", self.penalty)

            if self.trial >= 90:
                self.learn_results_without_random_last_10.set_value(self.trial, "distance", self.distance)
                self.learn_results_without_random_last_10.set_value(self.trial, "currentDeadline",
                                                                    self.current_deadline)
                self.learn_results_without_random_last_10.set_value(self.trial, "initialDeadline", self.deadline)
                self.learn_results_without_random_last_10.set_value(self.trial, "usedStep",
                                                                    (self.deadline - self.current_deadline))
                self.learn_results_without_random_last_10.set_value(self.trial, "totalReward", self.total_reward)
                self.learn_results_without_random_last_10.set_value(self.trial, "penalty", self.penalty)

    def print_result(self):
        self.learn_results_without_random = self.learn_results_without_random[
            (self.learn_results_without_random.T != 0).any()]
        self.learn_results_without_random_last_10 = self.learn_results_without_random_last_10[
            (self.learn_results_without_random_last_10.T != 0).any()]

        # print "{} {} {} {} {} {} {} {} {} learn_results".format(self.learning_count_delta, self.gamma, self.epsilon,
        #                                                         self.learn_results.size / 6,
        #                                                         self.learn_results["distance"].mean(),
        #                                                         self.learn_results["initialDeadline"].mean(),
        #                                                         self.learn_results["usedStep"].mean(),
        #                                                         self.learn_results["totalReward"].mean(),
        #                                                         self.learn_results["penalty"].mean(),
        #                                                         )
        # print "{} {} {} {} {} {} {} {} {} learn_results_without_random".format(self.learning_count_delta, self.gamma,
        #                                                                        self.epsilon,
        #                                                                        self.learn_results_without_random.size / 6,
        #                                                                        self.learn_results_without_random[
        #                                                                            "distance"].mean(),
        #                                                                        self.learn_results_without_random[
        #                                                                            "initialDeadline"].mean(),
        #                                                                        self.learn_results_without_random[
        #                                                                            "usedStep"].mean(),
        #                                                                        self.learn_results_without_random[
        #                                                                            "totalReward"].mean(),
        #                                                                        self.learn_results_without_random[
        #                                                                            "penalty"].mean(),
        #                                                                        )
        print "{} {} {} {} {} {} {} {} {}".format(self.learning_count_delta,
                                                  self.gamma, self.epsilon,
                                                  self.learn_results_without_random_last_10.size / 6,
                                                  self.learn_results_without_random_last_10[
                                                      "distance"].mean(),
                                                  self.learn_results_without_random_last_10[
                                                      "initialDeadline"].mean(),
                                                  self.learn_results_without_random_last_10[
                                                      "usedStep"].mean(),
                                                  self.learn_results_without_random_last_10[
                                                      "totalReward"].mean(),
                                                  self.learn_results_without_random_last_10[
                                                      "penalty"].mean(),
                                                  )
        print self.q_values


def run_instance(learning_count_delta, gamma, epsilon, agents):
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    a.learning_count_delta = learning_count_delta
    a.gamma = gamma
    a.epsilon = epsilon
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0001,
                    display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    agents.append(e.primary_agent)


def run():
    """Run the agent for a finite number of trials."""
    agents = []

    # test learning_count_delta
    run_instance(0.2, 0.8, 0.9, agents)

    for i in range(0, len(agents)):
        agents[i].print_result()


if __name__ == '__main__':
    run()
