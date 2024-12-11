import traci
import numpy as np
import random
import timeit
import os

# Phase codes based on design.net.xml
PHASE_NS_GREEN = 0  # Action 0: North-South green
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # Action 1: North-South left-turn green
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # Action 2: East-West green
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # Action 3: East-West left-turn green
PHASE_EWL_YELLOW = 7


class Simulation:
    def __init__(self, model, traffic_gen, sumo_cmd, max_steps, green_duration, yellow_duration, num_states, num_actions):
        self._model = model
        self._traffic_gen = traffic_gen
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_episode = []
        self._queue_length_episode = []

    def run(self, episode):
        """
        Executes a single simulation episode.
        """
        start_time = timeit.default_timer()

        # Generate route file for this episode and initialize SUMO
        self._traffic_gen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # Initialization
        self._step = 0
        self._waiting_times = {}
        old_total_wait = 0
        previous_action = -1  # Dummy initialization

        while self._step < self._max_steps:
            # Get the current state of the traffic intersection
            current_state = self._get_state()

            # Calculate reward as the reduction in cumulative waiting time
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            # Choose the next traffic light phase based on the current state
            action = self._choose_action(current_state)

            # Activate the yellow phase if switching actions
            if self._step > 0 and previous_action != action:
                self._set_yellow_phase(previous_action)
                self._simulate(self._yellow_duration)

            # Activate the selected green phase
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # Update variables for tracking
            previous_action = action
            old_total_wait = current_total_wait
            self._reward_episode.append(reward)

        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time

    def _simulate(self, steps_to_simulate):
        """
        Advances the simulation by a specified number of steps.
        """
        if self._step + steps_to_simulate >= self._max_steps:
            steps_to_simulate = self._max_steps - self._step

        while steps_to_simulate > 0:
            traci.simulationStep()
            self._step += 1
            steps_to_simulate -= 1
            queue_length = self._get_queue_length()
            self._queue_length_episode.append(queue_length)

    def _collect_waiting_times(self):
        """
        Gathers the cumulative waiting times of vehicles on incoming roads.
        """
        incoming_roads = ["-E1", "-E2", "E0", "-E3"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)
            if road_id in incoming_roads:
                self._waiting_times[car_id] = wait_time
            elif car_id in self._waiting_times:
                del self._waiting_times[car_id]
        return sum(self._waiting_times.values())

    def _choose_action(self, state):
        """
        Determines the optimal action based on the current state using the trained model.
        """
        return np.argmax(self._model.predict_one(state))

    def _set_yellow_phase(self, previous_action):
        """
        Activates the appropriate yellow phase in SUMO.
        """
        yellow_phase_code = previous_action * 2 + 1
        traci.trafficlight.setPhase("J1", yellow_phase_code)

    def _set_green_phase(self, action):
        """
        Activates the appropriate green phase in SUMO.
        """
        if action == 0:
            traci.trafficlight.setPhase("J1", PHASE_NS_GREEN)
        elif action == 1:
            traci.trafficlight.setPhase("J1", PHASE_NSL_GREEN)
        elif action == 2:
            traci.trafficlight.setPhase("J1", PHASE_EW_GREEN)
        elif action == 3:
            traci.trafficlight.setPhase("J1", PHASE_EWL_GREEN)

    def _get_queue_length(self):
        """
        Computes the total number of stationary vehicles in incoming lanes.
        """
        halt_N = traci.edge.getLastStepHaltingNumber("-E2")
        halt_S = traci.edge.getLastStepHaltingNumber("-E3")
        halt_E = traci.edge.getLastStepHaltingNumber("-E1")
        halt_W = traci.edge.getLastStepHaltingNumber("E0")
        return halt_N + halt_S + halt_E + halt_W

    def _get_state(self):
        """
        Retrieves the state of the intersection as a binary representation of cell occupancy.
        """
        state = np.zeros(self._num_states)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = 750 - traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)

            if lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 100:
                lane_cell = 6
            elif lane_pos < 160:
                lane_cell = 7
            elif lane_pos < 400:
                lane_cell = 8
            else:
                lane_cell = 9

            if lane_id in {"E0_0", "E0_1", "E0_2"}:
                lane_group = 0
            elif lane_id == "E0_3":
                lane_group = 1
            elif lane_id in {"-E2_0", "-E2_1", "-E2_2"}:
                lane_group = 2
            elif lane_id == "-E2_3":
                lane_group = 3
            elif lane_id in {"-E1_0", "-E1_1", "-E1_2"}:
                lane_group = 4
            elif lane_id == "-E1_3":
                lane_group = 5
            elif lane_id in {"-E3_0", "-E3_1", "-E3_2"}:
                lane_group = 6
            elif lane_id == "-E3_3":
                lane_group = 7
            else:
                lane_group = -1

            if 0 <= lane_group <= 7:
                car_position = int(f"{lane_group}{lane_cell}")
                state[car_position] = 1

        return state

    @property
    def queue_length_episode(self):
        return self._queue_length_episode

    @property
    def reward_episode(self):
        return self._reward_episode
