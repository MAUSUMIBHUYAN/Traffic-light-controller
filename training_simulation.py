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
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs):
        self._model = Model
        self._memory = Memory
        self._traffic_gen = TrafficGen
        self._gamma = gamma
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._training_epochs = training_epochs

        # Simulation statistics
        self._step = 0
        self._waiting_times = {}
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []

    def run(self, episode, epsilon):
        """
        Runs a simulation episode and trains the model.
        """
        start_time = timeit.default_timer()

        # Generate route file for the current episode and start SUMO
        self._traffic_gen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # Initialize variables
        self._step = 0
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        old_total_wait = 0
        old_state, old_action = -1, -1

        while self._step < self._max_steps:
            current_state = self._get_state()
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            if self._step != 0:
                self._memory.add_sample((old_state, old_action, reward, current_state))

            action = self._choose_action(current_state, epsilon)

            # Handle phase transition
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # Execute chosen green phase
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # Update variables
            old_state, old_action = current_state, action
            old_total_wait = current_total_wait

            if reward < 0:
                self._sum_neg_reward += reward

        self._save_episode_stats()
        print(f"Total reward: {self._sum_neg_reward} - Epsilon: {round(epsilon, 2)}")
        traci.close()

        simulation_time = round(timeit.default_timer() - start_time, 1)

        # Train the model
        print("Training...")
        start_time = timeit.default_timer()
        for _ in range(self._training_epochs):
            self._replay()
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time

    def _simulate(self, steps_todo):
        """
        Advance the simulation by a given number of steps, gathering statistics along the way.
        """
        steps_todo = min(steps_todo, self._max_steps - self._step)

        while steps_todo > 0:
            traci.simulationStep()
            self._step += 1
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length

    def _collect_waiting_times(self):
        """
        Calculate the total waiting time of vehicles on incoming roads.
        """
        incoming_roads = ["-E1", "-E2", "E0", "-E3"]
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            road_id = traci.vehicle.getRoadID(car_id)
            if road_id in incoming_roads:
                self._waiting_times[car_id] = traci.vehicle.getAccumulatedWaitingTime(car_id)
            elif car_id in self._waiting_times:
                del self._waiting_times[car_id]

        return sum(self._waiting_times.values())

    def _choose_action(self, state, epsilon):
        """
        Select an action using an epsilon-greedy policy.
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1)
        return np.argmax(self._model.predict_one(state))

    def _set_yellow_phase(self, old_action):
        """
        Activate the yellow phase corresponding to the previous action.
        """
        yellow_phase_code = old_action * 2 + 1
        traci.trafficlight.setPhase("J1", yellow_phase_code)

    def _set_green_phase(self, action_number):
        """
        Activate the green phase corresponding to the selected action.
        """
        phases = [PHASE_NS_GREEN, PHASE_NSL_GREEN, PHASE_EW_GREEN, PHASE_EWL_GREEN]
        traci.trafficlight.setPhase("J1", phases[action_number])

    def _get_queue_length(self):
        """
        Retrieve the number of halted vehicles on incoming roads.
        """
        halt_counts = [traci.edge.getLastStepHaltingNumber(edge) for edge in ["-E2", "-E3", "-E1", "E0"]]
        return sum(halt_counts)

    def _get_state(self):
        """
        Retrieve the current state of the intersection as a binary occupancy grid.
        """
        state = np.zeros(self._num_states)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = 750 - traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)

            lane_group = self._determine_lane_group(lane_id)
            lane_cell = self._determine_lane_cell(lane_pos)

            if lane_group is not None and lane_cell is not None:
                car_position = lane_group * 10 + lane_cell
                state[car_position] = 1

        return state

    def _determine_lane_group(self, lane_id):
        """
        Map a lane ID to its corresponding group.
        """
        lane_groups = {
            "E0_": 0, "-E2_": 2, "-E1_": 4, "-E3_": 6
        }
        for key, group in lane_groups.items():
            if lane_id.startswith(key):
                return group + (3 if lane_id.endswith("3") else 0)

        return None

    def _determine_lane_cell(self, lane_pos):
        """
        Map a lane position to its corresponding cell.
        """
        thresholds = [7, 14, 21, 28, 40, 60, 100, 160, 400, 750]
        for i, threshold in enumerate(thresholds):
            if lane_pos < threshold:
                return i
        return None

    def _replay(self):
        """
        Train the model using experience replay.
        """
        batch = self._memory.get_samples(self._model.batch_size)

        if batch:
            states = np.array([sample[0] for sample in batch])
            next_states = np.array([sample[3] for sample in batch])

            q_values = self._model.predict_batch(states)
            q_next_values = self._model.predict_batch(next_states)

            x = np.zeros((len(batch), self._num_states))
            y = np.zeros((len(batch), self._num_actions))

            for i, (state, action, reward, _) in enumerate(batch):
                current_q = q_values[i]
                current_q[action] = reward + self._gamma * np.amax(q_next_values[i])
                x[i], y[i] = state, current_q

            self._model.train_batch(x, y)

    def _save_episode_stats(self):
        """
        Record statistics for the completed episode.
        """
        self._reward_store.append(self._sum_neg_reward)
        self._cumulative_wait_store.append(self._sum_waiting_time)
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store

    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store
