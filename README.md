This is a Traffic light controller, This project implements a trained reinforcement learning-based traffic light controller to optimize traffic flow and reduce congestion at a single four-way intersection with four lanes per road (west, east, north, and south). The controller leverages a Deep Q-Network (DQN) trained over 2500 episodes using SUMO simulations to make real-time traffic signal decisions based on lane-specific data.

Tools and Technologies :-
SUMO and netedit: For traffic simulation and intersection network design.
Python: For reinforcement learning implementation.
TensorFlow/PyTorch: For DQN model development.
Matplotlib: For visualizing performance metrics.


https://github.com/user-attachments/assets/201d63eb-95ec-4c1c-9462-e40f08434a6d

The DQN model takes lane-specific features like queue lengths and waiting times as input to determine optimal signal phase actions. The trained model minimizes average waiting time and adapts efficiently to real-time traffic conditions.
