import numpy as np
import math

class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated):
        self.n_cars_generated = n_cars_generated 
        self.max_steps = max_steps

    def generate_routefile(self, seed):
        """
        Generates a route file specifying the traffic flow for one episode.
        """
        np.random.seed(seed)  

        timings = np.random.weibull(2, self.n_cars_generated)
        timings = np.sort(timings)

        car_gen_steps = []
        old_min = math.floor(timings[1])
        old_max = math.ceil(timings[-1])
        new_min, new_max = 0, self.max_steps

        for value in timings:
            scaled_value = ((new_max - new_min) / (old_max - old_min)) * (value - old_min) + new_min
            car_gen_steps.append(scaled_value)

        car_gen_steps = np.rint(car_gen_steps).astype(int) 

        with open("environment/design.rou.xml", "w") as routes:
            print("""<routes>
            <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

            <route id="W_N" edges="E0 E2"/>
            <route id="W_E" edges="E0 E1"/>
            <route id="W_S" edges="E0 E3"/>
            <route id="N_W" edges="-E2 -E0"/>
            <route id="N_E" edges="-E2 E1"/>
            <route id="N_S" edges="-E2 E3"/>
            <route id="E_W" edges="-E1 -E0"/>
            <route id="E_N" edges="-E1 E2"/>
            <route id="E_S" edges="-E1 E3"/>
            <route id="S_W" edges="-E3 -E0"/>
            <route id="S_N" edges="-E3 E2"/>
            <route id="S_E" edges="-E3 E1"/>
            """, file=routes)

            for car_id, step in enumerate(car_gen_steps):
                direction_probability = np.random.uniform()

                if direction_probability < 0.75:  
                    straight_route = np.random.choice(["W_E", "E_W", "N_S", "S_N"])
                    print(f'    <vehicle id="{straight_route}_{car_id}" type="standard_car" route="{straight_route}" depart="{step}" departLane="random" departSpeed="10" />', file=routes)
                else:  
                    turn_route = np.random.choice(["W_N", "W_S", "N_W", "N_E", "E_N", "E_S", "S_W", "S_E"])
                    print(f'    <vehicle id="{turn_route}_{car_id}" type="standard_car" route="{turn_route}" depart="{step}" departLane="random" departSpeed="10" />', file=routes)

            print("</routes>", file=routes)
