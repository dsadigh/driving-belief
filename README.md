This code does active information gathering to learn human's internal state in autonomous driving scenarios by actively probing the human drivers.

(Companion code to a paper presented at IROS 2016)

### Running

To run simply execute the following command: ./run {name_of_world}, where name_of_world can be anything defined in world.py.
The following are all valid for example

./run world1
./run world1_active
./run world1_active_distracted

To visualize a saved scenario you can run

./vis {path_of_saved_file}

### Modules

- dynamics.py: This contains code for car dynamics.
- car.py: Relevant code for different car models (human-driven, autonomous, etc.)
- feature.py: Definition of features.
- lane.py: Definition of driving lanes.
- trajectory.py: Definition of trajectories.
- world.py: This code contains different scenarios (each consisting of lanes/cars/etc.).
- visualize.py: This contains the code for visualization (GUI).
