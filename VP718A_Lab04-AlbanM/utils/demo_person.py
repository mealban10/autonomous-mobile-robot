import pybullet as p
import os
import random


class ValidPathGraph():
    def __init__(self):
        # Store movement nodes and connections
        self.nodes = set()
        self.connections = {}
    
    def add_position(self, x, y):
        # Add a valid position
        self.nodes.add((x, y))

    def add_positions(self, xs):
        # Add multiple positions
        for x in xs:
            self.add_position(x[0], x[1])
        return self
    
    def add_connection(self, p1, p2):
        # Connect two valid positions
        if p1 in self.nodes and p2 in self.nodes:

            if p1 not in self.connections:
                self.connections[p1] = []

            if p2 not in self.connections:
                self.connections[p2] = []

            self.connections[p1].append(p2)
            self.connections[p2].append(p1)    

    def random_target_from(self, x):
        # Select a random connected target
        print(self.connections)
        return random.sample(self.connections[x], 1)[0]


class DemoPerson:
    def __init__(self, x=0, y=0):
        # Load the person model
        urdf_path = os.path.abspath("models/bubbleperson.urdf")
        self.objectId = p.loadURDF(urdf_path, [25, 0, 0])

        # Prevent the person from rotating and falling
        p.changeDynamics(
            self.objectId,
            -1,
            angularDamping=1.0
        )

        self.new_orn = p.getQuaternionFromEuler([0, 0, 0])

        self.collidable = set()

        self.collision_callback = None
        self.movements = None
        self.speed = 0.03

        self.current_location = (25, 0)
        self.active_steps = []

        # Update movement every 25 simulation steps
        self.movement_counter = 0
        self.movement_interval = 25

    def set_movement_graph(self, g):
        # Assign movement graph and initial position
        self.movements = g

        x = random.sample(
            list(self.movements.nodes),
            1
        )[0]

        self.set_position(x[0], x[1])

    def set_position(self, x, y):
        # Set person position
        p.resetBasePositionAndOrientation(
            self.objectId,
            [x, y, 0.9],
            self.new_orn
        )

        self.current_location = (x, y)

    def set_collision_callback(self, f):
        # Set collision callback
        self.collision_callback = f

    def add_collidable(self, new_id):
        # Register a collidable object
        self.collidable.add(new_id)

        p.setCollisionFilterPair(
            new_id,
            self.objectId,
            -1,
            -1,
            enableCollision=0
        )

    def pick_new_path(self):
        # Select a random connected target
        targ = self.movements.random_target_from(
            self.current_location
        )

        dx = targ[0] - self.current_location[0]
        dy = targ[1] - self.current_location[1]

        dist = (dx * dx + dy * dy) ** 0.5
        steps = int(dist / self.speed)

        # Generate intermediate movement positions
        self.active_steps = [
            (
                self.current_location[0] + (float(i) / steps) * dx,
                self.current_location[1] + (float(i) / steps) * dy
            )
            for i in range(1, steps)
        ]

        self.active_steps.append(targ)
        self.active_steps.reverse()

    def update_position(self):

        # Update movement counter
        self.movement_counter += 1

        # Move only every few simulation steps
        if self.movement_counter < self.movement_interval:
            return

        self.movement_counter = 0

        # Generate a new path if needed
        if len(self.active_steps) == 0:
            self.pick_new_path()

        else:
            # Move to the next position
            (x, y) = self.active_steps.pop()
            self.set_position(x, y)

    def step_action(self):
        """Overwrite this to change behaviour"""

        # Check for collisions
        for o in self.collidable:

            contact_points = p.getContactPoints(
                bodyA=o,
                bodyB=self.objectId
            )

            if len(contact_points) > 0:
                print("Collision detected!")

                if self.collision_callback is not None:
                    self.collision_callback()

        # Update movement if a graph is assigned
        if self.movements is not None:
            self.update_position()
        