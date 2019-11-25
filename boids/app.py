""" App for running a Boids simulation. """
from itertools import count
from sys import exit
from typing import List, Optional, Tuple

from pyglet.gl import *
from pyglet.graphics import Batch, OrderedGroup
from pyglet.window import Window, key, mouse

from .constants import WINDOW_HEIGHT, WINDOW_WIDTH
from .models import Model
from .system import BoidSystem
from .utilities import set_framerate

Color = Tuple[int, int, int]
GREEN = (0, 255, 0)
ORANGE = (255, 125, 125)
RED = (255, 0, 0)
WHITE = (255, 255, 255)


# noinspection PyAbstractClass
class App(Window):
    """ The applet which runs a Boid simulation. """

    def __init__(self, boid_count: int = 45, obstacle_count: int = 10, *, seed: Optional[int] = None):
        """
        :param boid_count: The number of Boids in the simulation.
        :param obstacle_count: The number of Obstacles in the simulation.
        :param seed: A seed to the random number generator used by the RNG.
        """
        super().__init__(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.system = BoidSystem(boid_count, obstacle_count, seed=seed)
        self._show_lines = False

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        """ Define functionality for key presses. """
        if symbol == key.SPACE:
            self.system.population, self.system.obstacles = self.system.new()
        if symbol == key.ESCAPE:
            self.on_close()

        # Toggles the lines.
        if symbol == key.L:
            self._show_lines ^= True

        # Adjustments to the alignment coefficient.
        if symbol == key.A:
            self.system.alignment += 1
        if symbol == key.Z:
            self.system.alignment -= 1

        # Adjustments to the cohesion coefficient.
        if symbol == key.D:
            self.system.cohesion += 1
        if symbol == key.C:
            self.system.cohesion -= 1

        # Adjustments to the fear coefficient.
        if symbol == key.F:
            self.system.fear += 1
        if symbol == key.V:
            self.system.fear -= 1

        # Adjustments to the separation coefficient.
        if symbol == key.S:
            self.system.separation += 1
        if symbol == key.X:
            self.system.separation -= 1

    def on_close(self):
        """ Close the window gracefully. """
        exit()

    def on_mouse_press(self, x: float, y: float, button: int, modifiers: int) -> None:
        """ Define functionality for mouse presses. """
        if button == mouse.LEFT:
            self.system.add_boid(x, y)
        if button == mouse.RIGHT:
            self.system.add_obstacle(x, y)

    def render(self) -> None:
        """ Render the board onto the screen. """
        # Clear the old board.
        self.clear()

        # Draw the board in a single batch.
        batch = Batch()
        obstacle_layer, connection_layer, boid_layer = (OrderedGroup(order) for order in range(3))
        if self._show_lines:
            batch = self.draw_connections(batch, connection_layer, "neighbors", color=GREEN)
            batch = self.draw_connections(batch, connection_layer, "obstacles", color=RED)
        batch = self.draw_models(batch, boid_layer, self.system.population, color=WHITE)
        batch = self.draw_models(batch, obstacle_layer, self.system.obstacles, color=RED)
        batch.draw()

        # Send to screen.
        self.flip()

    def draw_connections(self, batch: Batch, layer: OrderedGroup, collection: str, color: Color) -> Batch:
        """
            Draw the connections, as lines, between Boids and Models within their field of view.
        :param batch: The batch which the vertices are written to for batch processing.
        :param layer: The layer to associate the batch rendered to, which defines the order of rendering.
        :param collection: The string for the attribute used to pull the iterable of Models from each Flock.
        :param color: The color in which to render the line connections.
        :returns batch:
        """
        nodes = []  # Accumulates the positions of the nodes to be drawn.
        indices = []  # Accumulates the indices corresponding to the nodes to be drawn.
        node_index = count(0)  # Counter for tracking the indices.

        # Iterate over the flocks (This double counts all the connections).
        for flock in self.system.flocks:
            center_index = next(node_index)
            nodes.extend(flock.boid.position)
            for model in getattr(flock, collection):
                if (virtual_boid_position := flock.boid.get_image(model)) != flock.boid.position:
                    # If the connection takes place over a screen wrap, use the virtual positions of
                    #   the boids and models to accurately represent the toroidal space.
                    phantom_model_position = model.get_image(flock.boid)
                    nodes.extend([*phantom_model_position, *virtual_boid_position, *model.position])
                    indices += (center_index, next(node_index), next(node_index), next(node_index))
                else:
                    nodes.extend(model.position)
                    indices += (center_index, next(node_index))

        point_count = len(nodes) // 2
        colors = ('c3B', color * point_count)
        batch.add_indexed(point_count, GL_LINES, layer, indices, ('v2f', nodes), colors)
        return batch

    def draw_models(self, batch: Batch, layer: OrderedGroup, models: List[Model], color: Color) -> Batch:
        """
            Draw the models.
        :param batch: The batch which the vertices are written to for batch processing.
        :param layer: The layer to associate the batch rendered to, which defines the order of rendering.
        :param models: The list of models to render.
        :param color: The color in which to render the models.
        :returns batch:
        """
        number_of_sides = len(models[0].coordinates) // 2
        coordinates = [coord for model in models for coord in model.coordinates]
        coordinate_count = len(models) * number_of_sides
        indices = range(coordinate_count)

        # Determine the GL model to use based upon the number of sides.
        if number_of_sides == 3:
            mode = GL_TRIANGLES
        elif number_of_sides == 4:
            mode = GL_QUADS
        else:
            raise NotImplementedError

        colors = ('c3B', color * coordinate_count)
        batch.add_indexed(coordinate_count, mode, layer, indices, ('v2f', coordinates), colors)
        return batch

    def run(self):
        """ Run the application in manual mode. """
        @set_framerate(60)
        def run_(timestep: float):
            """ Run in a help function to limit framerate. """
            self.render()
            self.dispatch_events()
            self.system.update(timestep)

        while True:
            run_()
