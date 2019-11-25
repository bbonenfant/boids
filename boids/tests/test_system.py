""" Tests for system.py """
import numpy as np

from ..models import Boid
from ..system import Flock
from ..vector2d import Vector2D


class TestFlock:
    """ Tests for the Flock class. """
    zero_vector = Vector2D(0, 0)
    frame = Vector2D(40, 30)

    def construct_boid(self,
                       position: Vector2D = zero_vector,
                       velocity: Vector2D = zero_vector,
                       acceleration: Vector2D = zero_vector,
                       ) -> Boid:
        """ Construct a Boid with default vectors of magnitude zero and mocked frame dimensions. """
        boid = Boid(position, velocity, acceleration)
        boid.frame = self.frame
        return boid

    def test_get_neighbors(self):
        """ Test that the get_neighbors method collects neighboring Boids expected. """
        # Arrange
        radius = 10
        center_boid = self.construct_boid(position=(self.frame / 2))
        neighbors = [self.construct_boid(position=((10 * Vector2D.random()) + center_boid.position)) for _ in range(10)]
        strangers = [self.construct_boid(position=Vector2D.random()) for _ in range(10)]
        population = [center_boid, *neighbors, *strangers]

        # Act
        results = Flock.get_neighbors(center_boid, population, radius)

        # Assert
        assert set(results.keys()) == set(neighbors)

    def test_average_velocity(self):
        """ Test that the average velocity of the flock is computed correctly. """
        # Arrange
        velocities = np.random.rand(10, 2)
        average_velocity = Vector2D.from_array(list(np.mean(velocities, axis=0)))

        radius = 10
        center_boid = self.construct_boid()
        neighbors = [self.construct_boid(position=abs(Vector2D.random()), velocity=Vector2D.from_array(velocity))
                     for velocity in velocities]
        flock = Flock(center_boid, population=neighbors, obstacles=[], radius=radius,
                      alignment=1.0, cohesion=1.0, fear=1.0, separation=1.0)

        # Act
        results = flock.average_velocity

        # Assert
        assert results == average_velocity

    def test_center_of_mass(self):
        """ Test that the center of mass of the flock is computed correctly. """
        # Arrange
        positions = np.random.rand(10, 2)
        average_position = Vector2D.from_array(list(np.mean(positions, axis=0)))

        radius = 10
        center_boid = self.construct_boid()
        neighbors = [self.construct_boid(position=Vector2D.from_array(position)) for position in positions]
        flock = Flock(center_boid, population=neighbors, obstacles=[], radius=radius,
                      alignment=1.0, cohesion=1.0, fear=1.0, separation=1.0)

        # Act
        results = flock.center_of_mass

        # Assert
        assert results == average_position
