""" The functionality for determining the movement of Boids in their environment."""
from __future__ import annotations
from functools import cached_property
import random
from typing import Dict, Iterator, List, Optional, Tuple

from .models import Boid, Model, Obstacle
from .utilities import mean
from .vector2d import Vector2D, Zero


class Flock:
    """ The collection of Boids which will influence the movement of the center Boid. """

    def __init__(self, boid: Boid, population: Iterator[Boid], obstacles: Iterator[Obstacle],
                 radius: int, alignment: float, cohesion: float, fear: float, separation: float):
        """
        :param boid: The Boid constructing this Flock.
        :param population: A collection of Boids which may be within the center Boid's field of view.
        :param obstacles: A collection of obstacles which may be within the center Boid's field of view.
        :param radius: The radius from the center Boid which defines it's field of view.
        :param alignment: The alignment coefficient for determining the steering force.
        :param cohesion: The cohesion coefficient for determining the steering force.
        :param fear: The fear coefficient for determining the steering force.
        :param separation: The separation coefficient for determining the steering force.
        """
        self.boid = boid
        self.radius = radius
        self.neighbors = self.get_neighbors(boid, population, radius)
        self.obstacles = self.get_neighbors(boid, obstacles, radius)

        self.alignment_coefficient = alignment
        self.cohesion_coefficient = cohesion
        self.fear_coefficient = fear
        self.separation_coefficient = separation

    @cached_property
    def average_velocity(self) -> Vector2D:
        return mean(neighbor.velocity for neighbor in self.neighbors)

    @cached_property
    def center_of_mass(self) -> Vector2D:
        return mean(neighbor.get_image(self.boid) for neighbor in self.neighbors)

    @cached_property
    def alignment_vector(self) -> Vector2D:
        """ Calculate the alignment vector. """
        return self.steering_vector(self.average_velocity)

    @cached_property
    def cohesion_vector(self) -> Vector2D:
        """ Calculate the cohesion vector. """
        correction_vector = self.center_of_mass - self.boid.position
        return self.steering_vector(correction_vector)

    @cached_property
    def fear_vector(self) -> Vector2D:
        """ Calculate the fear vector. """
        correction_vector = mean(
            obstacle.displacement(self.boid) / distance
            for obstacle, distance in self.obstacles.items()
        )
        return self.steering_vector(correction_vector)

    @cached_property
    def separation_vector(self) -> Vector2D:
        """ Calculate the separation vector. """
        correction_vector = mean(
            neighbor.displacement(self.boid) / distance
            for neighbor, distance in self.neighbors.items()
        )
        return self.steering_vector(correction_vector)

    def apply_impulse(self):
        """ Update the acceleration vector of the boid. """
        impulse = Zero
        if self.neighbors:
            impulse += (self.alignment_coefficient * self.alignment_vector)
            impulse += (self.cohesion_coefficient * self.cohesion_vector)
            impulse += (self.separation_coefficient * self.separation_vector)
        if self.obstacles:
            impulse += (self.fear_coefficient * self.fear_vector)
        self.boid.acceleration = impulse

    def steering_vector(self, desired_velocity: Vector2D) -> Vector2D:
        """ Calculate the steering vector. """
        if not desired_velocity:
            return Zero
        return desired_velocity.resize(self.boid.max_velocity) - self.boid.velocity

    @staticmethod
    def get_neighbors(boid: Boid, population: Iterator[Model], radius: float) -> Dict[Boid, float]:
        """ Returns a list of neighbors about the center Boid. """
        return {
            other_boid: distance for other_boid in population
            if 0 < (distance := boid.distance(other_boid)) < radius
        }


class BoidSystem:
    """ The collection of Boids in the system. """

    def __init__(self, boid_count: int, obstacle_count: int, radius: int = 150,
                 *, seed: Optional[int] = None):
        """
        :param boid_count: The number of Boids in the simulation.
        :param seed: A seed to the random number generator used by the RNG.
        """
        self.seed = seed
        random.seed(self.seed)
        self._initial_boid_count = boid_count
        self._initial_obstacle_count = obstacle_count

        self.radius = radius
        self.population, self.obstacles = self.new()
        self.flocks = []

        self.alignment = self.cohesion = self.fear = 10.0
        self.separation = 12.0

    def add_boid(self, x_coordinate: float, y_coordinate: float):
        position = Vector2D(x_coordinate, y_coordinate)
        velocity = acceleration = Vector2D(0, 0)
        baby_boid = Boid(position=position, velocity=velocity, acceleration=acceleration)
        self.population.append(baby_boid)

    def add_obstacle(self, x_coordinate: float, y_coordinate: float):
        position = Vector2D(x_coordinate, y_coordinate)
        self.obstacles.append(Obstacle(position=position))

    def new(self) -> Tuple[List[Boid], List[Obstacle]]:
        """ Recreate a new system. """
        boids = [Boid.random() for _boid in range(self._initial_boid_count)]
        obstacles = [Obstacle.random() for _obstacle in range(self._initial_obstacle_count)]
        return boids, obstacles

    def update(self, timestep: float) -> BoidSystem:
        """ Update the position of all the Boids. """
        self.flocks = [Flock(boid, self.population, self.obstacles, self.radius,
                             self.alignment, self.cohesion, self.fear, self.separation)
                       for boid in self.population]
        for flock in self.flocks:
            flock.apply_impulse()
            flock.boid.update(timestep)
        return self
