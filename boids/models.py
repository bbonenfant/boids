""" The objects of the system. """
from __future__ import annotations
from abc import ABC, abstractmethod
from math import copysign
from typing import Collection

from .constants import WINDOW_HEIGHT, WINDOW_WIDTH
from .vector2d import Vector2D


class Model(ABC):
    """ Base class for Models. """
    frame = Vector2D(WINDOW_WIDTH, WINDOW_HEIGHT)

    def __init__(self, position: Vector2D):
        """
        :param position: The coordinates of the Model.
        """
        self.position = position % self.frame

    @property
    @abstractmethod
    def coordinates(self) -> Collection[float]:
        """ Returns the coordinates of the vertices that define the Model. """
        ...

    def distance(self, other: Model) -> float:
        """ Calculate the euclidean distance to another Model in the toroidal space. """
        return self.displacement(other).magnitude

    def displacement(self, other: Model) -> Vector2D:
        """ Calculate the displacement vector to another Model in the toroidal space. """
        delta_x, delta_y = other.position - self.position
        return Vector2D(
            x=(delta_x - copysign(self.frame.x, delta_x)) if (abs(2 * delta_x) > self.frame.x) else delta_x,
            y=(delta_y - copysign(self.frame.y, delta_y)) if (abs(2 * delta_y) > self.frame.y) else delta_y,
        )

    def get_image(self, other: Model) -> Vector2D:
        """
            Calculate the "virtual" position with relation to the other Model.
            Meaning, in the toroidal space, return the valid position that is
              closest to the other Model in Cartesian space.
        """
        return other.position + other.displacement(self)


class Boid(Model):
    """ Implementation of a Boid object. """
    size = 12  # Size of the Boid (in pixels)
    max_velocity = 200
    max_acceleration = 50

    def __init__(self, position: Vector2D, velocity: Vector2D, acceleration: Vector2D):
        """
        :param position: The coordinates of the Boid's position.
        :param velocity: The velocity vector of the Boid. (pixels per second).
        :param velocity: The acceleration vector of the Boid. (pixels per second per second).
        """
        super().__init__(position)
        self._velocity = velocity
        self._acceleration = acceleration

        self._reference = [
            Vector2D(-(self.size // 3), -(self.size // 3)),
            Vector2D(2 * (self.size // 3), 0),
            Vector2D(-(self.size // 3), (self.size // 3))
        ]

    def __repr__(self) -> str:
        return f"Boid(position={self.position}, velocity={self.velocity})"

    @property
    def angle(self) -> float:
        """ Getter for the angle of the velocity (Direction of movement). """
        return self.velocity.angle

    @property
    def acceleration(self) -> Vector2D:
        """ Getter for the acceleration vector. """
        return self._acceleration

    @acceleration.setter
    def acceleration(self, new_acceleration: Vector2D) -> None:
        """ Sets the acceleration, limited by the max acceleration. """
        if new_acceleration.magnitude > self.max_acceleration:
            new_acceleration = self.max_acceleration * new_acceleration.unit
        self._acceleration = new_acceleration

    @property
    def coordinates(self) -> Collection[float]:
        """ Returns the coordinates of the vertices that define the Boid. """
        angle = self.angle
        vectors = (vector.rotate(angle) + self.position for vector in self._reference)
        return [coord for vector in vectors for coord in vector]

    @property
    def velocity(self) -> Vector2D:
        """ Getter for the velocity vector. """
        return self._velocity

    @velocity.setter
    def velocity(self, new_velocity: Vector2D) -> None:
        """ Sets the velocity, limiting it by the maximum velocity. """
        if new_velocity.magnitude >= self.max_velocity:
            new_velocity = self.max_velocity * new_velocity.unit
        self._velocity = new_velocity

    def update(self, timestep: float) -> Boid:
        """
            Update the position of the Boid.
        :param timestep: The timestep over which the update the Boid's position.
        """
        self.velocity += (timestep * self.acceleration)
        self.position = (self.position + (timestep * self.velocity)) % self.frame
        return self

    @classmethod
    def random(cls):
        """ Constructs a Boid with random velocity and acceleration at the center of the frame. """
        return cls(
            position=(cls.frame / 2),
            velocity=(cls.max_velocity * Vector2D.random()),
            acceleration=Vector2D.random(),
        )


class Obstacle(Model):
    """ Implementation of an Obstacle. """
    size = 16

    def __init__(self, position: Vector2D):
        """
        :param position: The position of the Obstacle.
        """
        super().__init__(position)
        self.position = round(self.position)

        # Define the coordinates of this Obstacle since it will not move.
        lower_left = self.position + Vector2D(-self.size // 2, -self.size // 2)
        lower_right = self.position + Vector2D(self.size // 2, -self.size // 2)
        upper_right = self.position + Vector2D(self.size // 2, self.size // 2)
        upper_left = self.position + Vector2D(-self.size // 2, self.size // 2)
        self._coordinates = [*lower_left, *lower_right, *upper_right, *upper_left]

    @property
    def coordinates(self) -> Collection[int]:
        """ Returns the coordinates of the vertices that define the Obstacle. """
        return self._coordinates

    @classmethod
    def random(cls):
        """ Constructs a random Obstacle. """
        return cls(position=(Vector2D.random() * cls.frame))
