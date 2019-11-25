""" Implementation of a 2D vector. """
from __future__ import annotations
from functools import cached_property
from math import atan2, cos, pi, sin, sqrt
from operator import add, mod, mul, sub, floordiv, truediv
from random import random
from typing import Any, Callable, Iterator, List, Union

Operator = Callable[[float, float], float]


class Vector2D:
    """ An implementation of a frozen vector with a simplified API. """

    def __init__(self, x: float, y: float):
        self._x = x
        self._y = y

    def __repr__(self) -> str:
        return f"Vector2D({self.x}, {self.y})"

    def __bool__(self) -> bool:
        return bool(self.magnitude)

    def __eq__(self, other) -> bool:
        return (abs(other.x - self.x) < 1e-12) and (abs(other.y - self.y) < 1e-12)

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @cached_property
    def angle(self) -> float:
        """ Returns the angle of the vector. """
        return atan2(self.y, self.x)

    @cached_property
    def magnitude(self) -> float:
        """ Returns the magnitude of the vector. """
        return sqrt((self.x * self.x) + (self.y * self.y))

    @cached_property
    def unit(self) -> Vector2D:
        """ Returns the corresponding unit vector. """
        magnitude = self.magnitude
        return Vector2D(x=(self.x / magnitude), y=(self.y / magnitude))

    def rotate(self, angle: float) -> Vector2D:
        """ Rotates the vector by the specified angle, in radians. """
        cos_angle = cos(angle)
        sin_angle = sin(angle)
        return Vector2D(
            x=((self.x * cos_angle) - (self.y * sin_angle)),
            y=((self.y * cos_angle) + (self.x * sin_angle)),
        )

    def resize(self, magnitude: float) -> Vector2D:
        """ Resizes the vector to a different magnitude. """
        if self.magnitude == 0:
            raise ZeroDivisionError(f"Cannot resize: {self}")
        return (magnitude / self.magnitude) * self

    def to_numpy(self) -> Any:
        """ Convert to a numpy vector. """
        try:
            import numpy as np
        except ImportError:
            raise ImportError("Cannot convert Vector2D to a numpy array. The numpy package is not installed.")
        return np.asarray([self.x, self.y])

    @classmethod
    def from_array(cls, array: List) -> Vector2D:
        """ Constructs a Vector2D from an index-able array. """
        return cls(x=array[0], y=array[1])

    @classmethod
    def from_radial(cls, angle: float, magnitude: float = 1.0) -> Vector2D:
        """ Constructs a Vector2D from an angle, in radians, and magnitude. """
        return cls(x=(magnitude * cos(angle)), y=(magnitude * sin(angle)))

    @classmethod
    def random(cls) -> Vector2D:
        """ Constructs a random vector within the unit disk. """
        return cls.from_radial(angle=(random() * 2 * pi), magnitude=random())

    # The implementation of the double-under arithmetic methods.

    def __abs__(self) -> Vector2D:
        return Vector2D(x=abs(self.x), y=abs(self.y))

    def __add__(self, other: Union[float, Vector2D]) -> Vector2D:
        return self._apply_operator(add, other)

    def __radd__(self, other: Union[float, Vector2D]) -> Vector2D:
        return self._apply_operator(add, other)

    def __mod__(self, other: Union[float, Vector2D]) -> Vector2D:
        return self._apply_operator(mod, other)

    def __mul__(self, other: Union[float, Vector2D]) -> Vector2D:
        return self._apply_operator(mul, other)

    def __rmul__(self, other: Union[float, Vector2D]) -> Vector2D:
        return self._apply_operator(mul, other)

    def __neg__(self) -> Vector2D:
        return Vector2D(x=(-self.x), y=(-self.y))

    def __round__(self, n=None):
        return Vector2D(x=round(self.x, n), y=round(self.y, n))

    def __sub__(self, other: Union[float, Vector2D]) -> Vector2D:
        return self._apply_operator(sub, other)

    def __rsub__(self, other: Union[float, Vector2D]) -> Vector2D:
        return self._apply_operator(sub, other)

    def __floordiv__(self, other: Union[float, Vector2D]) -> Vector2D:
        return self._apply_operator(floordiv, other)

    def __truediv__(self, other: Union[float, Vector2D]) -> Vector2D:
        return self._apply_operator(truediv, other)

    def _apply_operator(self, operator: Operator, other: Union[float, Vector2D]) -> Vector2D:
        """ Applies an operator between this vector and another object. """
        if isinstance(other, (int, float)):
            return Vector2D(x=operator(self.x, other), y=operator(self.y, other))
        elif isinstance(other, Vector2D):
            return Vector2D(x=operator(self.x, other.x), y=operator(self.y, other.y))
        raise NotImplementedError(f'Operation "{operator.__name__}" between Vectors and {type(other)}')


Zero = Vector2D(0, 0)
