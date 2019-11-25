""" Tests for vector2d.py """
from math import degrees, radians
from operator import abs, add, mod, mul, neg, sub, floordiv, truediv
from random import uniform
from typing import Callable, List, Union

import pytest

from ..vector2d import Vector2D

# Type Definitions.
Operand = Union[int, float, Vector2D, List]
Operator = Callable[[Operand, Operand], Operand]
UnaryOperator = Callable[[Operand], Operand]


@pytest.fixture(name='commutative_operator', params=(add, mul))
def _commutative_operator(request) -> Operator:
    yield request.param


@pytest.fixture(name='non_commutative_operator', params=(mod, sub, floordiv, truediv))
def _non_commutative_operator(request) -> Operator:
    yield request.param


@pytest.fixture(name='unary_operator', params=(abs, neg))
def _unary_operator(request) -> UnaryOperator:
    yield request.param


@pytest.fixture(name='scalar')
def _scalar() -> float:
    """ Return a random number on -10 and 10. """
    return uniform(-10, 10)


@pytest.fixture(name='vector')
def _vector() -> Vector2D:
    """ Return a random Vector2D. """
    return Vector2D.random()


@pytest.fixture(name='other_vector')
def _other_vector() -> Vector2D:
    """ Return a second random Vector2D. """
    return Vector2D.random()


class TestVector2D:
    """ Tests for the Vector2D class. """

    @staticmethod
    def test_from_radial(vector):
        """ Test that the construction of a vector from its angle and magnitude is correctly performed. """
        # Arrange
        angle = vector.angle
        magnitude = vector.magnitude

        # Act
        result = Vector2D.from_radial(angle, magnitude)

        # Assert
        assert result == vector

    @staticmethod
    def test_numpy_serialization(vector):
        """ Test the serialization and deserialization to and from numpy is correctly performed. """
        # Arrange

        # Act
        new_vector = Vector2D.from_array(vector.to_numpy())

        # Assert
        assert new_vector == vector

    @staticmethod
    def test_commutative_operators(vector, scalar, commutative_operator):
        """ Test that the commutative operators on a vector and a scalar are performed as expected. """
        # Arrange
        expected = Vector2D.from_array(commutative_operator(vector.to_numpy(), scalar))

        # Act
        result_1 = commutative_operator(vector, scalar)
        result_2 = commutative_operator(scalar, vector)

        # Assert
        assert result_1 == result_2 == expected

    @staticmethod
    def test_commutative_vector_operators(vector, other_vector, commutative_operator):
        """ Test that the commutative operators on a vector and another vector are performed as expected. """
        # Arrange
        expected = Vector2D.from_array(commutative_operator(vector.to_numpy(), other_vector.to_numpy()))

        # Act
        result_1 = commutative_operator(vector, other_vector)
        result_2 = commutative_operator(other_vector, vector)

        # Assert
        assert result_1 == result_2 == expected

    @staticmethod
    def test_non_commutative_operators(vector, scalar, non_commutative_operator):
        """ Test that the non-commutative operators on a vector and a scalar are performed as expected. """
        # Arrange
        expected = Vector2D.from_array(non_commutative_operator(vector.to_numpy(), scalar))

        # Act
        result = non_commutative_operator(vector, scalar)

        # Assert
        assert result == expected

    @staticmethod
    def test_non_commutative_vector_operators(vector, other_vector, non_commutative_operator):
        """ Test that the non-commutative operators on a vector and another vector are performed as expected. """
        # Arrange
        expected = Vector2D.from_array(non_commutative_operator(vector.to_numpy(), other_vector.to_numpy()))

        # Act
        result = non_commutative_operator(vector, other_vector)

        # Assert
        assert result == expected

    @staticmethod
    def test_unary_vector_operators(vector, unary_operator):
        """ Test that the unary operators on a vector are performed as expected. """
        # Arrange
        expected = Vector2D.from_array(unary_operator(vector.to_numpy()))

        # Act
        result = unary_operator(vector)

        # Assert
        assert result == expected

    @staticmethod
    def test_iteration(vector):
        """ Test that the iteration double-underscore method is implemented as expected. """
        # Arrange
        iterable_vector = iter(vector)

        # Act & Assert
        assert next(iterable_vector) == vector.x
        assert next(iterable_vector) == vector.y

    @staticmethod
    @pytest.mark.parametrize(
        'x, y, expected_angle',
        [(0, 0, 0), (1, 0, 0), (1, 1, 45), (0, 1, 90), (-1, 1, 135),
         (-1, 0, 180), (-1, -1, -135), (0, -1, -90), (1, -1, -45)]
    )
    def test_angle(x, y, expected_angle):
        """ Test that the angle property returns expected values. """
        # Arrange
        vector = Vector2D(x, y)

        # Act
        result = vector.angle

        # Assert
        assert degrees(result) == expected_angle

    @staticmethod
    @pytest.mark.parametrize(
        'x, y, expected_magnitude',
        [(0, 0, 0), (3, 4, 5), (-3, 4, 5), (3, -4, 5), (-3, -4, 5),
         (5, 12, 13), (8, 15, 17), (7, 24, 25), (12, 35, 37)]
    )
    def test_angle(x, y, expected_magnitude):
        """ Test that the magnitude property returns expected values. """
        # Arrange
        vector = Vector2D(x, y)

        # Act
        result = vector.magnitude

        # Assert
        assert result == expected_magnitude

    @staticmethod
    def test_unit(scalar):
        """ Test that the unit property returns the expected unit vector. """
        # Arrange
        expected_vector = Vector2D.from_radial(scalar, magnitude=1.0)
        scaled_vector = abs(scalar) * expected_vector

        # Act
        result = scaled_vector.unit

        # Assert
        assert result == expected_vector

    @staticmethod
    @pytest.mark.parametrize(
        'x, y, angle, expected_x, expected_y',
        [(1, 0, 0, 1, 0), (0, 1, 90, -1, 0), (-1, 0, 180, 1, 0), (1, 0, -90, 0, -1)]
    )
    def test_rotate(x, y, angle, expected_x, expected_y, scalar):
        """ Test that the rotation method is implemented as expected. """
        # Arrange
        vector = scalar * Vector2D(x, y)
        expected_vector = scalar * Vector2D(expected_x, expected_y)

        # Act
        result = vector.rotate(radians(angle))

        # Assert
        assert result == expected_vector
