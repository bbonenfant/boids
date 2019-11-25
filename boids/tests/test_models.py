""" Tests for models.py """
import pytest

from ..models import Boid
from ..vector2d import Vector2D


class TestBoid:
    """ Tests for the Boid class. """
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

    @pytest.mark.parametrize(
        'position, destination, expected_displacement',
        [(Vector2D(5, 5), Vector2D(10, 15), Vector2D(5, 10)),
         (Vector2D(5, 5), Vector2D(30, 15), Vector2D(-15, 10)),
         (Vector2D(5, 5), Vector2D(10, 25), Vector2D(5, -10)),
         (Vector2D(5, 5), Vector2D(30, 25), Vector2D(-15, -10))]
    )
    def test_displacement(self, position, destination, expected_displacement):
        """ Test that the displacement calculation is performed correctly in a toroidal space. """
        # Arrange
        first_boid = self.construct_boid(position=position)
        second_boid = self.construct_boid(position=destination)

        # Act
        result = first_boid.displacement(second_boid)

        # Assert
        assert result == expected_displacement

    @pytest.mark.parametrize(
        'position, destination, expected_distance',
        [(Vector2D(0, 0), Vector2D(3, 4), 5),
         (Vector2D(-3, -4) % frame, Vector2D(3, 4), 10),
         (Vector2D(-8, 0) % frame, Vector2D(8, 0), 16),
         (Vector2D(0, 7), Vector2D(0, -7) % frame, 14)]
    )
    def test_distance(self, position, destination, expected_distance):
        """ Test that the distance calculation is performed correctly. """
        # Arrange
        first_boid = self.construct_boid(position=position)
        second_boid = self.construct_boid(position=destination)

        # Act
        result = first_boid.distance(second_boid)

        # Assert
        assert result == expected_distance

    @pytest.mark.parametrize(
        'position, velocity, expected_position',
        [(Vector2D(5, 10), Vector2D(10, 0), Vector2D(15, 10) % frame),
         (Vector2D(5, 10), Vector2D(0, 15), Vector2D(5, 25) % frame),
         (Vector2D(5, 10), Vector2D(-10, 0), Vector2D(-5, 10) % frame),
         (Vector2D(5, 10), Vector2D(0, -15), Vector2D(5, -5) % frame)]
    )
    def test_update_position(self, position, velocity, expected_position):
        """ Test that the update tot eh Boid's position is performed as expected. """
        # Arrange
        timestep = 1
        boid = self.construct_boid(position=position, velocity=velocity)

        # Act
        boid.update(timestep)

        # Assert
        assert boid.position == expected_position
