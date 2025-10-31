"""
Tile grid system for Red Light Green Light game.

Manages a 5x10 grid of tiles with lava floor mechanics.
"""

import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger

hailo_logger = get_logger(__name__)


class TileState(Enum):
    """
    Tile state enumeration.

    States:
        GREEN: Safe to step on.
        RED: Lava/forbidden tile.
    """

    GREEN = "green"
    RED = "red"


@dataclass
class Tile:
    """
    Represents a single tile in the grid.

    Attributes:
        row (int): Row index (0 = top/goal, 9 = bottom/start).
        col (int): Column index (0-4).
        state (TileState): Current state of the tile.
        proj_x (int): Top-left X coordinate in projector space.
        proj_y (int): Top-left Y coordinate in projector space.
        proj_width (int): Tile width in projector space.
        proj_height (int): Tile height in projector space.
    """

    row: int
    col: int
    state: TileState
    proj_x: int
    proj_y: int
    proj_width: int
    proj_height: int

    def contains_point(self, x: int, y: int) -> bool:
        """
        Check if a point is inside this tile.

        Args:
            x (int): X coordinate in projector space.
            y (int): Y coordinate in projector space.

        Returns:
            bool: True if point is inside tile.
        """
        return (
            self.proj_x <= x < self.proj_x + self.proj_width
            and self.proj_y <= y < self.proj_y + self.proj_height
        )

    def get_center(self) -> Tuple[int, int]:
        """
        Get the center point of the tile.

        Returns:
            Tuple[int, int]: (x, y) center coordinates in projector space.
        """
        return (
            self.proj_x + self.proj_width // 2,
            self.proj_y + self.proj_height // 2,
        )


class TileGrid:
    """
    Manages the game's tile grid system.

    Handles tile states, path generation, and player position tracking.

    Attributes:
        rows (int): Number of rows (default: 10).
        cols (int): Number of columns (default: 5).
        starting_row (int): Starting row for player (default: 9).
        goal_row (int): Goal row for player (default: 0).
        penalty_rows_back (int): Rows to move back on penalty.
        tiles (Dict[Tuple[int, int], Tile]): Grid of tiles indexed by (row, col).
        current_revealed_row (int): Highest row currently revealed.
        player_row (int): Current player row position.
        player_col (int): Current player column position.
    """

    def __init__(
        self,
        projector_width: int,
        projector_height: int,
        rows: int = 10,
        cols: int = 5,
        starting_row: int = 0,
        goal_row: int = 9,
        penalty_rows_back: int = 2,
        path_width: int = 1,
        obstacle_density: float = 0.3,
    ):
        """
        Initialize tile grid system.

        Args:
            projector_width (int): Projector width in pixels.
            projector_height (int): Projector height in pixels.
            rows (int): Number of rows.
            cols (int): Number of columns.
            starting_row (int): Starting row index.
            goal_row (int): Goal row index.
            penalty_rows_back (int): Rows to move back on penalty.
            path_width (int): Width of the safe path (tiles per row).
            obstacle_density (float): Density of red obstacle tiles (0-1).
        """
        self.rows = rows
        self.cols = cols
        self.starting_row = starting_row
        self.goal_row = goal_row
        self.penalty_rows_back = penalty_rows_back
        self.path_width = path_width
        self.obstacle_density = obstacle_density

        # Calculate tile dimensions with perspective
        self.tile_width = projector_width // cols
        
        # Calculate row heights with perspective scaling
        # Top rows (far from camera) are thinner, bottom rows (close) are thicker
        self.row_heights = self._calculate_perspective_row_heights(projector_height, rows)
        self.tile_height = sum(self.row_heights) // rows  # Average height for reference

        # Initialize grid
        self.tiles: Dict[Tuple[int, int], Tile] = {}
        self._initialize_grid(projector_width, projector_height)

        # Player position
        self.player_row = starting_row
        self.player_col = cols // 2  # Start in middle column

        # Track revealed rows (progressive revelation)
        self.current_revealed_row = starting_row

        # Generate path
        self.path: List[Tuple[int, int]] = []
        self._generate_path()

        hailo_logger.info(
            "TileGrid initialized: %dx%d grid, tile size: %dx%d pixels",
            cols,
            rows,
            self.tile_width,
            self.tile_height,
        )

    def _calculate_perspective_row_heights(self, total_height: int, num_rows: int) -> List[int]:
        """
        Calculate row heights with perspective scaling.
        Top rows (far from camera) are thinner, bottom rows (close) are thicker.

        Args:
            total_height (int): Total available height in pixels.
            num_rows (int): Number of rows.

        Returns:
            List[int]: Height in pixels for each row (index 0 = top/far, index n-1 = bottom/close).
        """
        # Use linear perspective scaling: row 0 (top/far) = 0.7x, row n-1 (bottom/close) = 1.3x
        # Scaling factor increases linearly from top to bottom
        scale_min = 0.7  # Top row scale
        scale_max = 1.3  # Bottom row scale
        
        # Calculate scale for each row
        scales = []
        for row in range(num_rows):
            # Linear interpolation from scale_min to scale_max
            t = row / max(1, num_rows - 1)  # 0.0 at top, 1.0 at bottom
            scale = scale_min + (scale_max - scale_min) * t
            scales.append(scale)
        
        # Calculate base height that makes total equal to total_height
        sum_scales = sum(scales)
        base_height = total_height / sum_scales
        
        # Calculate actual heights
        row_heights = [int(base_height * scale) for scale in scales]
        
        # Adjust last row to ensure exact total
        row_heights[-1] = total_height - sum(row_heights[:-1])
        
        return row_heights

    def _initialize_grid(self, projector_width: int, projector_height: int) -> None:
        """
        Initialize the grid with tiles using perspective-scaled row heights.
        All tiles start as RED (lava) - they will be revealed as GREEN progressively.

        Args:
            projector_width (int): Projector width in pixels.
            projector_height (int): Projector height in pixels.
        """
        # Calculate cumulative Y positions for each row
        cumulative_y = 0
        row_y_positions = []
        for row_height in self.row_heights:
            row_y_positions.append(cumulative_y)
            cumulative_y += row_height
        
        for row in range(self.rows):
            for col in range(self.cols):
                proj_x = col * self.tile_width
                proj_y = row_y_positions[row]
                row_height = self.row_heights[row]

                # Ensure tiles don't exceed projector bounds
                actual_width = min(self.tile_width, projector_width - proj_x)
                actual_height = min(row_height, projector_height - proj_y)

                tile = Tile(
                    row=row,
                    col=col,
                    state=TileState.RED,  # All tiles start as RED/lava
                    proj_x=proj_x,
                    proj_y=proj_y,
                    proj_width=actual_width,
                    proj_height=actual_height,
                )
                self.tiles[(row, col)] = tile

    def _generate_path(self) -> None:
        """
        Generate a winding path from start to goal.

        Creates a path of green tiles with red obstacle tiles.
        """
        self.path = []

        # Start in the middle of starting row
        current_col = self.cols // 2

        # Determine direction (forward if goal > start, backward if goal < start)
        if self.goal_row > self.starting_row:
            # Moving down (towards camera): start=0, goal=9
            row_range = range(self.starting_row, self.goal_row + 1)
        else:
            # Moving up (away from camera): start=9, goal=0
            row_range = range(self.starting_row, self.goal_row - 1, -1)

        # Generate path from start to goal
        for row in row_range:
            # Add current position to path
            self.path.append((row, current_col))

            # Randomly move left or right for next row (with bounds checking)
            if row != self.goal_row:
                direction = random.choice([-1, 0, 1])
                next_col = current_col + direction
                # Keep within bounds
                next_col = max(0, min(self.cols - 1, next_col))
                current_col = next_col

        # Mark starting row as revealed with path
        self._reveal_row(self.starting_row)
        
        # Count initial green tiles
        green_count = len([t for t in self.tiles.values() if t.state == TileState.GREEN])

        hailo_logger.info("Generated path with %d tiles, revealed starting row %d (%d green tiles)", 
                         len(self.path), self.starting_row, green_count)

    def _reveal_row(self, row: int) -> None:
        """
        Reveal a row by setting tile states.
        Tiles in the path become GREEN, others stay RED.

        Args:
            row (int): Row index to reveal.
        """
        # Find which columns in this row are part of the path
        path_cols = [col for r, col in self.path if r == row]

        for col in range(self.cols):
            if (row, col) in self.tiles:
                if col in path_cols:
                    # This is part of the safe path - make it green
                    self.tiles[(row, col)].state = TileState.GREEN
                # All other tiles remain RED (already initialized as RED)

        hailo_logger.debug("Revealed row %d with path at columns: %s", row, path_cols)

    def advance_path(self, rows_to_reveal: int = 1) -> None:
        """
        Reveal more rows of the path (called during green light phases).

        Args:
            rows_to_reveal (int): Number of rows to reveal.
        """
        for _ in range(rows_to_reveal):
            # Determine direction based on goal position
            if self.goal_row > self.starting_row:
                # Moving towards higher row numbers (towards camera)
                if self.current_revealed_row < self.goal_row:
                    self.current_revealed_row += 1
                    self._reveal_row(self.current_revealed_row)
                    hailo_logger.info("Advanced path to row %d", self.current_revealed_row)
                else:
                    hailo_logger.debug("Already at goal row, no more rows to reveal")
            else:
                # Moving towards lower row numbers (away from camera)
                if self.current_revealed_row > self.goal_row:
                    self.current_revealed_row -= 1
                    self._reveal_row(self.current_revealed_row)
                    hailo_logger.info("Advanced path to row %d", self.current_revealed_row)
                else:
                    hailo_logger.debug("Already at goal row, no more rows to reveal")

    def get_tile_at_position(self, proj_x: int, proj_y: int) -> Optional[Tile]:
        """
        Get the tile at a given projector position.

        Args:
            proj_x (int): X coordinate in projector space.
            proj_y (int): Y coordinate in projector space.

        Returns:
            Optional[Tile]: Tile at position, or None if out of bounds.
        """
        for tile in self.tiles.values():
            if tile.contains_point(proj_x, proj_y):
                return tile
        return None
    
    def is_position_on_board(self, proj_x: int, proj_y: int) -> bool:
        """
        Check if a position is within the game board area.

        Args:
            proj_x (int): X coordinate in projector space.
            proj_y (int): Y coordinate in projector space.

        Returns:
            bool: True if position is on any tile in the board.
        """
        return self.get_tile_at_position(proj_x, proj_y) is not None

    def get_player_current_tile(
        self, leg_position_proj: Optional[Tuple[int, int]]
    ) -> Optional[Tile]:
        """
        Get the tile where the player is currently standing.

        Args:
            leg_position_proj (Optional[Tuple[int, int]]): Leg position in projector coordinates.

        Returns:
            Optional[Tile]: Current tile, or None if position not available.
        """
        if leg_position_proj is None:
            return None

        proj_x, proj_y = leg_position_proj
        tile = self.get_tile_at_position(proj_x, proj_y)

        # Update player position if tile found
        if tile:
            self.player_row = tile.row
            self.player_col = tile.col
            hailo_logger.debug("Player at tile (%d, %d)", tile.row, tile.col)

        return tile

    def is_tile_safe(self, tile: Tile) -> bool:
        """
        Check if a tile is safe to step on.

        Args:
            tile (Tile): Tile to check.

        Returns:
            bool: True if tile is green (safe).
        """
        return tile.state == TileState.GREEN

    def is_tile_dangerous(self, tile: Tile) -> bool:
        """
        Check if a tile is dangerous (lava).

        Args:
            tile (Tile): Tile to check.

        Returns:
            bool: True if tile is red (lava).
        """
        return tile.state == TileState.RED

    def apply_penalty(self) -> Tuple[int, int]:
        """
        Move player back as penalty.

        Returns:
            Tuple[int, int]: New (row, col) position.
        """
        # Move player back towards starting row
        old_row = self.player_row

        # Determine direction based on goal position
        if self.goal_row > self.starting_row:
            # Goal is at higher row number, so penalty moves to lower row number
            self.player_row = max(self.starting_row, self.player_row - self.penalty_rows_back)
        else:
            # Goal is at lower row number, so penalty moves to higher row number
            self.player_row = min(self.starting_row, self.player_row + self.penalty_rows_back)

        # Keep player on the path
        path_cols_at_new_row = [col for r, col in self.path if r == self.player_row]
        if path_cols_at_new_row:
            # Find closest path column to current position
            self.player_col = min(
                path_cols_at_new_row, key=lambda c: abs(c - self.player_col)
            )

        hailo_logger.info(
            "Applied penalty: moved from row %d to row %d, col %d",
            old_row,
            self.player_row,
            self.player_col,
        )

        return (self.player_row, self.player_col)

    def get_player_position(self) -> Tuple[int, int]:
        """
        Get current player position.

        Returns:
            Tuple[int, int]: (row, col) position.
        """
        return (self.player_row, self.player_col)

    def has_reached_goal(self) -> bool:
        """
        Check if player has reached the goal.

        Returns:
            bool: True if player is at goal row.
        """
        return self.player_row == self.goal_row

    def reset(self) -> None:
        """
        Reset the grid for a new game.
        """
        # Reset all tiles to RED
        for tile in self.tiles.values():
            tile.state = TileState.RED

        # Reset player position
        self.player_row = self.starting_row
        self.player_col = self.cols // 2

        # Reset revealed row
        self.current_revealed_row = self.starting_row

        # Generate new path
        self._generate_path()

        hailo_logger.info("TileGrid reset for new game")

    def get_all_tiles(self) -> List[Tile]:
        """
        Get all tiles in the grid.

        Returns:
            List[Tile]: List of all tiles.
        """
        return list(self.tiles.values())

    def get_tiles_by_state(self, state: TileState) -> List[Tile]:
        """
        Get all tiles with a specific state.

        Args:
            state (TileState): State to filter by.

        Returns:
            List[Tile]: List of tiles with the specified state.
        """
        return [tile for tile in self.tiles.values() if tile.state == state]

    def get_player_tile(self) -> Tile:
        """
        Get the tile at the player's current position.

        Returns:
            Tile: Tile at player position.
        """
        return self.tiles[(self.player_row, self.player_col)]
    
    def remove_tiles_behind_player(self) -> None:
        """
        Remove (set to RED) all green tiles behind the player's current position.
        Only affects tiles that are behind the player in the direction away from goal.
        Keeps the player's current row green to ensure they always have a tile to stand on.
        """
        if self.goal_row > self.starting_row:
            # Moving down (towards higher row numbers)
            # Remove tiles with row < player_row (but not current row)
            rows_to_remove = [r for r in range(self.starting_row, self.player_row)]
        else:
            # Moving up (towards lower row numbers)
            # Remove tiles with row > player_row (but not current row)
            rows_to_remove = [r for r in range(self.player_row + 1, self.starting_row + 1)]
        
        tiles_removed = 0
        for row in rows_to_remove:
            for col in range(self.cols):
                if (row, col) in self.tiles and self.tiles[(row, col)].state == TileState.GREEN:
                    self.tiles[(row, col)].state = TileState.RED
                    tiles_removed += 1
        
        green_count_after = len(self.get_tiles_by_state(TileState.GREEN))
        
        if tiles_removed > 0:
            hailo_logger.info(
                "Removed %d green tiles behind player (row %d), %d green tiles remain",
                tiles_removed,
                self.player_row,
                green_count_after
            )
        
        # Safety check: ensure at least current revealed row has green tiles
        if green_count_after == 0:
            hailo_logger.error("ERROR: No green tiles remaining! Re-revealing current row %d", self.current_revealed_row)
            self._reveal_row(self.current_revealed_row)

