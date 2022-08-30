import numpy as np

from PIL import Image, ImageDraw


class Game2048:
    
    """
    Basic game logic for N-sized 2048 game
    """
    
    def __init__(self, grid_size=4, max_pow=13):
        
        """
        Args:
        
            grid_size: the number of tiles forming the
            playing square grid. Default: 4
            
            max_pow: the maximum tile value (2**max_pow)
            after reaching which the episode terminates.
            Default: 13 (2**13=8192)
        """
        
        self.mat = np.zeros(shape=(grid_size, grid_size))
        self.score = 0
        self.grid_size = grid_size
        self.max_value = 2**max_pow
        
        # create assets for rendering 
        # calculate cell size
        size = len(str(2**max_pow))*6
        self.cell_width = size
        self.cell_height = 15 # Hardcoded

        # create cell values grid
        val_grid = [0] + [2**p for p in range(1, max_pow+1)]

        # create red channel shift grid
        c_step = int(np.floor(250 / len(val_grid)))
        r_shifts = [c_step*t for t in range(len(val_grid))]

        # create green channel shift grid
        N = len(val_grid)
        g_shifts = [np.round(-(1000/N**2)*t**2+(1000/N)*t) for t in range(N)]

        # initialize assets
        self.assets_dict = {}

        # generate assets (small cell pictures with numbers)
        # To use them later as building blocks for rendering
        for cell_val, r_shift, g_shift in zip(val_grid, r_shifts, g_shifts):

            cell_val_str = str(cell_val)
            pixel_width = len(cell_val_str) * 6
            x_start = size // 2 - pixel_width // 2

            img = Image.new('RGB', 
                (size, self.cell_height), 
                color = (250-int(r_shift), 
                         250-int(g_shift), 
                         255)
            )

            d = ImageDraw.Draw(img)

            if cell_val > 0:
                d.text((x_start, 2), cell_val_str, fill=(0,0,0))

            img_arr = np.array(img)

            self.assets_dict[cell_val] = img_arr
            
            del d, img
        
    @property
    def empty_cells(self):
        """
        Array of empty grid cells coordinates
        """
        is_zero = self.mat == 0
        return np.transpose(np.nonzero(is_zero))
    
    @property
    def filled_cells(self):
        """
        Array of filled grid cells coordinates
        
        Sorted in order to have upper left points first
        """
        filled_arr = np.transpose(np.nonzero(self.mat))
        filled_arr = sorted( 
            filled_arr, 
            key=lambda v: (v[0], v[1])
        )
        return np.array(filled_arr)
    
    @property
    def left_shift_mat(self):
        """
        Shifted grid to detect similar neighbors
        """
        return np.concatenate(
            [np.zeros(self.mat.shape[0])[:, np.newaxis], 
             self.mat[:, :-1]], 
            axis=-1
        )

    def add_new_2(self):
        """
        Randomly add '2' on an empty cell of the grid
        """
        cell_ind = np.random.randint(
            0, self.empty_cells.shape[0]
        )
        
        r, c = self.empty_cells[cell_ind]
        
        self.mat[r][c] = 2
        
    def left_compress_point(self, index: list) -> bool:
        """
        Move specific point to the left if possible
        (No other points on its path)
        !Note: Does not merge any cells
        """
        r, c = index
        value = self.mat[r][c]
        change = False
        if c != 0: # already not at left side
            row_empty = self.empty_cells[self.empty_cells[:, 0] == r, 1]
            c_left = c
            while c_left - 1 in row_empty:
                c_left -= 1
            if c_left < c:
                self.mat[r][c_left] = value
                self.mat[r][c] = 0.
                change = True
        return change
        
    def left_compress(self) -> bool:
        """
        Move all filled grid cells to the left
        """
        change = False
        
        for point in self.filled_cells:
            change_i = self.left_compress_point(index=point)
            if change_i:
                change = True
                
        return change
    
    def left_merge(self) -> bool:
        """
        Merge neighbour cells to the left if equal
        """
        change = False
        
        # detect equal neighbour cells
        equalities = np.logical_and(
            self.mat == self.left_shift_mat,
            self.mat > 0
        )
        
        # Avoid duplicate merge
        # E.g. three same cells in a row
        non_duplicate = equalities.cumsum(axis=-1) % 2 == 1

        equalities = np.logical_and(equalities,
                                    non_duplicate)

        equal_neighbors = np.transpose(np.nonzero(equalities))

        for en in equal_neighbors:
            r, c = en
            self.mat[r][c-1] *= 2
            self.mat[r][c] = 0
            self.score += self.mat[r][c-1]
            change = True
            
        return change
    
    def reverse(self):
        self.mat = self.mat[:, ::-1]
        
    def transpose(self):
        self.mat = self.mat.T
        
    def reset(self):
        self.score = 0 
        self.mat = np.zeros(
            shape=(self.grid_size, 
                   self.grid_size)
        )
            
    def move_left(self) -> bool:
        
        change1 = self.left_compress()
        
        change2 = self.left_merge()
        
        change3 = self.left_compress()
        
        return any([change1, change2, change3])
    
    def move_right(self) -> bool:
        
        self.reverse()
        
        change = self.move_left()
        
        self.reverse()
        
        return change
    
    def move_up(self) -> bool:
        
        self.transpose()
        
        change = self.move_left()
        
        self.transpose()
        
        return change
    
    def move_down(self) -> bool:
        
        self.transpose()
        
        change = self.move_right()
        
        self.transpose()
        
        return change
    
    def is_game_over(self) -> bool:
        
        game_over = False
        
        # max value reached?
        max_val_reached = self.mat.max() >= self.max_value
        if max_val_reached:
            game_over = True
        
        # if no empty cells and max value not reached
        if len(self.empty_cells) == 0 and not max_val_reached:
            
            # log current game state:
            log_mat = self.mat.copy()
            log_score = self.score
            
            # start checking if further action 
            # would change anything
            
            # left
            change_left = self.move_left()
            # return to past state
            self.mat = log_mat
            self.score = log_score
            
            # right
            change_right = self.move_right()
            # return to past state
            self.mat = log_mat
            self.score = log_score
            
            # up
            change_up = self.move_up()
            # return to past state
            self.mat = log_mat
            self.score = log_score
            
            # down
            change_down = self.move_down()
            
            # return to past state
            self.mat = log_mat
            self.score = log_score
            
            game_over_vec = [
                not change_left,
                not change_right,
                not change_up,
                not change_down
            ]
            
            # if all actions lead to no change - game is over
            game_over = all(game_over_vec)
            
        return game_over
    
    def render(self, add_score=False) -> np.ndarray:
        
        """
        Render game's grid as an `np.array`
        using pre-defined assets as building blocks
        (see `__init__()` method)
        
        Args:
        
            add_score: `bool` adds score information
            tile to the top of the rendered grid frame
            
        Returns:
        
            frame: `np.ndarray` with a size of
            (self.grid_size*self.cell_height,
             self.grid_size*self.cell_width,
             3)
             with self.cell_height being hardcoded to 15.
             Dtype: np.uint8
        """
        
        row_vec = []
        
        for row in self.mat:
            row_arr = np.concatenate(
                [self.assets_dict[v] for v in row], axis=1
            )
            row_vec.append(row_arr)
            
        frame = np.concatenate(row_vec, axis=0)
        
        if add_score:
            
            score = Image.new('RGB', 
                (self.cell_width*self.grid_size, self.cell_height), 
                 color = (40, 40, 250)
            )

            d = ImageDraw.Draw(score)

            score_text = f'Score: {int(self.score)}'
            d.text((2, 2), score_text, fill=(0,0,0))
            
            frame = np.concatenate([score, frame], axis=0)
        
        return frame.copy().astype(np.uint8)
    