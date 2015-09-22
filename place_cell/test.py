from path_integrating_cell import PathIntegratingCell
from visual_place_cell import VisualPlaceCell

import numpy as np

# c = PathIntegrateCell((9, 9))
#
# c.move(0, (0, 0))
# c.move(1, (1, 0))
# c.move(2, (0, 0))

c = VisualPlaceCell((9, 9))
c.move(0, np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]))
print(c.virtual_coordinate)
c.move(1, 0)
print(c.virtual_coordinate)
c.move(2, 0)
print(c.virtual_coordinate)
