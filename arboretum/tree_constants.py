'''
tree_constants.py holds constants used in defining decision trees, 
such as column definitions.

Column definitions:
    0) Split feature, -1 if leaf
    1) Split threshold
    2) Node number of this node (nodes are numbered in pre-order).
    3) Node number of left child, -1 if leaf
    4) Node number of right child, -1 if leaf
    5) Number of data points in this node
    6) Value of this node
'''

# Position constants for the fields in the tree
FEATURE_COL = 0
THR_COL = 1
NODE_NUM_COL = 2
CHILD_LEFT_COL = 3
CHILD_RIGHT_COL = 4
CT_COL = 5
VAL_COL = 6

# FEATURE_COL value if no split/leaf
NO_FEATURE = -2

# THR_COL value if no split/leaf
NO_THR = -2.0

# CHILD_LEFT_COL and CHILD_RIGHT_COL values if leaf (no children)
NO_CHILD = -1
