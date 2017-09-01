'''
tree_constants.py holds constants used in defining decision trees, 
such as column definitions.

Column definitions:
    0) Split feature, -1 if leaf
    1) Split threshold
    2) Number of data points in this node
    3) Number of positives in this node
    4) Node number of this node (nodes are numbered in pre-order).
    5) Node number of left child, -1 if leaf
    6) Node number of right child, -1 if leaf
'''

# Position constants for the fields in the tree
FEATURE_COL = 0
THR_COL = 1
CT_COL = 2
POS_COL = 3
NODE_NUM_COL = 4
CHILD_LEFT_COL = 5
CHILD_RIGHT_COL = 6

# FEATURE_COL value if no split/leaf
NO_FEATURE = -2

# THR_COL value if no split/leaf
NO_THR = -2.0

# CHILD_LEFT_COL and CHILD_RIGHT_COL values if leaf (no children)
NO_CHILD = -1
