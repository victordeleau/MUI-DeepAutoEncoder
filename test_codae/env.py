import sys
import os

RECURSIVE_LEVEL = 2

def recursively_add_parent_dir_to_path( level, path ):

    if level == 0:
        sys.path.append( path )
    else:
        recursively_add_parent_dir_to_path( level-1, os.path.dirname(path) )

recursively_add_parent_dir_to_path( RECURSIVE_LEVEL, os.path.abspath(__file__) )