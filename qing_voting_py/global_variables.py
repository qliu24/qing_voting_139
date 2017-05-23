import os
import sys

# 
g_project_root_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

g_data_folder = os.path.join(g_project_root_dir, 'data')
g_src_folder = os.path.join(g_project_root_dir, 'src')
g_cache_folder = os.path.join(g_project_root_dir, 'cache')

sys.path.insert(0, os.path.join(g_src_folder, 'slim'))
