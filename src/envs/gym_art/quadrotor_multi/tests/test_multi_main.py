import sys
import os

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取当前文件的目录
current_dir = os.path.dirname(current_file_path)

# 获取项目的根目录
project_root = os.path.dirname(current_dir)
project_root = os.path.dirname(project_root)
project_root = os.path.dirname(project_root)
# 将项目的根目录添加到 sys.path
sys.path.append(project_root)
print(sys.path)
from test_multi_env import TestMultiEnv
test_env=TestMultiEnv()
test_env.test_basic()
# test_env.test_basic()