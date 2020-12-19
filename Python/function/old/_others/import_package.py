import os
import sys

# Import tqdm.
package_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "package")
if package_path not in sys.path: sys.path.insert(0, package_path)
from tqdm import tqdm