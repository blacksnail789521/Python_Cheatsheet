import shutil
import os

# Remove all old figures.
output_path = os.path.join("output")
if os.path.exists( os.path.join(output_path, "figure") ):
    shutil.rmtree( os.path.join(output_path, "figure") )