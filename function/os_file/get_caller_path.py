import os
import inspect


# hierarchy = 0 means itself.
caller_path = os.path.abspath(inspect.stack()[0].filename)
print(caller_path)