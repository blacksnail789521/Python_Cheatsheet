import argparse

parser = argparse.ArgumentParser(description='argparse_example',
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter) # Show default value

# Positional arguments (must given)
parser.add_argument('x', type=int, help='the base')
parser.add_argument('y', type=int, help='the exponent', choices = [0, 1, 2], default = 2)

# Optional arguments
parser.add_argument('-b', '--bias', type=int, help='the bias') # metavar = 'BIAS', default = None
parser.add_argument('--batch_size', type=int, help='the batch_size (not use in this case)', metavar='N')
parser.add_argument('--enable_bn', type=bool, help='enable batch normalization', default = True)
parser.add_argument('--model_name', type=str, help='model name', default = 'DNN')
parser.add_argument('-l1','--list1', nargs='+', help='set all elements in a list', default = ['0'])   # python arg.py -l1 1 2 3 4
parser.add_argument('-l2','--list2', action='append', help='add element one by one', default = ['0']) # python arg.py -l2 1 -l2 2 -l2 3 -l2 4


# Exclusive group
group = parser.add_mutually_exclusive_group()
group.add_argument('-v', '--verbose', action='store_true', help='increase output verbosity')
group.add_argument('-q', '--quiet', action='store_true') # action --> act as a flag (Don't use 'metavar')

'''----------------------------------------------------------------------'''
# Assume we use `python argparse_example.py 3 2`
args = parser.parse_args()
print(args)
'''
Namespace(batch_size=None, bias=None, list1=None, list2=None, quiet=False, verbose=False, x=3, y=2)
'''


# Use `python argparse_example.py -h` to get the help message.
'''
usage: argparse_example.py [-h] [-v | -q] [-b BIAS] [--batch_size N] x {0,1,2}

calculate X to the power of Y and add a bias

positional arguments:
  x                     the base
  {0,1,2}               the exponent

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase output verbosity
  -q, --quiet
  -b BIAS, --bias BIAS  the bias
  --batch_size N        the batch_size (not use in this case)
'''