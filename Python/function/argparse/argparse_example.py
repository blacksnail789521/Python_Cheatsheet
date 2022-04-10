import argparse

parser = argparse.ArgumentParser(description='calculate X to the power of Y and add a bias',
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter) # Show default value

# Exclusive group
group = parser.add_mutually_exclusive_group()
group.add_argument('-v', '--verbose', action='store_true', help='increase output verbosity')
group.add_argument('-q', '--quiet', action='store_true') # action --> act as a flag (Don't use 'metavar')

# Positional arguments
parser.add_argument('x', type=int, help='the base')
parser.add_argument('y', type=int, help='the exponent', choices = [0, 1, 2], default = 2)

# Optional arguments
parser.add_argument('-b', '--bias', type=int, help='the bias') # metavar = 'BIAS'
parser.add_argument('--batch_size', type=int, help='the batch_size (not use in this case)', metavar='N')


'''----------------------------------------------------------------------'''

args = parser.parse_args()
answer = args.x**args.y+args.bias

if args.quiet:
    print(answer)
elif args.verbose:
    print(f'{args.x} to the power {args.y} adding {args.bias} equals {answer}')
else:
    print(f'{args.x}^{args.y}+{args.bias} == {answer}')

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