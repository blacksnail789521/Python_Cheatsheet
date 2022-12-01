import timeit
import inspect
import time


def test(n_iters):
    
    for _ in range(n_iters):
        time.sleep(0.001)


def main():
    
    SETUP_CODE = inspect.cleandoc( \
        """
        from __main__ import test
        
        n_iters = 3
        """)
    
    TEST_CODE = inspect.cleandoc( \
        """
        result = test(n_iters)
        """)
    
    print("----------------------------")
    print("Run TEST_CODE with 10 times:")
    print(timeit.timeit(setup = SETUP_CODE, stmt = TEST_CODE, number = 10))
    
    print("----------------------------")
    print("Run TEST_CODE with 100 times: (Run 3 rounds.)")
    print(timeit.repeat(setup = SETUP_CODE, stmt = TEST_CODE, number = 100, repeat = 3))


if __name__ == "__main__":
    
    main()