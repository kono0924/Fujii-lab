import multiprocessing
import time
import os


def run(i):
    time.sleep(5)
    print(time.time())


def main():
    with multiprocessing.Pool(processes=500) as pool:
        r = pool.map(run, range(500)) # ←変更


if __name__ == "__main__":
    t_s = time.time()
    main()
    t_e = time.time()
    print(t_e-t_s)
