import pandas as pd
import numpy as np
import numpy.linalg
import argparse
import time
import os

from option import opt
from programs import subleq_TF
from utils import create_folder

def main():
    
    # make folder for saving the outputs
    create_folder(opt.output_folder)
    create_folder(os.path.join(opt.output_folder, opt.mode))

    # run programs
    if opt.mode.lower() == 'subleq':
        subleq_TF(opt)
     


if __name__ == "__main__":
    start_total_time = time.time()
    main()
    end_total_time = time.time()
    print('Total ptime: {:.2f}'.format(end_total_time - start_total_time))