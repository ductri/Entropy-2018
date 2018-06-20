import os
import pandas as pd


ALL_DATASET = '/home/ductri/code/all_dataset/'
ENTROPY_DATASET = os.path.join(ALL_DATASET, 'entropy_2018')


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(ENTROPY_DATASET, 'training_data.csv'))
