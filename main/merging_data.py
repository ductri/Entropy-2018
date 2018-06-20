import pandas as pd


if __name__ == '__main__':
    ALL_DATASET = '/home/ductri/code/all_dataset/'
    ENTROPY_DATASET = ALL_DATASET + 'entropy_2018/'
    df_neg = pd.read_csv(ENTROPY_DATASET + 'Negative_train.csv', header=None)
    df_neg.columns = ['sentence']
    df_neg['sentiment'] = 'negative'

    df_neu = pd.read_csv(ENTROPY_DATASET + 'Neutral_train.csv', header=None)
    df_neu.columns = ['sentence']
    df_neu['sentiment'] = 'neutral'

    df_pos = pd.read_csv(ENTROPY_DATASET + 'Positive_train.csv', header=None, engine='python')
    df_pos.columns = ['sentence']
    df_pos['sentiment'] = 'positive'

    print('Total: {}'.format(df_neg.shape[0] + df_neu.shape[0] + df_pos.shape[0]))

    df = df_neg.append(df_neu)
    df = df.append(df_pos)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    print('Total dataset: {}'.format(df.shape[0]))
    df.to_csv(ENTROPY_DATASET + 'training_data.csv', index=None)
