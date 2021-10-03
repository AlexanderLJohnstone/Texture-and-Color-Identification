import pandas as panda

COLOR = "color"
TEXTURE = "texture"
IMAGE = "image"
ID = "id"

panda.options.mode.chained_assignment = None

'''
A set of functions related to data preprocessing
'''

def clean_bad_rows(df):
    '''
    This function finds and rows with missing values and removes them.
    :param df: The dataframe
    :return: A dataframe with no bad input.
    '''
    null_df = panda.isnull(df)
    bad_rows = []
    for each in null_df:
        for i in range(0, len(null_df[each])):
            if null_df[each].iloc[i]:
                bad_rows.append(i)
    bad_rows = set(bad_rows)
    df = remove_rows(df, bad_rows)
    return df


def fix_rows(df):
    '''
    This function finds rows with missing input and fills them with zeros
    :param df: The dataframe with missing values
    :return: A dataframe with no missing values
    '''
    null_df = panda.isnull(df)
    for each in df:
        for i in range(0, len(null_df[each])):
            if null_df[each].iloc[i]:
                df[each].iloc[i] = 0
    return df


def remove_rows(df, rows):
    '''
    The function removes rows from a dataframe
    :param df: The dataframe
    :param rows: The row to be removed
    :return: A re-indexed dataframe with the row removed
    '''
    df = df.drop(rows)
    df = df.reset_index(drop=True)
    return df


def remove_columns(df):
    '''
    This function removes specific features from a dataframe
    :param df: The dataframe
    :return: A dataframe with no output features or ID features
    '''
    if COLOR in df:
        del df[COLOR]
    if TEXTURE in df:
        del df[TEXTURE]
    del df[ID]
    del df[IMAGE]
    return df
