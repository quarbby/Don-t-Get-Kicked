#!/usr/bin/python

'''
This file preprocesses the data
1. Fill up missing values (primitive solution)
- Nominal Columns: Filled with mode 
- Numeric Columns: Filled with median
2. Convert nominal columns to integers
- Standardize the numeric variables in the training split
'''


import pandas as pd 
from sklearn.cross_validation import train_test_split
    
nominal_cols = ['Auction', 'Make', 'Trim', 'TopThreeAmericanName', 'Model', 'SubModel', 'Color', 'Transmission', 'WheelType', 
                'PRIMEUNIT', 'AUCGUART', 'Nationality', 'Size', 'VNST']

num_cols = ['VehicleAge', 'WheelTypeID', 'VehOdo', 'AuctionAve', 'BYRNO', 'VNZIP1', 'IsOnlineSale', 'WarrantyCost']

global df

def preprocess(dataframe):
    global df
    
    df = dataframe
    # df = pd.read_csv('data/training.csv', header=0) 
    df = df.drop(['RefId'], axis=1)
    df = feature_engineering(df)    
    df = fill_missing_values(df)
    df = convert_nominal_cols(df)
    # df = standardize_dataframes(df)   
    return df 

def split_training_set(df):
    train_X, test_X, train_Y, test_Y = split_train_test()
    return train_X, test_X, train_Y, test_Y

def fill_missing_values(df):
    '''
    This function fills in the missing values
    Currently it's a simple solution
    - Mode for nominal columns
    - Median for numerical columns
    '''

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    
    for col in nominal_cols:
        mode = df[col].mode()[0]
        df[col] = df[col].fillna(mode)

    return df

def feature_engineering(df):
    '''
    This function performs the following feature feature_engineering
    - Drop PurchDate & VehYear since PurchDate = VehYear + VehicleAge
    - All the Average Prices are related => so just take the average 
    '''
    df = df.drop(['PurchDate'], axis=1)
    df = df.drop(['VehYear'], axis=1)
    df = merge_auction_ave_price(df)
    
    return df 

def convert_nominal_cols(df):
    '''
    This function converts nominal cols to integers
    Takes the unique values into an array and fills in an array index
    '''

    for col in nominal_cols:
        listOfItems = list(df[col].unique())
        df[col] = df[col].map(lambda x : listOfItems.index(x))

    return df

def split_train_test(df):
    '''
    This function splits the dataset into training and test sets
    into the 60-40 ratio
    Then it standardises dataframes
    '''
    temp = df.drop("IsBadBuy", axis=1)
    train_X, test_X, train_Y, test_Y = train_test_split(temp, df.IsBadBuy,
                                                        test_size=0.4, random_state=4531)
    train_X = pd.DataFrame(train_X, columns=df.columns[1:])
    test_X = pd.DataFrame(test_X, columns=df.columns[1:])
    train_Y = pd.DataFrame(train_Y, columns=['IsBadBuy'])
    test_Y = pd.DataFrame(test_Y, columns=['IsBadBuy'])
   
    standardize_dataframes(train_X)
    standardize_dataframes(test_X)

    return train_X, test_X, train_Y, test_Y

def standardize_dataframes(dataframe):
    '''
    This function standardises numerical dataframes
    Leaves out first two columns: IsBadBuy & PurchDate
    Note: Standardize reduces accuracy
    '''
    frames_to_standardize = dataframe.columns[range(2, len(dataframe.columns))]
    mean = dataframe[frames_to_standardize].mean()
    std = dataframe[frames_to_standardize].std()
    dataframe[frames_to_standardize] = (dataframe[frames_to_standardize] - mean) / std
    
    return dataframe
    
def merge_auction_ave_price(dataframe):
    '''
    This function takes the average of the 8 variables of the auction average prices
    '''
    auction_averages = ['MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice',
                        'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice',
                        'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice',
                        'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice']
                        
    
    dataframe['AuctionAve'] = sum(dataframe[ave] for ave in auction_averages) /len(auction_averages)
    dataframe = dataframe.drop(auction_averages, axis=1)
    
    return dataframe