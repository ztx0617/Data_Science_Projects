import sys
import pandas as pd
from sqlalchemy import create_engine

def load_clean_data(messages_filepath, categories_filepath):
    
    # read files
    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
    
    messages = pd.read_csv(messages_filepath).drop_duplicates(subset = 'id')
    categories = pd.read_csv(categories_filepath).drop_duplicates(subset = 'id')
    
    #deal with categories
    print('Cleaning data...')
    
    categories_expand = categories['categories'].str.split(';', expand=True)
    category_colnames = categories_expand.loc[0,:].apply(lambda x: x.split('-')[0]).tolist()
    categories_expand = categories_expand.applymap(lambda x: x.split('-')[1]).astype('int')
    categories_expand.columns = category_colnames
    categories_expand = categories_expand[categories_expand <= 1].dropna(how = 'any', axis = 0)
    categories = pd.concat([categories['id'], categories_expand], axis = 1)
    
    #merge messages and categories based on id
    df = pd.merge(messages, categories, on = 'id', how = 'right')
    
    return df
    



def save_data(df, database_filename):
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        df = load_clean_data(messages_filepath, categories_filepath)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()