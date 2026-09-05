# import dependencies
import os
from pathlib import Path
import pandas as pd

kpi_cols = ['5g_sa_data_session_success_rate','5g_sa_drb_accessibility',
            '5g_sa_drop_rate','5g_sa_ng_accessibility','5g_sa_rrc_accessibility']

def select_data(data_path: Path) -> pd.DataFrame:
    
    # create a list ot sort the data and a variable where we will sort the selected id's we will work on
    good_cells = None
    dfs= []
    
    # iterate over the data folder to select the files 
    for file in os.listdir(data_path):
        # read the file
        part= pd.read_parquet(os.path.join(data_path, file))
        
        # check if object ids has been selected  if not select them
        if good_cells is None:
            good_cells= part['object_id'].unique()[:100] # lock 100 object id from the first file
            
        # now we will get only those rows from the data where we will find those ids
        filtered= part[part['object_id'].isin(good_cells)]
        dfs.append(filtered)
    df= pd.concat(dfs, ignore_index=True)
    return df

# def tower_selection(df: pd.DataFrame, cell_id: str)-> pd.DataFrame:
#     s= df[df['object_id'] == cell_id].sort_values('start_time_utc')
#     if s.empty:
#         raise ValueError(f"cell_id {cell_id!r} not found in df")
#     s= s.set_index('start_time_utc')
#     return s

def regularize(cell_df, kpi_cols: list):
    # first we will calculate a consistant 15 mins timeline
    full_index= pd.date_range(cell_df.index.min(), cell_df.index.max(), freq='15min')
    # this reshape the dataframe and where we had missing timeline till will give nan to cols their
    result= cell_df.reindex(full_index)
    # since we are working only with one cell we will fill missing values with that cell id
    result['object_id']= cell_df['object_id'].iloc[0]
    # now iterate over all cols
    for col in kpi_cols:
        result[col + '_mask'] = result[col].notnull().astype(int)
        result[col] = result[col].fillna(0)
        
    return result

def regularize_all(df: pd.DataFrame, kpi_cols: list[str]) -> pd.DataFrame:
    """Apply regularize() to every cell in df, concat the results."""
    all_cells = []
    for cell_id, group in df.groupby('object_id'):
        group = group.sort_values('start_time_utc').set_index('start_time_utc')
        all_cells.append(regularize(group, kpi_cols))
    return pd.concat(all_cells)
    

if __name__ == "__main__": 
    df= select_data(data_path= 'data/unzip')
    print(f"Shape of the data is: {df.shape} and number of unique object id's are {df['object_id'].nunique()}")
    # print(df.isnull().sum())
    s= tower_selection(df= df, cell_id='10048NCn011')
    out = regularize(s, kpi_cols)
    print(out['object_id'].isnull().sum())   # should be 0 now
    # out.head(10)