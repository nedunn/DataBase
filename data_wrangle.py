#Database linker CSV->DB DB->DF
import sqlite3
import os
import pandas as pd
import numpy as np
import json
from prettytable import PrettyTable as pt
from prettytable import ALL
import preprocess

class DataTable:
    def __init__(self, db_path,table_name=None):
        #Connect to DB
        self.conn=sqlite3.connect(db_path)
        self.cursor=self.conn.cursor()

        #Object must be initialized with a dataTABLE
        if table_name is None:
            stmt=('A table must be selected for the Object to be initialized.\nAvailable tables:')
            tables=self.query('SELECT name FROM sqlite_master WHERE type=\'table\';')

            stmt=(f'A table must be selected for the Object to be initialized.\nAvailable tables ({len(tables)}):')
            # [stmt+=f'\n\t{t}' for t in res]
            for table in tables:
                stmt+=f'\n\t{table[0]}'
            print(stmt)
        self.table=table_name
        
        self.x='raman_shift' #Database table name where 'x' values are stored
        self.y='original_intensity'
        self.id='id' #Database table column name where spectra specific 'id' valuesare stored

        #Optional condition for Query searchs
        self.condition=None
            
    def query(self,stmt,fetch=True):
        if fetch==True:
            res = self.cursor.execute(f'{stmt}')
            return self.cursor.fetchall()
        else:
            res = self.cursor.execute(f'{stmt}')
            return None

    def columns(self):
        '''Returns a list of column names in the current database table.'''
        columns=self.query(f'PRAGMA table_info({self.table})')
        cols=[col[1] for col in columns]
        return cols

    def col_info(self, *incols):
        """
        Returns information about the columns in the current database table.

        Args:
            *incols (str): Optional; column names to display unique values for.

        Returns:
            Union[prettytable.PrettyTable, Tuple[prettytable.PrettyTable, List[prettytable.PrettyTable]]]: 
            If no column names are passed as arguments, returns a `prettytable.PrettyTable` object 
            displaying column information such as name, data type, whether it allows NULL values, 
            whether it is part of the primary key, and the number of unique values in each column.
            If one or more column names are passed as arguments, returns a tuple of two `prettytable.PrettyTable` 
            objects. The first one displays column information, and the second one displays unique values 
            for each of the specified columns.

        This method uses the `PRAGMA table_info` SQL statement to retrieve information
        about the columns in the current table, including name, data type, whether it allows 
        NULL values, and whether it is part of the primary key. It then executes a 
        SELECT COUNT(DISTINCT col_name) statement for each column to determine the number 
        of unique values in that column. Finally, it creates a `prettytable.PrettyTable` 
        object to display this information in a readable format.

        If one or more column names are passed as arguments, the method also displays the 
        unique values in those columns using a SELECT DISTINCT statement.
        """
        columns = self.query(f'PRAGMA table_info({self.table})')
        all_cols = pt()
        all_cols.title = 'Summary of Columns for Entire DataTable'
        all_cols.field_names = ['colID', 'name', 'unique_vals_per_col', 'type', 'notnull', 'dflt_value', 'primary_key']
        for c in columns:
            unique_vals_per_column = self.query(f'SELECT COUNT(DISTINCT {c[1]}) FROM {self.table}')
            col_info = list(c)
            col_info.insert(2, unique_vals_per_column[0][0])
            all_cols.add_row(col_info)

        if len(incols) > 0:
            col_tables=[]
            for col in incols:
                if self.condition:
                    uniques = self.query(
                        f"SELECT DISTINCT {col}, COUNT(*) FROM {self.table} {self.condition} GROUP BY {col}"
                    )
                else:
                    uniques = self.query(
                        f"SELECT DISTINCT {col}, COUNT(*) FROM {self.table} GROUP BY {col}"
                    )
                col_table = pt()
                col_table.title = f'{col}'
                col_table.field_names = ['Value', 'Frequency']
                for u in uniques:
                    col_table.add_row(u)
                    col_tables.append(col_table)
            return all_cols, col_tables
        else:
            return all_cols

    def update(self,col_name,old_value,new_value):
        stmt=f"UPDATE {self.table} SET {col_name} = '{new_value}' WHERE {col_name} = '{old_value}'"
        self.cursor.execute(stmt)
        self.conn.commit()
        
    def table_info(self):
        #Initialize data table
        table=pt(header=False, hrules=ALL, vrules=ALL)

        #Information to grab based on Condtion
        if self.condition:
            stmt=f'SELECT COUNT(*) FROM {self.table} {self.condition}'
            table.title=f'Summary of Dataset - With Conditions'
        else:
            stmt=f'SELECT COUNT(*) FROM {self.table}'
            table.title=f'Summary for \'{self.table}\' Table'
        
        #Get Row information
        self.cursor.execute(stmt)
        rows=self.cursor.fetchone() #num_rows = f'Number of rows: {len(rows)}'?

        #Display info
        table.add_row([
            'Table Name',f'{self.table}','|\n|',
            'Number of samples\n(rows)',rows[0],'|\n|',
            'Number of columns',f'{len(self.columns())}'])
        #Show Data table dimensions
        print(table) 
        #Show column details
        print(self.col_info())
    
    def data_info(self):
        '''Future update will combine with table_info()'''
        if self.condition:
            self.col_info(*self.columns())
        else:
            self.table_info()
    
    def insert_row(self, *col_val: tuple):
        #Column portion of the statement
        collist=[cv[0] for cv in col_val]
        colstr=(', ').join(collist)

        #Value portion of statement
        vallist=[cv[1] for cv in col_val]

        # '?' portion of statement
        n=(len(col_val))
        qs=['?' for item in (list(range(n)))]
        qstr=(', ').join(qs)

        #Build the statement
        stmt=f'INSERT INTO {self.table} ({colstr}) VALUES ({qstr})'
        self.cursor.execute(stmt,vallist)

        #Commit - dont forget to commit
        self.conn.commit()

    def data_from_db(self, x_loc=2, y_loc=3):
        self.cursor.execute(f'SELECT * FROM {self.table}')
        res=self.cursor.fetchall()
        
        for item in res:
            xdic=json.loads(item[x_loc].decode('utf-8'))
            ydic=json.loads(item[y_loc].decode('utf-8'))
            x=[float(val) for key,val in xdic.items()]
            y=[float(val) for key,val in ydic.items()]

    def data_as_df(self,query):
        #untested
        df=pd.read_sql_query(f'{query} FROM {self.table};',self.conn)
        return df
    
    def blob2list(self,byte_data):
        '''Pass blob/byte data to np.array'''
        
        # t=byte_data.decode('utf-8')

        t=np.frombuffer(byte_data,dtype=np.int64)
        return t
    
    def raman_shifts(self):#, table_col_name='raman_shift', table_col_id='id'):
        '''table_col_name = name of column in data table were x data is.
            table_col_id = name of column in data table where unique identifier for each individual spectra is.
            ***Add qualities for query search***
        '''
        #Grab data from DB
        if self.condition:
            rawres=self.query(f'SELECT {self.id}, {self.x} FROM {self.table} {self.condition}')
        else:
            rawres=self.query(f'SELECT {self.id}, {self.x} FROM {self.table}')
        #Convert byte data in raw tuples to list data
        tup_list=[(tup[0],self.blob2list(tup[1])) for tup in rawres]

        #Convert tuples list to dictionary
        data_dict={t[0]:t[1] for t in tup_list} #(id,[xs values])

        ###Compare xs, look for outliers
        #Create a DF from the dictionary
        df=pd.DataFrame.from_dict(data_dict,orient='index') #Columns = rs index, index=sampleID (int)
        # print(df)
        #Calc per Column (Raman Shift)
        mean=df.mean() #Series of means per Raman Shift
        std=df.std() #Series of stds per Raman Shift
        #Define a threshold for IDing outliers
        threshold=2.0
        #ID the outliers for each column
        outliers=(np.abs(df-mean) >  threshold*std) #bool DF, TRUE = original value is outlier
        
        #Display information on outliers
        #...might need to add if any() statement

        #Get columns that contain at least 1 outlier
        


        # for col in outliers.columns: #Check each column of outliers df (bool where TRUE = outlier)
        #     table=pt(['Sample ID', 'Outlier (Raman Shift) Value'])
        #     table.title=f'Outlier(s) in column **{col}**'
        #     boolcol=outliers[col]
        #     inds=boolcol.index[boolcol]
        #     for i in inds:
        #         outlier_value=(df.loc[i,col])
        #         table.add_row([i,outlier_value])
        #     print(table)
    
        return mean
    
    def intens(self, table_col_name='original_intensity', table_col_id='id'):
        #Grab data from DB
        if self.condition:
            rawres=self.query(f'SELECT {self.id}, {self.y} FROM {self.table} {self.condition}')
        else:
            rawres=self.query(f'SELECT {self.id}, {self.y} FROM {self.table}')
        #Convert byte data to list data, then DF
        tup_list=[(tup[0],self.blob2list(tup[1])) for tup in rawres]
        prepro_list=[(tup[0], preprocess.process(tup[1])) for tup in tup_list]        #Preprocess the data
        data_dict={t[0]:t[1] for t in prepro_list}
        df=pd.DataFrame.from_dict(data_dict,orient='index')
        return df
    
    def names(self, table_col_id='id', name_cols=['filename','frame','general_loc']):
        #Returns a dictionary with sample id as keys
        names=(', ').join(name_cols)
        #Get data
        if self.condition:
            rawres=self.query(f'SELECT {self.id}, {names} FROM {self.table} {self.condition}')
        else:
            rawres=rawres=self.query(f'SELECT {self.id}, {names} FROM {self.table}')
        #Turn result into a dictionary with sample ID as the key
        id_dic={tup[0]:tup[1:] for tup in rawres}
        return id_dic
    


    # def columns(self,*incols): #V1x
    #     """
    #     Returns information about the columns in the current database table.

    #     Parameters:
    #         *incols (str): Optional; column names to display unique values for.

    #     Returns:
    #         cols (list): List of strings representing the column names.
    #         table (prettytable.PrettyTable): Table object displaying column info.

    #     This method uses the `PRAGMA table_info` SQL statement to retrieve information
    #     about the columns in the current table, including name, data type, whether it 
    #     allows NULL values, and whether it is part of the primary key. It then executes 
    #     a SELECT COUNT(DISTINCT col_name) statement for each column to determine the number 
    #     of unique values in that column. Finally, it creates a `prettytable.PrettyTable` 
    #     object to display this information in a readable format.

    #     If one or more column names are passed as arguments, the method also displays the 
    #     unique values in those columns using a SELECT DISTINCT statement.
    #     """
        
    #     columns=self.query(f'PRAGMA table_info({self.table})')
    #     cols=[col[1] for col in columns]
        
    #     #Make a table of column info
    #     all_cols=pt() #initialize table
    #     all_cols.title='Column Information'
    #     all_cols.field_names=['colID','name','unique_vals_per_col','type','notnull','dflt_value','primary_key']
    #     for c in columns:
    #         # c = 'tuple of column info'
    #         unique_vals_per_column=self.query(f'SELECT COUNT(DISTINCT {c[1]}) FROM {self.table}')
    #         col_info=list(c)
    #         # col_info.append(unique_vals_per_column[0][0])
    #         col_info.insert(2,unique_vals_per_column[0][0])
    #         all_cols.add_row(col_info)

    #     if len(incols)>0:
            

    #         #Run through each column 'incol'        
    #         for col in incols:
    #             if self.condition: #If conditions has been set:
    #                 uniques = self.query(
    #                     f"SELECT DISTINCT {col}, COUNT(*) FROM {self.table} {self.condition} GROUP BY {col}"
    #                     )
    #             else:
    #                 uniques = self.query(
    #                     f"SELECT DISTINCT {col}, COUNT(*) FROM {self.table} GROUP BY {col}"
    #                     )
    #             #Create a new table for each input column
    #             col_table=pt()
    #             col_table.title=f'{col}'
    #             col_table.field_names=['Value', 'Frequency']
    #             for u in uniques:
    #                 col_table.add_row(u)
    #             print(col_table)

    #     return cols, all_cols
        