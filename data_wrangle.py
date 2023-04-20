#Database linker CSV->DB DB->DF
import sqlite3
import os
import pandas as pd
import numpy as np
import json
from prettytable import PrettyTable as pt
from prettytable import ALL
import preprocess
from collections import Counter

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

        #Track sample IDs that are dropped bc they are flagged with outliers
        self.dead_list=[] #IDs of samples
    
    def __del__(self):
        '''Automaticallly close the database connection when the object is destroyed (program terminated).'''
        self.conn.close()
            
    # def query(self,stmt,fetch=True):
    #     #Original
    #     if fetch==True:
    #         res = self.cursor.execute(f'{stmt}')
    #         return self.cursor.fetchall()
    #     else:
    #         res = self.cursor.execute(f'{stmt}')
    #         return None
    def query(self, stmt, params=None, fetch=True):
        if params is not None:
            self.cursor.execute(stmt, params)
        else:
            self.cursor.execute(stmt)
        if fetch:
            return self.cursor.fetchall()
    # ###EXAMPLE USEAGE###
    # db = Database('mydatabase.db')
    # stmt = "SELECT name, age FROM users WHERE gender=? AND occupation=?"
    # params = ('male', 'engineer')
    # results = db.query(stmt, params=params)
    # for row in results:
    #     print(row['name'], row['age'])
    # #Why use?
    # 1)Security: By using parameters, you can protect your code from SQL injection attacks. SQL injection is a common hacking technique where an attacker tries to insert malicious SQL code into a query by exploiting vulnerabilities in the input validation.
    # 2)Performance: Using parameters can improve query performance by allowing the database engine to optimize the execution plan for the query.
    # 3)Reusability: By using parameters, you can reuse the same query with different input values. This can save you time and effort in writing and maintaining multiple similar queries.

    def columns(self):
        '''Returns a list of column names in the current database table.'''
        columns=self.query(f'PRAGMA table_info({self.table})')
        cols=[col[1] for col in columns]
        return cols
    
    def data_info(self, *incols):
        # if len (incols)==0:
            # return self.table_info()
        if not incols:
            return self.table_info()
        else:
            if self.condition:
                stmt=f'SELECT COUNT(*) FROM {self.table} {self.condition}'
            else:
                stmt=f'SELECT COUNT(*) FROM {self.table}'
            print(f'Number of samples selected: {self.query(stmt)[0][0]}')
            return self.col_info(*incols)

    def col_info(self, *incols):
        """
        Generates a list of PrettyTable objects, each containing information about the unique values and their frequency for 
        the columns specified in *incols. If no arguments are passed, an error message is printed and an empty list is returned.
        If a column name specified in *incols is not a valid column name in the table, an error message is printed and that 
        column is skipped.

        Args:
            *incols: A variable-length argument list of column names to retrieve information for.

        Returns:
            A list of PrettyTable objects, each containing information about the unique values and their frequency for the 
            specified columns.
        """
        if not incols:
            print(f"Error col_info(): Please specify at least one column name.")
            return []
        
        # Convert list of columns to a set
        all_cols = set(self.columns())

        col_tables=[]
        if all_cols.issuperset(incols):
            for col in incols:
                if self.condition:
                    uniques = self.query(
                        f"SELECT DISTINCT {col}, COUNT(*) FROM {self.table} {self.condition} GROUP BY {col}"
                        )
                else:
                    uniques = self.query(
                        f"SELECT DISTINCT {col}, COUNT(*) FROM {self.table} GROUP BY {col}"
                        )
                col_table=pt(title = f'{col}')
                col_table.field_names = ['Value', 'Frequency']
                for u in uniques:
                    col_table.add_row(u)
                col_tables.append(col_table)                

        else:
            # Convert set of columns to a list and sort it for consistency
            missing_cols = sorted(list(all_cols - set(incols)))
            print(f'Error\nupdate this to remove invalid column and cotinue on')
            # print(f"Error: thethe following columns are valid: {', '.join(missing_cols)}")
        return col_tables

    def update(self,col_name,old_value,new_value):
        #Original
        stmt=f"UPDATE {self.table} SET {col_name} = '{new_value}' WHERE {col_name} = '{old_value}'"
        self.cursor.execute(stmt) 
        self.conn.commit()

    def update(self, col_name, old_value, new_value):
        #Try this method with parameterized queries 
        stmt = f"UPDATE ? SET ? = ? WHERE ? = ?"
        args = (self.table, col_name, new_value, col_name, old_value)
        self.cursor.execute(stmt, args)
        self.conn.commit()
        
    def table_info(self):
        '''
        Generates a summary of the specified table, including number of rows and columns,
        and details on each column such as column ID, name, unique values, data type, 
        and whether the column is nullable or has a default value.

        Returns:
        None
        '''
        #Initialize data table
        table=pt(header=False, hrules=ALL, vrules=ALL)

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
        columns = self.query(f'PRAGMA table_info({self.table})')
        all_cols = pt()
        all_cols.title = 'Summary of Columns for Entire DataTable'
        all_cols.field_names = ['colID', 'name', 'unique_vals_per_col', 'type', 'notnull', 'dflt_value', 'primary_key']
        for c in columns:
            unique_vals_per_column = self.query(f'SELECT COUNT(DISTINCT {c[1]}) FROM {self.table}')
            col_info = list(c)
            col_info.insert(2, unique_vals_per_column[0][0])
            all_cols.add_row(col_info)
        print(all_cols)
    
    def insert_row(self, *col_val: tuple):
        """Insert a new row with the given values into the database table.

        Args:
            *col_val: Variable length argument list of tuples, where each tuple contains
                a column name and a value to insert into that column for the new row.

        Returns:
            None.

        Raises:
            sqlite3.Error: If there is an error executing the INSERT statement.
        """
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
        """
        Retrieve data from the database table associated with this `database` instance.
        
        Arguments:
        - x_loc (int): The index of the column containing the X data in the table. Defaults to 2.
        - y_loc (int): The index of the column containing the Y data in the table. Defaults to 3.
        
        Returns:
        - x (list): A list of floats representing the X values from all rows in the table.
        - y (list): A list of floats representing the Y values from all rows in the table.
        
        """
        self.cursor.execute(f'SELECT * FROM {self.table}')
        res=self.cursor.fetchall()
        
        for item in res:
            xdic=json.loads(item[x_loc].decode('utf-8'))
            ydic=json.loads(item[y_loc].decode('utf-8'))
            x=[float(val) for key,val in xdic.items()]
            y=[float(val) for key,val in ydic.items()]
    
    def blob2list(self,byte_data):
        '''Pass blob/byte data to np.array'''
        # t=byte_data.decode('utf-8')
        t=np.frombuffer(byte_data,dtype=np.int64)
        return t
    
    def raman_shifts(self, len_range=[600,700]):
        '''
        Retrieve Raman shift data from the database and perform outlier detection.

        Args:
        - len_range (list of int): Optional. A list specifying the expected range of lengths
        for Raman shift data. Samples with lengths outside of this range will be removed
        and their sample IDs will be added to the dead_list.

        Returns:
        - mean (pandas.Series): A series containing the mean values of each Raman shift.

        Raises:
        - None.

        This method retrieves Raman shift data from the database table, filters out samples
        whose lengths are outside the expected range, and performs outlier detection on the
        remaining samples. The dead_list instance variable is updated with the sample IDs
        of the filtered out samples. The method returns a series containing the mean values
        of each Raman shift. Outliers are printed to the console.

        '''
        #Grab data from DB
        if self.condition:
            rawres=self.query(f'SELECT {self.id}, {self.x} FROM {self.table} {self.condition}')
        else:
            rawres=self.query(f'SELECT {self.id}, {self.x} FROM {self.table}')
       
        #Filter rawres by removing any samples whos ID has been added to dead_list
        res=[tup for tup in rawres if tup[0] not in self.dead_list]

        #Convert byte data in raw tuples to list data
        all_tup_list=[(tup[0],self.blob2list(tup[1])) for tup in res]
        tup_list=[]
        #Data outside of the set range will be flagged to be dropped *self.dead_list*
        #Filter by length -> helps to prevent mismatched data from being compared
        dropped_id=[]
        table=pt(title='Samples Removed: length outside of expected range')
        table.field_names=['Sample ID','Length']
        for tup in all_tup_list:
            if len(tup[1]) >= len_range[0] and len(tup[1]) <= len_range[1]:
                tup_list.append(tup)
            else: #EDIT NEEDED also remove these tup[0]'s from the intensity dataset
                table.add_row([tup[0],len(tup[1])])
                dropped_id.append(tup[0])
        print(table)

        #Convert tuples list to dictionary
        data_dict={t[0]:t[1] for t in tup_list} #(id,[xs values])

        ###Compare xs, look for outliers
        #Create a DF from the dictionary
        df=pd.DataFrame.from_dict(data_dict,orient='index') #Columns = rs index, index=sampleID (int)
        #Change a value for testing purposes
        # df.iloc[0,0]=1000 #Change a value for testing purposes
        
        #Calc per Column (Raman Shift)
        mean=df.mean() #Series of means per Raman Shift
        
        std=df.std() #Series of stds per Raman Shift
        #Define a threshold for IDing outliers
        threshold=2.0
        #ID the outliers for each column
        outliers=(np.abs(df-mean) >  threshold*std) #bool DF, TRUE = original value is outlier
        
        #Display information on outliers
        #...might need to add if any() statement

        #Display Outliers
        series=outliers.stack() #Convert outliers (df) into a series w/ 2 index locations
        truth=series[series==True] #Store True values (and loc) as a pd.series
        outlier_loc=truth.index.tolist() #List of locs in df where an outlier is present
        ids_with_outlier=list(set([l[0] for l in outlier_loc]))
        # #Alt code: counts = Counter(ids_with_outlier) #(sample id, frequency)        
        table=pt()
        table.title='Outliers'
        if len(ids_with_outlier)==0:
            table.field_names=['There are no outliers in the remaining data!']
        else:
            table.field_names=['Sample ID','# Outliers','Outlier Values']
            for sample in ids_with_outlier:
                tups=[loc for loc in outlier_loc if loc[0]==sample]
                values=[df.loc[loc] for loc in tups]
                table.add_row([sample,len(values),values])
        print(table)

        #Add sample ids that have been id'ed to drop to the 'dead list'
        for sample in dropped_id:
            if sample not in self.dead_list:
                self.dead_list.append(sample)

        return mean
    
    def intens(self):
        """
        Returns a Pandas DataFrame of the preprocessed Raman spectra intensity data, with rows
        corresponding to the sample ID and columns corresponding to the Raman shift index.

        Returns:
        --------
        pandas.DataFrame:
            DataFrame with rows corresponding to sample ID and columns corresponding to Raman shift index.
        """
        #Grab data from DB
        if self.condition:
            rawres=self.query(f'SELECT {self.id}, {self.y} FROM {self.table} {self.condition}')
        else:
            rawres=self.query(f'SELECT {self.id}, {self.y} FROM {self.table}')
        #Filter rawres by removing any samples whos ID has been added to dead_list
        res=[tup for tup in rawres if tup[0] not in self.dead_list]
        #Convert byte data to list data, then DF
        tup_list=[(tup[0],self.blob2list(tup[1])) for tup in res]
        prepro_list=[(tup[0], preprocess.process(tup[1])) for tup in tup_list]        #Preprocess the data
        data_dict={t[0]:t[1] for t in prepro_list}
        df=pd.DataFrame.from_dict(data_dict,orient='index')
        #DF with rows = sample id, columns = raman shift index
        return df
    
    def names(self, name_cols=['filename','frame']):
        """
        Returns a dictionary of sample names (or any other specified metadata)
        for each sample ID in the database table.

        Args:
        - table_col_id (str): the name of the column in the database table containing the unique
                              identifier for each sample. Defaults to 'id'.
        - name_cols (list of str): a list of column names in the database table containing the
                                   sample names or other metadata. Defaults to ['filename', 'frame',
                                   'general_loc'].

        Returns:
        - A dictionary with sample IDs as keys, and a tuple of sample names/metadata as values.

        Example usage:
        db.names()  # Returns a dictionary of all sample names/metadata for all samples in the database
        db.names(name_cols=['filename'])  # Returns a dictionary of filenames for all samples in the database
        """
        names=(', ').join(name_cols)
        #Get data
        if self.condition:
            rawres=self.query(f'SELECT {self.id}, {names} FROM {self.table} {self.condition}')
        else:
            rawres=rawres=self.query(f'SELECT {self.id}, {names} FROM {self.table}')
        #Turn result into a dictionary with sample ID as the key
        id_dic={tup[0]:tup[1:] for tup in rawres}
        return id_dic
