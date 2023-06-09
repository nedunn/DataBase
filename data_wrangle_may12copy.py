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

def unique_count(lst):
    result={}
    for item in lst:
        if item in result:
            result[item] += 1
        else:
            result[item]=1
    return result

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
        self.all_ids=self.get_ids()
        # self.ids=self.query(f'SELECT {self.id} FROM {self.table}')

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
    '''###EXAMPLE USEAGE###
    db = Database('mydatabase.db')
    stmt = "SELECT name, age FROM users WHERE gender=? AND occupation=?"
    params = ('male', 'engineer')
    results = db.query(stmt, params=params)
    for row in results:
        print(row['name'], row['age'])
    #Why use?
    1)Security: By using parameters, you can protect your code from SQL injection attacks. SQL injection is a common hacking technique where an attacker tries to insert malicious SQL code into a query by exploiting vulnerabilities in the input validation.
    2)Performance: Using parameters can improve query performance by allowing the database engine to optimize the execution plan for the query.
    3)Reusability: By using parameters, you can reuse the same query with different input values. This can save you time and effort in writing and maintaining multiple similar queries.'''

    def get_ids(self):
        ids_badformat=self.query(f'SELECT {self.id} FROM {self.table}')
        ids=[item[0] for item in ids_badformat]
        return ids

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
        
    def find_replace(self, find_value, new_value, target_col, find_col=None):
        if not find_col:
            find_col=target_col
        if find_value.lower() == 'null':
            stmt = f"UPDATE {self.table} SET {target_col} = ? WHERE {find_col} IS NULL"
            values=(new_value)
        else:
            stmt=f'UPDATE {self.table} SET {target_col} = ? WHERE {find_col} = ?'
            values=(new_value, find_value)
        
        try:
            # Retrieve list of INITIAL values in the target_col column
            before_update = [row[0] for row in self.cursor.execute(f'SELECT {target_col} FROM {self.table}')] 
            before_update=unique_count(before_update)

            #Execute the update
            self.cursor.execute(stmt,values)
            print(f'{self.cursor.rowcount} rows updated.')
            self.conn.commit()
            
            #Get list of UPDATED values
            after_update = [row[0] for row in self.cursor.execute(f'SELECT {target_col} FROM {self.table}')]
            after_update=unique_count(after_update)
        
            print(f'before: {before_update}\nafter: {after_update}')

            self.conn.rollback()
            self.conn.commit()

        except:
            print('AHHHH ERROR')
 




        # try:
        #     self.cursor.execute(stmt,values)
        #     print(f'{self.cursor.rowcount} rows updated.')

        # self.cursor.execute(stmt, values)
        # self.conn.commit()

        # # Retrieve the list of values in the target_col column before the update
        # before_update = [row[0] for row in self.cursor.execute(f'SELECT {target_col} FROM {self.table}')]
        # self.cursor.execute(stmt, values)
        # self.conn.commit()
        # # Retrieve the list of values in the target_col column after the update
        # after_update = [row[0] for row in self.cursor.execute(f'SELECT {target_col} FROM {self.table}')]


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
        try: # byte data look like: b'{"0":282,"1":283,"2":285,"3":2... (dop specific)
            asstr=byte_data.decode('utf-8')
            dic=json.loads(asstr)
            clean=np.array(list(dic.values()))
            # print('decoding for do_pac')
        
        except: #byte data look like: b'\xf0\x01\x00\x00\
            clean=np.frombuffer(byte_data,dtype=np.int64)
            print('prob dopac')
        
        return clean

    def match_index(self,df):
        unique_rows=df.drop_duplicates()
        unique_rows.reset_index(inplace=True, drop=True)
        for row in unique_rows.iterrows():
            match_idx=df.index[df.apply(lambda x: x.equals(row[1]), axis=1)].tolist()
            print(f'row {row[0]}\nmatchindex {len(match_idx)}\n')
        return unique_rows

    # def match_index(self, df, selected_row=None):
    #     if selected_row is not None:
    #         unique_rows = df[df.iloc[:, 0] == selected_row]
    #         print('got here')
    #     else:
    #         unique_rows = df.drop_duplicates()
            
    #     unique_rows.reset_index(inplace=True, drop=True)
        
    #     for row in unique_rows.iterrows():
    #         match_idx = df.index[df.apply(lambda x: x.equals(row[1]), axis=1)].tolist()
    #         print(f'row {row[0]}\nmatchindex {len(match_idx)}\n')
            
    #     return unique_rows 

    
    def raman_shifts(self,  **kwargs):
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
        #Data outside of the set range will be flagged to be dropped *self.dead_list*
        #Filter by length -> helps to prevent mismatched data from being compared
        table=pt(title='Samples Removed: length outside of expected range')
        table.field_names=['Sample ID','Length']

        #Find average length x input per sample
        #May combine with "Find each x type" section of function later. I am busy ATM
        lengths=([len(tup[1]) for tup in all_tup_list])
        unique_lengths=(unique_count(lengths))
        dropped_id=[]
        if len(unique_lengths) > 1:
            tup_list=[]
            length_to_use=max(unique_lengths, key=unique_lengths.get) # get most common length
            for tup in all_tup_list: 
                if len(tup[1])==length_to_use: #keep this data
                    tup_list.append(tup)
                else: #remove this data
                    table.add_row([tup[0],len(tup[1])])
                    
            for tup in all_tup_list:
                table.add_row([tup[0],len(tup[1])])
                dropped_id.append(tup[0])
                dropped_id.append(tup[0])
        else: #all of the x lists are the same length!
            tup_list=all_tup_list
        
        ##Should be able to delete this section by the time i forget about it
        # tup_list=[]   
        # for tup in all_tup_list:
        #     if len(tup[1]) >= len_range[0] and len(tup[1]) <= len_range[1]:
        #         print('if')
        #         tup_list.append(tup)
        #     else: #EDIT NEEDED also remove these tup[0]'s from the intensity dataset
        #         print('else')
        #         table.add_row([tup[0],len(tup[1])])
                
        # print(table) **EDIT dont print if empty

        #Convert tuples list to dictionary
        data_dict={t[0]:t[1] for t in tup_list} #(id,[xs values])

        ###Compare xs, look for outliers
        #Create a DF from the dictionary
        df=pd.DataFrame.from_dict(data_dict,orient='index') #Columns = rs index, index=sampleID (int)
        #Change a value for testing purposes
        # df.iloc[0,0]=1000 #Change a value for testing purposes
        
        #Find each 'x' type
        unique_x_sets=set(map(tuple, df.values))
        print(unique_x_sets )
        #ALT code = unique_cols = df.T.drop_duplicates()
        print(f'There are {len(unique_x_sets)} found in the dataset.')
        xlist=list(unique_x_sets)
        if len(unique_x_sets) > 1:
            pass
        table=pt(title='Xs Detected')
        table.field_names=['index', 'length', 'head', 'tail']

        for x in xlist:
            ind=xlist.index(x)
            table.add_row([ind,len(x), x[:3],x[-3:]])
            mask = np.all(df==x, axis=1) #checks each row (sample) for whether or not the 'x' matches the row's x
            match_idx=np.where(mask)[0].tolist()
        
        #Alternative
        # self.match_index(df, **kwargs)
        #**NEXT select which x to use, drop all 'ys' that are not supported


        # unique_df=df.drop_duplicates()
        # for i in range(0,len(unique_df)):
        #     anx=np.array(unique_df.iloc[i,:])
        #     # if np.array_equal()


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
        for sample in ids_with_outlier:
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
        tup_list=[(tup[0],self.blob2list(tup[1])) for tup in rawres]

        # # test=[len(tup[1]) for tup in tup_list]
        # # print(list(set(test)))
        prepro_list=[(tup[0], preprocess.process(tup[1])) for tup in tup_list]        #Preprocess the data
        data_dict={t[0]:t[1] for t in prepro_list}
        df=pd.DataFrame.from_dict(data_dict,orient='index')
        #DF with rows = sample id, columns = raman shift index
        return df
    
    def apply_snv(self,df): #where each sample is a ROW
        row_means=df.mean(axis=1) #Calculate the mean for each row
        df_centered=df.sub(row_means, axis=0) #Subtract the row mean from each rs in the row
        row_std=np.sqrt(df_centered.pow(2).sum(axis=1))/(df.shape[1]-1) #Calculate teh STD per row
        result=df_centered.div(row_std,axis=0) #Divide each element in the row by the row standard deviation
        return result
    
    def apply_snv(self,df):
        res=np.zeros_like(df)
        
    
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
        res=[tup for tup in rawres if tup[0] not in self.dead_list]
        id_dic={tup[0]:tup[1:] for tup in res}
        return id_dic
    
    def average_lists(self,*lists):
        '''Example:
        l1  =  [3,4,3]
        l2  =  [3,5,2]
        l3  =  [3,6,1]
        result=[3,5,2]
        '''
        stack=np.vstack(lists)
        return np.mean(stack,axis=0)
    
    def get_spectra(self, *spec_ids):
        """
        Returns a list of tuples containing the spectral data for each specified ID.

        Args:
            *spec_ids: Variable-length argument list of integer IDs or tuples of integer IDs to be
            averaged. If a tuple of IDs is provided, the spectra for each ID will be averaged and
            the resulting tuple will be included in the returned list.

        Returns:
            A list of tuples, where each tuple contains the ID (or tuple of IDs), x-values, and y-values
            for the specified spectra.

        Example:
            To get the spectral data for IDs 1, 2, and (3, 4) and average the data for IDs 3 and 4:
            >>> data = obj.get_spectra(1, 2, (3, 4))
            >>> print(data)
            [
                (1, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
                (2, [1.0, 2.0, 3.0], [8.0, 7.0, 6.0]),
                ('3, 4', [2.0, 3.0, 5.0], [2.0, 1.5, 3.5])
            ]
        """

        #*EDIT* check that id is valid *update: will be easier to impliment now that we have 'all_ids' argument
        # ids=', '.join(str(id) for id in spec_ids)
        # stmt=f'SELECT {self.id}, {self.x}, {self.y} FROM {self.table} WHERE id IN ({ids})'
        # rawres=self.query(stmt)
        data=[]

        #If no IDs given, return all data that is not excluded bc of condition
        if not spec_ids:
            ids_to_grab=[id for id in self.all_ids if id not in self.dead_list]
            for id in ids_to_grab:
                stmt=f'SELECT {self.id}, {self.x}, {self.y} FROM {self.table} WHERE id == {id}'
                rawres=self.query(stmt)
                single_spec_tup=(rawres[0][0], self.blob2list(rawres[0][1]), self.blob2list(rawres[0][2]))
                data.append(single_spec_tup)

        #If specific IDs given:
        if spec_ids: #**EDIT DONT FORGET TO TEST, SINCE FUNCTION HAS BEEN EDITIED
            for item in spec_ids:            
                if isinstance(item,tuple): #ids placed within a tuple will be averaged
                    ind=spec_ids.index(item)
                    single_spec_tup=self.ave_spectra(item)
                    data.append(single_spec_tup)
                elif isinstance(item,int):
                    stmt=f'SELECT {self.id}, {self.x}, {self.y} FROM {self.table} WHERE id == {item}'
                    rawres=self.query(stmt)
                    single_spec_tup=(rawres[0][0], self.blob2list(rawres[0][1]), self.blob2list(rawres[0][2]))
                    data.append(single_spec_tup)
                elif isinstance(item,list):
                    for i in item:
                        stmt=f'SELECT {self.id}, {self.x}, {self.y} FROM {self.table} WHERE id == {i}'
                        rawres=self.query(stmt)
                        single_spec_tup=(rawres[0][0], self.blob2list(rawres[0][1]), self.blob2list(rawres[0][2]))
                        data.append(single_spec_tup)
        return data
                
    def ave_spectra(self,ids):
        """
        Given a tuple of `ids`, returns a tuple of averaged x and y spectra.

        Parameters:
        ids (tuple): A tuple of `id` values to be averaged.

        Returns:
        tuple: A tuple of the format `(name, x, y)`, where `name` is a string representation
        of the `ids` tuple, `x` is the averaged x spectrum, and `y` is the averaged y spectrum.
        """
        if isinstance(ids,tuple):
            stmt=f'SELECT {self.id}, {self.x}, {self.y} FROM {self.table} WHERE id IN {ids}'
            rawres=self.query(stmt)
            res=[(tup[0], self.blob2list(tup[1]), self.blob2list(tup[2])) for tup in rawres]
            x=self.average_lists([tup[1] for tup in res])
            y=self.average_lists([tup[2] for tup in res])
            name = str(ids)[1:-1] #Convert to string and drop the '()'
            return (name,x,y)
        else:
            print('Error: \'ids\' argument must be a tuple.')