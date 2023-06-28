#Datatable linker CSV->DB DB->DF
import sqlite3
import pandas as pd
import numpy as np
import json
from prettytable import PrettyTable as pt
from prettytable import ALL
import preprocess
from collections import Counter
import utils
import scipy.signal as ss
import pybaselines.spline as py
# from scipy.sparse import csc_matrix, eye, diags #smooth.whitaker
# from scipy.sparse.linalg import spsolve #smooth.whitaker
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class DataTable:
    ##EDIT add a `check col` function to ensure that columns are correctly input into 'label_dict'
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

        self.name_cols=('filename', 'frame')
        self.name_join=', '
        
        self.x='raman_shift' #Database table name where 'x' values are stored
        self.y='original_intensity'
        self.id='id' #Database table column name where spectra specific 'id' valuesare stored
 
        #Optional condition for Query searchs
        self.condition=None

        #Track sample IDs that are dropped bc they are flagged with outliers
        self.dead_list=[] #IDs of samples
        if table_name is not None:
            self.all_ids=self.get_ids()

    def __del__(self):
        '''Automaticallly close the database connection when the object is destroyed (program terminated).'''
        self.conn.close()

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

    def get_ids(self, condition=False):
        if condition:
            ids_badformat=self.query(f'SELECT {self.id} FROM {self.table} {condition}')
        else:
            ids_badformat=self.query(f'SELECT {self.id} FROM {self.table}')
        ids=[item[0] for item in ids_badformat]
        return ids
    
    def select_ids(self):
        return self.get_ids(condition=self.condition)
    
    def columns(self):
        '''Returns a list of column names in the current database table.'''
        columns=self.query(f'PRAGMA table_info({self.table})')
        cols=[col[1] for col in columns]
        return cols
    
    def summary(self,*incols,detail='low'):
        if len(incols) !=0:
            detail='high'
        if detail == 'low':
            self.small_sum()
        elif detail == 'high':
            self.high_sum(*incols)

    def high_sum(self, *incols):
        # if len (incols)==0:
            # return self.table_info()
        if not incols:
            return self.table_info()
        else:
            if self.condition:
                stmt=f'SELECT COUNT (*) FROM {self.table} {self.condition}'
            else:
                stmt=f'SELECT COUNT (*) FROM {self.table}'
            display=f'Currently the dataset is accessing {self.query(stmt)[0][0]} samples from the database.\n'
            display+=('+')
            display+=('-'*47)
            display+=('+')
            display+=('\n|Summaries of the data in the selected columns: | ')
            print(display)
            for col_table in self.col_info(*incols):
                print(col_table)

    def small_sum(self):
        print(f'All IDs: {len(self.all_ids)}')
        print(f'Selected IDs: {len(self.select_ids())}')
        print(f'Condition: {self.condition}')

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
        stmt=f'SELECT COUNT(*) FROM {self.table}'
        title=('-' * (len(stmt)+4))
        title+=f'\n| Summary for \'{self.table}\' Table |\n'
        title+=('-' * (len(stmt)+4))
        print(title)
        
        # #Get Row information
        self.cursor.execute(stmt)
        rows=self.cursor.fetchone() #num_rows = f'Number of rows: {len(rows)}'?
        
        print(f'Number of samples (rows): {rows[0]}')
        print(f'Number of columns in table: {len(self.columns())}')

        ########################

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
        #Make sure user wants to update the data table
        # input('Press ENTER to continue...')

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
            print('AHHHH ERROR at find_replace in data_wrangle')

    def blob2list(self,byte_data):
        '''Pass blob/byte data to np.array'''
        try: # byte data look like: b'{"0":282,"1":283,"2":285,"3":2... (dop specific)
            asstr=byte_data.decode('utf-8')
            dic=json.loads(asstr)
            clean=np.array(list(dic.values()))
            # print('decoding for do_pac')
        
        except: #byte data look like: b'\xf0\x01\x00\x00\
            clean=np.frombuffer(byte_data,dtype=np.int64)
        
        return clean
    
    def grab_raw_data(self,*cols):
        # SELECT CONCAT(column1, column2) AS combined_values
        # FROM your_table
        # WHERE CONCAT(column1, column2) NOT IN ('10', '01');
        # if len(cols) > ... use concat?                                                                                                                                                                                                                                                                                                                                                                        

        if cols==None:
            pass
        else:
            cols=', '.join(cols)
        if self.condition:
            rawres=self.query(f'SELECT {self.id}, {cols} FROM {self.table} {self.condition}')
        else:
            rawres=self.query(f'SELECT {self.id}, {cols} FROM {self.table}') 
        return rawres
    
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

    def dataset(self,  show_x_summary=False, dropnans=True, summary=True):

        rawres=self.grab_raw_data(self.x,self.y)

        #Filter rawres by removing any samples whos ID has been added to dead_list
        res=[tup for tup in rawres if tup[0] not in self.dead_list]

        #Convert byte data in raw tuples to list data
        all_tup_list=[(tup[0],self.blob2list(tup[1]),self.blob2list(tup[2])) for tup in res]
        
        # Make each spectra a dataframe with single row
        dfs=[]
        for tup in all_tup_list:
            unique_index=utils.modify_duplicates(tup[1])
            df=pd.DataFrame(tup[2], index=unique_index, columns=[tup[0]])
            dfs.append(df)

        # # Check out `x`s
        # summary=utils.summarize_lists([tup[1] for tup in all_tup_list])
        #the ** happened to my function in utils??!?
        # if show_x_summary:
        #     for xlst in summary:
        #         print(xlst)
        
        stmt=f'\tNumber of samples compiled: {len(dfs)}'

        df1=pd.concat(dfs,axis=1)
        stmt+=f'\n\tInitial dataset contains {df1.shape[0]} Raman Shifts and {df1.shape[1]} individual spectra.'

        if dropnans:
            df=df1.dropna()
            stmt+=f'\n\tFinal DF contains {df.shape[0]} Raman Shifts and {df.shape[1]} individual spectra.'
            stmt+=f'\n\t`NaN` values were removed.'
        else:
            df=df1
            stmt+='\n\tDF returned with possible `NaN` values.'
        if summary:
            print('DataFrame Returned:')
            print(stmt)
            print('\tRow index = sample ID || Column header = Raman Shift')
        return df.T 

    def label_dict(self,*cols, val_as_tup=False, include_col_name=True,
                  drop_deadlisted=False, name_sep=', '): #original func(names)
        # Check for inputs
        if len(cols)==0:
            cols=self.name_cols
            if type(cols) is str:
                col_str=cols
            else:
                col_str=name_sep.join(cols)
        elif len(cols) == 1:
            col_str=cols[0]
        elif len(cols) > 1:
            col_str=name_sep.join(cols)
        # print(f'{col_str}\n\ttype: {type(col_str)}\n\t{len(col_str)}')

        # Access DB, get tuples
        rawres=self.grab_raw_data(col_str)
        
        # Return dictionary: {ID:Name}
        res={}
        if include_col_name==True:
            # *Optional* add column names to each value
            for tup in rawres:
                new_items=[]
                for item in tup[1:]:
                    idx=tup.index(item)
                    new_item=f'{item} {cols[idx-1]}'
                    new_items.append(new_item)
                    #if val_as_tup == False: #Get the returns as a dictionary where 'id' = key
                res[tup[0]]=name_sep.join(new_items)    
        elif include_col_name==False:
            for tup in rawres:
                new_items=[]
                for item in tup[1:]:
                    new_item=f'{item}'
                    new_items.append(new_item)
                res[tup[0]]=name_sep.join(new_items)
        else:
            return 'Error: `include_col_name` must be either `True` or `False`.'
        
        return res

class PreproSpectra:
    """
    A class for preprocessing a Raman spectra.

    Parameters
    ----------
    original_intensity : list or array-like
        The original intensity values of the spectral data.

    raman_shifts : list or array-like, optional
        The corresponding Raman shift values. If not provided, bogus Raman shift values will be generated.

    alerts : bool, optional
        Flag indicating whether to display alerts generated when initializing instance.

    name : str, optional
        Name of the PreproSpectra instance.

    no_neg : bool, optional
        Flag indicating whether to replace negative intensity values with 0. Default is True.
    """
    def __init__(self, original_intensity, raman_shifts=None,
                 name=None, alerts=True, no_neg=True,
                 snv=True,
                 smooth_window=9, smooth_poly=3, zap_window=2, zap_threshold=5,
                 **params):
        
        self.y=original_intensity
        if raman_shifts is None:
            alert=f'You did not provide Raman Shift values.\n'
            self.x=list(range(0,len(self.y))) #Generate bogus Raman Shift values for X axis
        elif (len(original_intensity) != len(raman_shifts)) == True:
            alert=f'Error: the given `original_intensity` and `raman_shifts` are not of equal length.\n'
            alert+='\tBogus values will be used instead.\n'
            alert+=f'\tLength intensity values: {len(original_intensity)}\n\tLength RS values: {len(raman_shifts)}\n'
            self.x=list(range(0,len(self.y)))
        else:
            alert=''
            self.x=raman_shifts
        self.name=name

        # Smoothing parameters
        self.smooth_window=smooth_window
        self.smooth_poly=smooth_poly
        
        # Zap parameters
        self.zap_window=zap_window
        self.zap_threshold=zap_threshold

        # Apply Zap
        zap, zap_text = self.zap(self.y)
        self.y_zap=zap
        alert+=zap_text
        
        # Apply Smooth
        self.y_zap_smooth=ss.savgol_filter(self.y_zap, self.smooth_window, self.smooth_poly)
        alert+=f'Smoothing has been applied: window = {self.smooth_window}, polynomial = {self.smooth_poly}.\n'

        # Baseline
        self.baseline=py.pspline_asls(self.y_zap_smooth)[0]
        y_base=self.y_zap_smooth-self.baseline

        # SNV
        if snv:
            Y=self.snv(y_base)
            alert+='SNV normalization applied to spectra.'
        else:
            Y=y_base

        # Return preprocessed data
        if no_neg:
            self.Y=list(map(lambda num: num if num >= 0 else 0, Y))
            alert+='Intensity returned with all negative values replaced with `0`.\n'
        else:
            self.Y=Y
            alert+='Intensity returned with negative values.\n\t**Set `no_neg` to `True` to remove negatives.\n'

        # self.__dict__.update(params)

        # Display alerts
        if alerts:
            print(alert)
 
    def zscore(self,nums):
        """
        Calculate the Z-scores of the given numbers.

        Parameters
        ----------
        nums : list or array-like
            The input numbers.

        Returns
        -------
        zscores : ndarray
            The Z-scores of the input numbers.
        """
        mean=np.mean(nums)
        std=np.std(nums)
        zscores1=(nums-mean)/std
        zscores=np.array(abs(zscores1))
        return(zscores)
    
    def mod_zscore(self,nums):
        """
        Calculate the modified Z-scores (MAD Z-scores) of the given numbers.

        Parameters
        ----------
        nums : list or array-like
            The input numbers.

        Returns
        -------
        mod_z_scores : ndarray
            The modified Z-scores (MAD Z-scores) of the input numbers.
        """
        median_int=np.median(nums)
        mad_int=np.median([np.abs(nums-median_int)])
        mod_z_scores1=0.6745*(nums-median_int)/mad_int
        mod_z_scores=np.array(abs(mod_z_scores1))
        return mod_z_scores
    
    def WhitakerHayes_zscore(self, nums, threshold):
        """
        Whitaker-Hayes Function using Intensity Modified Z-Scores.

        Parameters
        ----------
        nums : list or array-like
            The input numbers.
        threshold : int or float
            The threshold value.

        Returns
        -------
        intensity_modified_zscores : ndarray
            The intensity modified Z-scores.
        """
        dist=0
        delta_intensity=[]
        for i in np.arange(len(nums)-1):
            dist=nums[i+1]-nums[i]
            delta_intensity.append(dist)
        delta_int=np.array(delta_intensity)
        
        #Run the delta_int through MAD Z-Score Function
        intensity_modified_zscores=np.array(np.abs(self.mod_zscore(delta_int)))
        return intensity_modified_zscores
    
    def detect_spikes(self,nums):
        """
        Detect spikes, or sudden, rapid changes in spectral intensity.

        Parameters
        ----------
        nums : list or array-like
            The input numbers.

        Returns
        -------
        spikes : ndarray
            Boolean array indicating whether each value is a spike (True) or not (False).
        """
        spikes=abs(np.array(self.mod_zscore(np.diff(nums))))>self.zap_threshold
        return spikes
    
    def zap(self,nums):
        """
        Replace spike intensity values with the average values that are not spikes in the selected range.

        Parameters
        ----------
        nums : list or array-like
            The input numbers.
        window : int = selected range
            Selection of points around the detected spike.
            Default = 2.
        threshold : int
            Binarization threshold. Increase value will increase spike detection sensitivity. (*I think*)
            Default = 5.

        Returns
        -------
        y_out : list or array-like
            Window Average.
            Average values that are around spikes in the selected range.
        """
        y_out=nums.copy() #Prevents overeyeride of input y
        spikes=abs(np.array(self.mod_zscore(np.diff(nums))))>self.zap_threshold
        try:
            for i in np.arange(len(spikes)):
                if spikes[i] !=0: #If there is a spike detected in position i
                    w=np.arange(i-self.zap_window, i+1+self.zap_window) #Select 2m+1 points around the spike
                    w2=w[spikes[w]==0] #From the interval, choose the ones which are not spikes                
                    if not w2.any(): #Empty array
                        y_out[i]=np.mean(nums[w]) #Average the values that are not spikes in the selected range        
                    if w2.any(): #Normal array
                        y_out[i]=np.mean(nums[w2]) #Average the values that are not spikes in the selected range
            return y_out, f'Zap has been applied with threshold = {self.zap_threshold}, window = {self.zap_window}.\n'
        except TypeError:
            return nums, 'Zap step has been skipped.\n'
    
    def snv(self,nums):
        vals=np.array(nums)
        ave=vals.mean() #Calculate the mean for the sample
        centered=vals-ave #Subtract the mean from each intensity value
        std=np.std(centered) #Calculate STD
        res=centered/std#Divide each item in the centered list by the STD
        return res

    def show(self):
        """
        Display the data before and after preprocessing.

        Returns
        -------
        fig: plotly.graphs.Figure
            The figure object containing the plot.
        """
        # Initalize figure
        fig=make_subplots(rows=2, cols=1,
                        shared_xaxes=True, shared_yaxes=True,
                        vertical_spacing=0.05,
                        x_title='Raman Shift (cm-1)', y_title='Intensity')
        #add traces
        fig.append_trace(go.Scatter(x=self.x, y=self.y, name='raw'),row=1,col=1)
        fig.append_trace(go.Scatter(x=self.x, y=self.baseline, name='baseline'),row=1,col=1)
        fig.append_trace(go.Scatter(x=self.x, y=self.Y, name='output'),row=2,col=1)
        
        # Adjust layout
        fig.update_layout(title_text=self.name,title_font_size=15,plot_bgcolor='rgba(0,0,0,0)')
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', gridcolor='lightgrey')
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', gridcolor='lightgrey')
        return fig
    
    def __str__(self):
        """
        Return a string representation of the PreproSpectra instance.

        Returns
        -------
        str
            A string representation of the PreproSpectra instance.
        """
        return f"PreproSpectra instance: {self.name if self.name else 'Unnamed'}"

    def __repr__(self) -> list:
        return repr(self.Y)
    
    def get(self):
        return np.array(self.Y)

class DataSet:
    '''
    df where index = sample ID and column = raman shift
    '''
    def __init__(self, df, 
                 trunc_before=None,trunc_after=None,
                 **params):
        n, n_rs = df.shape[0], df.shape[1]

        self.ids=df.index
        self.xax=df.columns

        # Truncate
        if trunc_before != None or trunc_after != None:
            self.before=utils.closest_number(trunc_before, self.xax)
            self.after=utils.closest_number(trunc_after, self.xax)
            self.raw=self.truncate(df)    
        else: # No truncate
            self.raw=df

        self.df=self.prepro_df(**params)
        
    def truncate(self, df):        
        return df.truncate(before=self.before, after=self.after, axis='columns')

    def prepro_df(self, **params):
        res=[]
        for i in self.ids:
            d=pd.DataFrame(PreproSpectra(list(self.raw.loc[i]), alerts=False, **params).get())
            d.columns=[i]
            d.set_index([self.raw.columns],inplace=True,drop=True)
            res.append(d)        
        return pd.concat(res,axis=1)
    
    def getdf(self):
        return self.df.T
        
    def __repr__(self):
        return repr(self.df.T)
    
class Spectral:
    """
    A class for calculating means and standard deviations of rows in a DataFrame based on given indexes.
    
    Attributes:
        df (pandas.DataFrame): The DataFrame containing the data to be analyzed.
        ave_index_dict (dict): A dictionary specifying the indexes to be averaged by group name.
        ave_dict (dict): A dictionary with index keys as keys and tuples of mean values and standard deviations as values.

    Methods:
        calc_ave_std_dict(): Calculates the means and standard deviations of rows in a DataFrame based on given indexes.
    
    """
    def __init__(self, data_df, 
                 averaging_dict=None, # key = group name, value = list of data_df indexes (sample IDs) to average by
                 **kwargs):

        self.df = data_df

        if averaging_dict is not None:
            self.ave_index_dict=averaging_dict
        else:
            print('A dictionary must still be given for this function to work.')
            print('Dict key = group name (baed on columns from database), value = list of sample IDs (as assigned in database) that fall under the given label/group.')
        
        self.ave_dict=self.calc_ave_std_dict()

        self.groups=self.list_groups()
        

    def calc_ave_std_dict(self):
        """Calculates the means and standard deviations of rows in a DataFrame based on given indexes.
        
        Returns a dictionary where each key corresponds to an index key and its value is a tuple
        containing the mean values and standard deviations of the rows for that key.
        
        Returns:
            dict: A dictionary with index keys as keys and tuples of mean values and standard deviations as values.
        """
        res={}
        for key, ids in self.ave_index_dict.items():    
            # Perform calculations
            rows=self.df.loc[ids]
            means=rows.mean()
            stds=rows.std()

            # Save calclations to new dictionary
            res[key] = (means.tolist(), stds.tolist())
        return res
    
    def list_groups(self):
        return(list(set(self.ave_index_dict.keys())))
    
    def get(self):
        return(self.ave_dict)

class Traces:
    def __init__(self,spect_dict,raman_shifts,
                 selected_groups=None, set_dict=None,
                 show_std=False):
        self.dict=spect_dict #Key is averaged spectra label, value = (mean, std)
        self.rs=raman_shifts

        # Identify traces to build based on groups
            #group=key from self.dict
        self.groups=list(set(self.dict.keys()))
        if selected_groups is not None:    #User selected groups to plot
            self.selected_groups=selected_groups #Note: the order of listed groups should dictate plotting order
        else:
            self.selected_groups = self.groups

        self.set_dict=set_dict
        if self.set_dict is not None:
            self.sets=set_dict.keys()

        # Trace attributes
        self.show_std=show_std
        self.subplots=True

    def key_trace(self, key):
        '''Returns Plotly Figure Trace Data relating to a key (group name) from the input spectral dictionary'''
        trace=go.Figure()
        ave=pd.Series(self.dict[key][0])
        std=pd.Series(self.dict[key][1])
        # Make Trace Data
        if self.show_std: #UNFINISHED set up for appying standard deviation
            trace.add_trace(go.Scatter(x=self.rs, y=ave + std, mode='lines', name='upper bound', line=dict(color='lightgrey')))
            trace.add_trace(go.Scatter(x=self.rs, y=ave, name=key))
            trace.add_trace(go.Scatter(x=self.rs, y=ave - std, mode='lines', name='lower bound', line=dict(color='lightgrey')))
        
        else:
            trace.add_trace(go.Scatter(x=self.rs, y=ave, name=key))
        return trace.data

    def plot_single(self):
        '''Returns figure with all groups on the same plot'''
        traces=[]
        for group in self.selected_groups:
            traces.extend(self.key_trace(group))
        return go.Figure(traces)

    def plot_per_key(self):
        '''Returns figures with each group on its own'''
        for i, group in enumerate(self.selected_groups):
            trace=self.key_trace(group)
            fig=go.Figure(trace)
            fig.update_layout(title=group)
            fig.show()
            # fig.add_trace(trace,row=i+1, col=1)
    
    def subplots(self):
        '''Returns a figure with each group as its own subplot'''
        fig=make_subplots(rows=len(self.selected_groups), cols=1,
                          shared_xaxes=True,
                          vertical_spacing=0.01)
        
        for i, group in enumerate(self.selected_groups):
            traces=self.key_trace(group)
            for trace in traces:
                fig.add_trace(trace,row=i+1,col=1)
        return fig
            
    def set_subplots(self):
        '''Returns figure with each subplot a SET as defined by set_dict'''
        if self.set_dict is None:
            return 'A set_dict must be provided to use the `set_subplot` function.'
        plot_list=self.set_dict.keys()

        fig=make_subplots(rows=len(plot_list), cols=1,
                          shared_xaxes=True,
                          subplot_titles=list(self.sets))

        #Traces per subplot
        for i, set_name in enumerate(self.set_dict):
            trace_names=self.set_dict[set_name]
            for name in trace_names:
                traces=self.key_trace(name)
                for trace in traces:
                    fig.add_trace(trace, row=i+1, col=1)
        return fig