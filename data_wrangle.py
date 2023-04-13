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

        self.condition=None
            
    def query(self,stmt,fetch=True):
        if fetch==True:
            res = self.cursor.execute(f'{stmt}')
            return self.cursor.fetchall()
        else:
            res = self.cursor.execute(f'{stmt}')
            return None

    def columns(self,*incols):
        columns=self.query(f'PRAGMA table_info({self.table})')
        cols=[col[1] for col in columns]
        
        #Make a table of column info
        table=pt()
        table.title='Column Information'
        table.field_names=['colID','name','unique_vals_per_col','type','notnull','dflt_value','primary_key']
        for c in columns:
            # c = 'tuple of column info'
            unique_vals_per_column=self.query(f'SELECT COUNT(DISTINCT {c[1]}) FROM {self.table}')
            col_info=list(c)
            # col_info.append(unique_vals_per_column[0][0])
            col_info.insert(2,unique_vals_per_column[0][0])
            table.add_row(col_info)
        if len(incols)>0:
            for col in incols:
                uniques=self.query(f'SELECT DISTINCT {col} FROM {self.table}')
                # print(uniques)
                stmt=f'\nValues in {col}:'
                for u in uniques:
                    stmt+=f'\n\t{u[0]}'
                    # print(u[0])
                    # if len(u[0])<=1:
                        # print(u[0])
                    # else:
                        # print(u[0])
                        # [print(f'{u[0]}') for i in list(range(len(u[0])))]
                        # [print(f'{u[0]}') for ]
                        # print(list(range(len(u[0]))))
                        # print(u)
                        # stmt=+[f'{u[0]}' for i in list(range(len(u[0])))]
                    # print(len(u[0]))
                    # print(type(u))
                    # stmt+=f'\t{u[0]}'
                print(stmt)

        return cols,table
    
    def update(self,col_name,old_value,new_value):
        stmt=f"UPDATE {self.table} SET {col_name} = '{new_value}' WHERE {col_name} = '{old_value}'"
        self.cursor.execute(stmt)
        self.conn.commit()
        

    def table_info(self,*cols):
        #row information
        stmt=f'SELECT COUNT(*) FROM {self.table}'
        self.cursor.execute(stmt)
        rows=self.cursor.fetchone() #num_rows = f'Number of rows: {len(rows)}'

        #Display info
        print(f'DATATABLE INFORMATION')

        table=pt(header=False, hrules=ALL, vrules=ALL)
        table.add_row(['Table Name',f'{self.table}','Number of samples\n(rows)',rows[0],
                       'Number of columns',f'{len(self.columns()[0])}'])
        #Show Data table dimensions
        print(table) 
        #Show column details
        print(self.columns()[1])
    
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


    def raman_shifts(self, table_col_name='raman_shift', table_col_id='id'):
        #DELETE#
        '''table_col_name = name of column in data table were x data is.
            table_col_id = name of column in data table where unique identifier for each individual spectra is.
            ***Add qualities for query search***
        '''
        #Grab data from DB
        rawres=self.query(f'SELECT {table_col_id}, {table_col_name} FROM {self.table}')
        # print(rawres[0][1])
        #Convert byte data in raw tuples to list data
        tup_list=[(tup[0],self.blob2list(tup[1])) for tup in rawres]
        # return tup_list

        #Convert tuples list to dictionary
        data_dict={t[0]:t[1] for t in tup_list} #(id,[xs values])
        # for d in data_dict:
            # print(d)
            # print(data_dict[d])
            # print()
        # return data_dict
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
        outliers=(np.abs(df-mean) >  threshold*std)
        #Print results of comparison if there are outliers
        for column in df.columns: #Check each column (each Raman Shift)
            column_outliers = outliers[column] #Check if there is an outlier, if there is, add it to this list as TRUE
            # print(type(column_outliers))
            print(column_outliers.index[column_outliers])
            # print(column_outliers)
            # print(df)
            if column_outliers.any(): #If outlier is detected... generate alert message
                outlier_values = df.loc[column_outliers, column].values #np.array of outlier values PER COLUMN
                # print(outlier_values)
            #     table=pt(['Raman Shift Value','Sample ID'])
            #     table.title=f'Outlier(s) in column **{column}**'
            #     for val in outlier_values: #For each outlier detected in the column
            #         ind=df.loc[df.iloc[:,column]==val].index[0] #Grab index (sampleID) of the outlier
            #         table.add_row([val,ind]) #Display the outlier value and the sampleID
            #     print(table)
        return mean
    
    # def data_from_db(self,table_col_name, table_col_id, condition=None):
    #     if condition:
    #         stmt=f'SELECT {table_col_id}, {table_col_name} FROM {self.table} WHERE {condition}'
    #     else:
    #         stmt=f'SELECT {table_col_id}, {table_col_name} FROM {self.table}'

    
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
        