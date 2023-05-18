#Datatable linker CSV->DB DB->DF
import sqlite3
import os
import pandas as pd
import numpy as np
import json
from prettytable import PrettyTable as pt
from prettytable import ALL
import preprocess
from collections import Counter

#Detect Oversaturation of spectra
def oversat_check(spectrum, threshold=0.1, window=5):
    """
    Detects oversaturation in a given spectrum by checking if there is a section 
    where the slope of the spectrum is consistently below a given threshold 
    over a given window of points.

    Parameters
    ----------
    spectrum : tuple
        A tuple containing the x and y data of the spectrum to be analyzed.
    threshold : float, optional
        The threshold slope value below which a section of the spectrum is considered 
        to be oversaturated. Default is 0.1.
    window : int, optional
        The size of the window over which to calculate the slopes. Default is 5.

    Returns
    -------
    bool
        True if an oversaturated section is detected, False otherwise.
    """
    x,y = spectrum #Grab x,y data from tuple input
    
    slopes = np.diff(y) / np.diff(x)
    for i in range(len(slopes) - window + 1):
        if np.allclose(slopes[i:i+window], slopes[i], atol=threshold):
            return True
    return False

class Spectral:
    def __init__(self, db_path,table_name=None):
        pass
    
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

        #Convert tuples list to dictionary
        data_dict={t[0]:t[1] for t in tup_list} #(id,[xs values])

        ###Compare xs, look for outliers
        #Create a DF from the dictionary
        df=pd.DataFrame.from_dict(data_dict,orient='index') #Columns = rs index, index=sampleID (int)
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
        # return result
        res = np.zeros_like(df)
        np.putmask(res, row_std != 0, result) # If row_std != 0, put the corresponding value of result in re
        return res
    
    def apply_snv(self,df): #Should be okay to remove
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