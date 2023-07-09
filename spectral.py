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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    def __init__(self, data_df, raman_shifts, name_dict=None, set_dict=None, 
                 color_dict=None,
                 ave_dict=None,
                 **kwargs):

        self.df = data_df
        self.rs = raman_shifts
        
        self.color_dict=color_dict
        
        #Dictionaries were key=label, values=sample ids
        self.ave = ave_dict

        if ave_dict is not None:
            self.ave_df=self.get_averaged_df()

        if name_dict:
            self.name_dict = name_dict
            self.groups=self.get_groups()
        if set_dict:
            self.set_dict = set_dict
            self.sets = self.get_sets()
        else:
            self.set_dict, self.sets = None, None
        
        self.trunc_before_x=None
        self.trunc_after_x=None
        
        self.show_deviation=False
        self.subplot_height=300
        self.title_size=30

        self.selected_groups='all' #not yet implimented, to replace `group` arg in functions.
        
        self.__dict__.update(kwargs)

    def get_groups(self):
        """
        Returns a dictionary of group names mapped to corresponding dataframes.

        Returns:
            dict: A dictionary where the keys are group names and the values are
            dataframes containing the spectra for each group.
        """
        groups = {}
        
        for group in set(self.name_dict.values()):  
            ids = [idx for idx, label in self.name_dict.items() if label == group]  
            group_df = self.df.loc[ids]
            groups[group] = group_df
        return groups

    def get_sets(self):
        """
        Returns a dictionary of set names mapped to sets of groups.

        Returns:
            dict: A dictionary where the keys are set names and the values are
            lists of groups.
        """
        sets = {}
        for set_name, group_list in self.set_dict.items():
            set_groups = [self.groups[group] for group in group_list if group in self.groups]
            sets[set_name] = set_groups
        return sets

    def ave_spectra(self,df_dict):
        ave_stds={}
        for group, df in df_dict.items():
            ave=df.mean(axis=0)
            std=df.std()
            ave_stds[group]=(ave,std)
        return ave_stds
    
    def apply_truncate(self,fig):
        if self.trunc_before_x is None and self.trunc_after_x is None:
            pass
        elif self.trunc_before_x is not None and self.trunc_after_x is None:
            fig.update_xaxes(range=[self.trunc_before_x, self.rs[len(self.rs)-1]])
        elif self.trunc_before_x is None and self.trunc_after_x is not None:
            fig.update_xaxes(range=[self.rs[0], self.trunc_after_x])
            
        elif self.trunc_before_x is not None and self.trunc_after_x is not None:
            fig.update_xaxes(range=[self.trunc_before_x,self.trunc_after_x])
        return fig
    
    def return_trunc_df(self):
        pass

    def generate_traces(self,datas):
        traces=[]
        if self.show_deviation:
            if self.color_dict is None:
                for group in datas:
                    traces.append(go.Scatter(x=self.rs, y=datas[group][0], name=group, width=1.2))
                    traces.append(go.Scatter(x=self.rs, y=datas[group][0] + datas[group][1], mode='lines', name=f'{group}+', line=dict(color='lightgrey',width=0.5)))
                    traces.append(go.Scatter(x=self.rs, y=datas[group][0] - datas[group][1], mode='lines', name=f'{group}-', line=dict(color='rgba(239, 239, 240, 0.1)',width=0.5),fill='tonexty')) #smokewhite=236 all
            else:
                for group in datas:
                    traces.append(go.Scatter(x=self.rs, y=datas[group][0], name=group, line=dict(color=self.color_dict[group], width=1.2)))
                    traces.append(go.Scatter(x=self.rs, y=datas[group][0] + datas[group][1], mode='lines', name=f'{group}+', line=dict(color='lightgrey',width=0.5)))
                    traces.append(go.Scatter(x=self.rs, y=datas[group][0] - datas[group][1], mode='lines', name=f'{group}-', line=dict(color='rgba(239, 239, 240, 0.1)',width=0.5),fill='tonexty')) #smokewhite=236 all
        else:
            if self.color_dict is None:
                traces=[go.Scatter(x=self.rs, y=datas[group][0], name=group) for group in datas]
            else:
                traces=[go.Scatter(x=self.rs, y=datas[group][0], name=group, line=dict(color=self.color_dict[group])) for group in datas]
        return traces
    
    def format_fig(self,fig):
        fig.update_layout(template='simple_white')
        fig=self.apply_truncate(fig)
        
        #Does figure have subplots? true = has subplots
        subplots=len(fig.layout.annotations)>1
        
        return fig
    
    def single_plot(self,groups=[]):
        if len(groups)==0:
            datas=self.ave_spectra(self.groups)
        else:
            print('error: Natalie hasnt finished this stuff.')
    
        traces=self.generate_traces(datas)
        fig=go.Figure(traces)
        
        fig=self.format_fig(fig)
        return fig
    
    def frame_plot(self, groups=[]):
        if len(groups)==0:
            datas=self.ave_spectra(self.groups)
        else:
            print('FINISH ME') #show selected groups
        
        fig=make_subplots(rows=len(datas),cols=1,
                             subplot_titles=[name for name in datas],
                             shared_xaxes=True,
                             vertical_spacing=0)
        traces=self.generate_traces(datas)
        for i, group in enumerate(datas):
            trace=traces[i]
            fig.add_trace(trace, row=i+1, col=1)
        fig.update_layout(height=self.subplot_height*len(datas),showlegend=False)
        
        for i in range(len(datas)):    
            # Adjust title location
            orig_y_loc=fig.layout.annotations[i]['y']
            fig.layout.annotations[i].update(y=orig_y_loc-0.015)
            # fig.layout.annotations[i].update(x=0.1)
        
        fig=self.format_fig(fig)
        return fig
    
    def set_plot(self,trace_height=300, move_y_title=0.015):
        if self.set_dict is None:
            return 'Need set dictionary to run'
        #Applying standard deviation is not currently possible, as the STD trace is not selected in the `group_traces` line
        
        else:
            datas=self.ave_spectra(self.groups)
            
            fig=make_subplots(rows=len(self.set_dict),cols=1,
                            subplot_titles=[name for name in self.set_dict],
                            # shared_xaxes=True,
                            vertical_spacing=0.1)
            
            traces=self.generate_traces(datas)
                        
            for i, set_name in enumerate(self.set_dict):
                group_name_list=self.set_dict[set_name]
                group_traces=[trace for trace in traces if trace.name in group_name_list]
                for trace in group_traces:
                    fig.add_trace(trace,row=i+1,col=1)
            
            fig.update_layout(height=self.subplot_height*len(datas))
            
            # Adjust title location
            for i in range(len(self.set_dict)):    
                orig_y_loc=fig.layout.annotations[i]['y']
                fig.layout.annotations[i].update(y=orig_y_loc-move_y_title)
                # fig.layout.annotations[i].update(x=0.1)
            for i in fig['layout']['annotations']:
                i['font']=dict(size=self.title_size)
            
            fig=self.format_fig(fig)
            return fig
        
    def peak_label(self,fig,peak_list,y_height=1,row=1, yref='paper'):
        fig=go.Figure(fig)
        # for p in peak_list:
        #     fig.add_annotation(x=p,y=y_height,
        #                     xref='x',yref=yref,
        #                     text=f'{p:.0f}',
        #                     showarrow=False,
        #                     textangle=-50,
        #                     font_size=20)
            
        fig.add_trace(go.Scatter(x=peak_list, y=self.df,
                                 mode='markers + text',#marker_symbol='line-ns',
                                 showlegend=True,
                                 text=[f'fuck' for x in peak_list],
                                 marker=dict(color='black', size=15, line=dict(color='grey', width=1.5))),
                                 row=row,col=1)
        return fig
    
    def anno_plot(self, groups=[], peak_list=[], sub_loc=[]):
        if len(groups) == 0:
            datas = self.ave_spectra(self.groups)
        else:
            print('FINISH ME')  # show selected groups

        fig = make_subplots(rows=len(datas), cols=1,
                            subplot_titles=[name for name in datas],
                            shared_xaxes=True,
                            vertical_spacing=0)
        traces = self.generate_traces(datas)
        for i, group in enumerate(datas):
            trace = traces[i]
            fig.add_trace(trace, row=i + 1, col=1)

        fig.update_layout(height=self.subplot_height * len(datas), showlegend=False)

        for i in range(len(datas)):
            # Adjust title location
            orig_y_loc = fig.layout.annotations[i]['y']
            fig.layout.annotations[i].update(y=orig_y_loc - 0.015)

            # Add scatter points for peak_list
            x_vals = []
            y_vals = []
            for peak in peak_list:
                x_vals.append(peak)
                y_vals.append(datas[i][0][self.rs.index(peak)])

            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='markers', name='Peaks', marker=dict(color='red')), row=i + 1, col=1)

        fig = self.format_fig(fig)
        return fig
    
    def add_peaks(self, fig, x_vals, y_vals, labels):
        
        trace=go.Scatter(
            x=x_vals, y=y_vals,
            mode='markers',
            marker=dict(color='red'),
            showlegend=False
        )
        return trace
    
    def set_plot(self, peak_list=[]):
        if self.set_dict is None:
            return 'Need set dictionary to run'
        #Applying standard deviation is not currently possible, as the STD trace is not selected in the `group_traces` line
        
        else:
            datas=self.ave_spectra(self.groups)
            
            fig=make_subplots(rows=len(self.set_dict),cols=1,
                            subplot_titles=[name for name in self.set_dict],
                            # shared_xaxes=True,
                            vertical_spacing=0.1)
            
            traces=self.generate_traces(datas)
                        
            for i, set_name in enumerate(self.set_dict):
                group_name_list=self.set_dict[set_name]
                group_traces=[trace for trace in traces if trace.name in group_name_list]
                for trace in group_traces:
                    fig.add_trace(trace,row=i+1,col=1)
                    
                # Add peaks
                if len(peak_list) > 0:
                    # for peak in peak_list:
                        # peak_idx=(self.rs.index.get_loc(peak))


                    # print([datas[i][0][self.rs.index(peak)] for peak in peak_list])
                    peak_trace=self.add_peaks(fig, peak_list, [1]*len(peak_list), 'fucker')
                    fig.add_trace(peak_trace, row=i+1, col=1)

            
            # fig.update_layout(height=self.subplot_height*len(datas))
                        
            fig=self.format_fig(fig)
            return fig

    def get_averaged_df(self):
        dfs=[]
        if self.ave != None:
            for group_name in self.ave.keys():
                ids=list(self.ave[group_name])
                subdf=self.df.loc[ids]
                ave=subdf.mean()
                ave.name=group_name
                dfs.append(ave)
            return pd.concat(dfs,axis=1).T
        else:
            return '`get_averaged_df() requries an averaging dictionary for the instance.\nKey=averaged group label, values=sample ids'
    
    def ave_dict(self):
        pass

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