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
    def __init__(self, data_df, raman_shifts, name_dict=None, set_dict=None, color_dict=None):
        self.df = data_df
        self.rs = raman_shifts

        self.color_dict=color_dict
        
        if name_dict:
            self.name_dict = name_dict
            self.groups=self.get_groups()
        if set_dict:
            self.set_dict = set_dict
            self.sets = self.get_sets()
            print('set_dict detected')
        else: #this needs to be tested
            self.set_dict=name_dict
            self.sets=self.groups
        
        self.trunc_before_x=None
        self.trunc_after_x=None
        
        self.show_deviation=False
            
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

    def single_plot(self,groups=[],show_std=False):
        if len(groups)==0:
            datas=self.ave_spectra(self.groups)
        else:
            print('error: Natalie hasnt finished this stuff.')
        traces=[]
        
        if show_std:
            for group in datas:
                # print(self.color_dict[group])
                print(type(group))
                traces.append(go.Scatter(x=self.rs, y=datas[group][0], name=group))
                traces.append(go.Scatter(x=self.rs, y=datas[group][0] + datas[group][1], mode='lines', name=f'{group}+', line=dict(color='lightgrey',width=0.5)))
                traces.append(go.Scatter(x=self.rs, y=datas[group][0] - datas[group][1], mode='lines', name=f'{group}-', line=dict(color='rgba(239, 239, 240, 0.1)',width=0.5),fill='tonexty')) #smokewhite=236 all
        else:
            
            traces=[go.Scatter(x=self.rs, y=datas[group][0], name=group, line=dict(color=self.color_dict[group])) for group in datas]
        fig=go.Figure(traces)
        fig.update_layout(template='simple_white')
        fig=self.apply_truncate(fig)
        return fig
    
    def frame_plot(self, groups=[],show_std=False,trace_height=300):
        if len(groups)==0:
            datas=self.ave_spectra(self.groups)
        else:
            print('FINISH ME') #show selected groups
        
        fig=make_subplots(rows=len(datas),cols=1,
                             subplot_titles=[name for name in datas],
                             shared_xaxes=True,
                             vertical_spacing=0)
        for i, group in enumerate(datas):
            fig.add_trace(go.Scatter(x=self.rs,y=datas[group][0]),row=i+1,col=1)
        fig.update_layout(height=trace_height*len(datas),showlegend=False,template='simple_white')
        for i in range(len(datas)):    
            # Adjust title location
            orig_y_loc=fig.layout.annotations[i]['y']
            fig.layout.annotations[i].update(y=orig_y_loc-0.015)
            # fig.layout.annotations[i].update(x=0.1)
        return fig
    
    def set_plot(self,trace_height=300, move_y_title=0):
        datas=self.ave_spectra(self.groups)
        for group in datas:
            pass #here is a group name
        
        fig=make_subplots(rows=len(self.set_dict),cols=1,
                          subplot_titles=[name for name in self.set_dict],
                          shared_xaxes=True,
                          vertical_spacing=0)
        
        # for set_name, group_names in self.set_dict.items():
            # for gname in group_names:
                # pass
                # fig.add_trace(go.Scatter(x=self.rs, y=datas[gname][0], name=gname))                
            
        for i, set_name in enumerate(self.set_dict):
            group_name_list=self.set_dict[set_name]
            for group in group_name_list:
                fig.add_trace(go.Scatter(x=self.rs,y=datas[group][0],name=group,color='black'),row=i+1, col=1)
        fig.update_layout(template='simple_white',height=trace_height*len(datas))
        
        # Adjust title location
        for i in range(len(self.set_dict)):    
            orig_y_loc=fig.layout.annotations[i]['y']
            fig.layout.annotations[i].update(y=orig_y_loc-move_y_title)
            # fig.layout.annotations[i].update(x=0.1)
        return fig
    