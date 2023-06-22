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
                 **kwargs):

        self.df = data_df
        self.rs = raman_shifts
        
        self.color_dict=color_dict
        
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

    # def single_plot(self, sets='all'):
    #     if sets == 'all':
    #         datas = self.ave_spectra(self.groups)
    #     else:
    #         if self.sets and sets in self.sets:
    #             print('1')
    #             group_list = self.sets[sets]
    #             print(group_list)
    #         else:
    #             print('2')
    #             group_list = sets
            
    #     #     selected_data = {group: self.groups[group] for group in group_list if group in self.groups}
    #     #     datas = self.ave_spectra(selected_data)
        
    #     # traces = self.generate_traces(datas)
    #     # fig = go.Figure(traces)
        
    #     # fig = self.format_fig(fig)
    #     return fig
