
import pandas as pd
import numpy as np
# from sklearn.decomposition import PCA

from sklearn import decomposition
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score
from math import ceil

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import scipy.signal as ss
import pybaselines.spline as py


class multivar:
    '''Input dataframe is expected to have Raman shift as columns and rows are samples.
    The index values for each row should be the sampleID.
    Dictionaries can be used to set the labels for hovertext, color, and symbol.''' 
    def __init__(self,intens,rs,
                 name_dic,
                 symbol_dic=None,
                 hover_dic=None,
                #  names, 
                 
                 color_by=None,
                 hover=None,
                 ncomp=10,
                 color_dict=None): #color_dic is color assignment for name values?
        
        self.sampid=intens.index.to_list()
        
        self.intens=intens.T #df
        self.rs=rs #series
        self.ncomp=ncomp

        self.color_dict=color_dict

        # Set dictionaries for labeling
        self.name_dic=name_dic
        if symbol_dic==None:
            self.symbol_dic=self.name_dic
        else:
            self.symbol_dic=symbol_dic
        # self.symbol_dic = self.name_dic if symbol_dic is None else symbol_dic
        if hover_dic==None:
            self.hover_dic=self.name_dic
        else:
            self.hover_dic=hover_dic

        self.rs_ax='Raman Shift (cm-1)'

    def label_gen(self,prefix,n):
        nums=list(range(1,n+1))
        res=['%s %s'%(prefix, num) for num in nums]
        return res
    
    def symbol_gen(self,len_needed):
        syms=['circle','square','diamond','cross','x','triangle-up','star-triangle-down','triangle-left','triangle-se','pentagon','hexagon2','star','diamond-tall']
        if len_needed <= len(syms):
            return syms[0:len_needed]
        else:
            times_to_mult=ceil((len_needed/len(syms)))
            rep_syms=syms*times_to_mult
            return rep_syms[0:len_needed]

    def pca_run(self):
        pca=decomposition.PCA(n_components=self.ncomp)
        
        #Apply PCA
        X=scale(self.intens)
        X=pca.fit_transform(X)

        #Get components
        pc_labels=self.label_gen('PC',self.ncomp)
        pcs=pca.components_.T*np.sqrt(pca.explained_variance_)
        pcs=pd.DataFrame(pcs,columns=pc_labels, index=self.sampid)
    
        #Get explained variance
        explained_list=[round(x*100,2) for x in pca.explained_variance_ratio_]
        explained={pc_labels[i]:explained_list[i] for i in range(self.ncomp)}

        #Get loadings
        loadings=pd.DataFrame(X,columns=pc_labels,index=self.rs)
        return pcs, loadings, explained
    
    def prep_labels(self, df):
        namelist = [self.name_dic.get(idx, idx) for idx in df.index]
        symlist = [self.symbol_dic.get(idx, idx) for idx in df.index] if self.symbol_dic is not None else None
        hoverlist = [self.hover_dic.get(idx, idx) for idx in df.index] if self.hover_dic is not None else namelist
        return namelist, symlist, hoverlist
        

    def pxfig(self,df,x_col,y_col):
        #some code to turn on/off 'color blind mode'
        #move to initial attributes?
        if self.name_dic == self.symbol_dic:
            print('pxfig note: add a color blind mode option here')
        else:
            print('pxfig note: different labels for symbols vs colors')

        names, symbols, hover = self.prep_labels(df)
        
        fig=px.scatter(df, x=x_col, y=y_col,
                       color=names, symbol=symbols, hover_name=hover,
                       color_discrete_map=self.color_dict)
        # TO ADD
        # + symbol_squence
        # + color_discrete_map = color_map_dictionary
        # + fig.update_traces(marker_size=10)
        return fig

    def pca_figs(self,x_col,y_col,x_ax_range=None,y_ax_range=None):
        pcs,loadings,explained=self.pca_run()
        names, symbols, hover = self.prep_labels(pcs)
        x,y=x_col,y_col

        #PCA scatter plot
        fig1=self.pxfig(pcs,x,y)#,color,color_dict=color_dict)
        fig1.update_layout(title='%s vs %s'%(x,y),font_size=20)
        fig1.update_xaxes(title_text='%s (%s%s)'%(x,explained[x],'%'),
                        range=x_ax_range)  # [0.8,1] dop range?
                                
        fig1.update_yaxes(title_text='%s (%s%s)'%(y,explained[y],'%'),
                          range=y_ax_range) #0,0.5 dop range?
                        
        
        # save_name='pc_zoom_mix'
        # pio.write_image(fig,'/mnt/c/Users/16162/Desktop/%s.svg'%save_name,
        #         width=1600,
        #         height=800)
        
        #PCA Loadings
        fig2=px.line(loadings,x=loadings.index,y=x,
                    title='%s (%s%s)'%(x,explained[x],'%'), template='simple_white',
                    labels={x:'','index':self.rs_ax})
        #fig.show()
        fig3=px.line(loadings,x=loadings.index,y=y,
                    title='%s (%s%s)'%(y,explained[y],'%'),template='simple_white',
                    labels={y:'','index':self.rs_ax})
        #fig.show()
        # fig1.show()
        return fig1,fig2,fig3

    def loadings(self,*pcs):
        pcs_df, load, explain = self.pca_run()
        figs=[]

        for pc in pcs:
            title=f'PC {pc}'
            fig=px.line(load, x=self.rs, y=load[title],
                        title='%s (%s%s)'%(title, explain[title], '%'),
                        template='simple_white',
                        labels={title:'', 'x':self.rs_ax})
            figs.append(fig)
        return figs
            
    def pca_3d(self, x='PC 3', y='PC 2', z='PC 1'):
        if self.name_dic==None:
            print('default 3d pca labels')
            #Need to add default code here later so namedic is not required.
        else:
            print('dictionary for color labels exists')
        
        pcs,loadings,explained=self.pca_run()
        
        #Get list for how to color datapoints on plot   
        names, symbols, hover = self.prep_labels(pcs)

        fig=px.scatter_3d(pcs,x=x,y=y,z=z, color=names,
                          symbol=symbols,
                          hover_name=hover
                        # color_discrete_map=self.color_dict,
                        # symbol_sequence=self.symbol_gen(len(color)),
                        # color_continuous_scale=px.colors.sequential.Jet, range_color=(0,-20),  #Jet,
                        )

        fig.update_layout(height=800)#, showlegend=False)#, template='simple_white')
        
        fig.update_layout(scene=dict(
            xaxis_title=f'{x} ({explained[x]}%)',
            yaxis_title=f'{y} ({explained[y]}%)',
            zaxis_title=f'{z} ({explained[z]}%)'),
            font=dict(family='Arial', size=18),
            margin=dict(b=50))
        
        # fig.update_layout(scene=dict(xaxis_width=800))
        # fig.update_xaxes(showline=True, linewidth=5, linecolor='black')
        return fig

    def pca_fig(self,x_col,y_col):
        figs=self.pca_figs(x_col, y_col)
        fig=make_subplots(rows=2,cols=2,specs=[[{'rowspan':2},{}],[None,{}]],
                          subplot_titles=[
                          figs[0].layout.title.text,figs[1].layout.title.text,figs[2].layout.title.text],
                          vertical_spacing=0.35)
        
        for trace in figs[0].data:
            fig.add_trace(trace,row=1,col=1)
            fig.update_xaxes(title_text=figs[0].layout.xaxis.title.text,row=1,col=1)
            fig.update_yaxes(title_text=figs[0].layout.yaxis.title.text,row=1,col=1)
        for trace in figs[1].data:
            fig.add_trace(trace,row=1,col=2)
        for trace in figs[2].data:
            fig.add_trace(trace,row=2,col=2)

        fig.update_xaxes(title_text=figs[1].layout.xaxis.title.text,row=1,col=2)
        fig.update_xaxes(title_text=figs[1].layout.xaxis.title.text,row=2,col=2)
        fig.update_xaxes(title_text=figs[0].layout.xaxis.title.text,row=1,col=1)
        fig.update_yaxes(title_text=figs[0].layout.yaxis.title.text,row=1,col=1)

        fig.update_layout(template='simple_white',
                          font_size=20,
                          font_family='ariel')
        return fig