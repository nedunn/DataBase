
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
    def __init__(self,intens,rs,
                 names, 
                 color_dict = None,
                 color_by=None,
                 hover=None,symbol=None,
                 ncomp=10):
        self.intens=intens #df
        self.rs=rs #series
        self.ncomp=ncomp
    
        self.names=list(names)

        # self.color_dict=color_dict,
        if color_by==None:
            self.color=self.names
        else:
            self.color=list(color_by)
        if hover==None:
            self.hover=self.names
        else:
            self.hover=hover
        if symbol==None:
            self.symbol=self.names
        else:
            self.symbol=symbol
        


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
        pcs=pd.DataFrame(pcs,columns=pc_labels, index=self.names)

        #Get explained variance
        explained_list=[round(x*100,2) for x in pca.explained_variance_ratio_]
        explained={pc_labels[i]:explained_list[i] for i in range(self.ncomp)}

        #Get loadings
        loadings=pd.DataFrame(X,columns=pc_labels,index=self.rs)
        return pcs, loadings, explained
    
    def pxfig(self,df,x,y,color,color_dict=None):
        syms=self.symbol_gen(len(df))
        fig=px.scatter(df,x=x,y=y,color=color,
                    #    color_discrete_map=color_dict,
                    #    color_continuous_scale=px.colors.qualitative.T10,
                    #    range_color=(-18,10),
                    #    symbol_sequence=syms, symbol=self.symbol,
                        symbol_sequence=['square','circle','diamond','cross'],
                        symbol=self.color,
                       #symbol_sequence=syms, symbol=self.names, #For color blind friendly figure
                       template='simple_white',hover_name=self.hover)
        
        # fig.update_xaxes(range=[0.8,0.9])
        fig.update_traces(marker_size=10)
        return fig

    def pca_figs(self,x,y,color=None,color_dict=None):
        if color is None:
            color=self.names

        pcs,loadings,explained=self.pca_run()

        #PCA scatter plot
        fig1=self.pxfig(pcs,x,y,color,color_dict=color_dict)
        fig1.update_layout(title='%s vs %s'%(x,y),font_size=20)
        fig1.update_xaxes(title_text='%s (%s%s)'%(x,explained[x],'%'),
                          range=[0.8,1])
        fig1.update_yaxes(title_text='%s (%s%s)'%(y,explained[y],'%'),
                          range=[0,0.5])
        
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
        pcs,loadings,explained=self.pca_run()
        
        fig=px.scatter_3d(pcs,x=x,y=y,z=z, color=self.color,
                        # color_discrete_map=self.color_dict,
                        symbol=self.symbol,
                        # symbol_sequence=self.symbol_gen(len(color)),
                        # color_continuous_scale=px.colors.sequential.Jet, range_color=(0,-20),  #Jet,
                        hover_name=self.hover
                        )
        fig.update_layout(height=800)#, showlegend=False)#, template='simple_white')
        fig.update_traces()

        fig.update_layout(scene=dict(
            xaxis_title=f'{x} ({explained[x]}%)',
            yaxis_title=f'{y} ({explained[y]}%)',
            zaxis_title=f'{z} ({explained[z]}%)'),
            font=dict(family='Arial', size=18),
            margin=dict(b=50))
        
        # fig.update_layout(scene=dict(xaxis_width=800))
        # fig.update_xaxes(showline=True, linewidth=5, linecolor='black')
        return fig

    def pca_fig(self,xpc,ypc):#,color_dict):
        #delete?
        figs=self.pca_figs(xpc,ypc,color=self.color)
        figs[1].show()
        figs[2].show()

    def pca_fig(self,figs):
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