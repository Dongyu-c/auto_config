import pandas as pd
import pyreadr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.stats import chisquare
from statsmodels.formula.api  import ols
from statsmodels.stats.anova  import anova_lm
import copy,time
from plotnine import ggplot, xlab,ggtitle,aes, geom_line,geom_boxplot,geom_point,facet_wrap,geom_hline,xlab,theme,scale_x_discrete,element_text

def pre_process(rawdata,parameter):
    """
    Before doing config analysis,find the part_num and the newest group_num for a certain parameter in the dataframe. 
    Input: 
        rawdata is dataframe with commodity data. 
        parameter is the yaxis for config analysis. eg:parameter='FLANGE_HT_PROJECTION'
    Output: 
        partnum_groupnum: dict structure

    """
    sup_partn_groupnum=dict()
    partn_groupnum=dict()
    for sup in rawdata['SUPPLIER_NAME'].unique():
        raw_sup=rawdata[rawdata['SUPPLIER_NAME']==sup]
        sup_partn_groupnum.update({sup:{}})
        for partnum in raw_sup['PART_NUM'].unique():
            df_motor=raw_sup[raw_sup['PART_NUM']==partnum]
            if len(df_motor['GROUP_NUM'].unique())<10:
                del sup_partn_groupnum[sup]
                pass
            else:
                ds=df_motor.groupby(['GROUP_NUM',
         'GRP_ACTUAL_DATE'],as_index=False).count()[['GROUP_NUM','GRP_ACTUAL_DATE',parameter]].sort_values(by='GRP_ACTUAL_DATE')
                if ds.tail(1)[parameter].values[0]<15 :
                    del sup_partn_groupnum[sup]
                    pass
                else:
                    groupnum=ds.tail(1)['GROUP_NUM'].values[0] ###the newest group_num
                    partn_groupnum.update({partnum:groupnum})
                    sup_partn_groupnum[sup].update(partn_groupnum)
    return sup_partn_groupnum

def pre_prpcessing(rawdata,parameter):
    """
    Before doing config analysis,find the part_num and the newest group_num for a certain parameter in the dataframe. 
    Input: 
        rawdata is dataframe with commodity data. 
        parameter is the yaxis for config analysis. eg:parameter='FLANGE_HT_PROJECTION'
    Output: 
        partn_sup_groupnum: dict structure

    """
    partn_sup_groupnum=dict()
    partn_groupnum=dict()
    for partnum in rawdata['PART_NUM'].unique():
        raw_part=rawdata[rawdata['PART_NUM']==partnum]
        for sup in raw_part['SUPPLIER_NAME'].unique():
            partn_sup_groupnum.update({partnum:{}})
            df_motor=raw_part[raw_part['SUPPLIER_NAME']==sup]
            if len(df_motor['GROUP_NUM'].unique())<10:
                del partn_sup_groupnum[partnum]
                pass
            else:
                ds=df_motor.groupby(['GROUP_NUM',
         'GRP_ACTUAL_DATE'],as_index=False).count()[['GROUP_NUM','GRP_ACTUAL_DATE',parameter]].sort_values(by='GRP_ACTUAL_DATE')
                if ds.tail(1)[parameter].values[0]<15 :
                    del partn_sup_groupnum[partnum]
                    pass
                else:
                    groupnum=ds.tail(1)['GROUP_NUM'].values[0] ###the newest group_num
                    partn_groupnum.update({sup:groupnum})
                    partn_sup_groupnum[partnum].update(partn_groupnum)
    return partn_sup_groupnum

def cal_mean_shift(df_t,df_b,parameter):
    mean_shift=df_t[parameter].mean()-df_b[parameter].mean()
    return mean_shift
def cal_var_shift(df_t,df_b,parameter):
    var_shift=(df_t[parameter].var()+df_t[parameter].mean()**2)-(df_b[parameter].var()+df_b[parameter].mean()**2)
    return var_shift


def cal_attr_ratio(df_b,df_t,attr):

    """ new data's attr layer need in the baseline 
    Base line ratio is only calculate these attr 
    """
    df_b.dropna(subset=[attr],inplace=True)
    df_t.dropna(subset=[attr],inplace=True)
    t_ratio=pd.crosstab(df_t[attr],df_t['status'], normalize='columns').reset_index() 
    df_b1=df_b[df_b[attr].isin(t_ratio[attr].unique())]
    b_ratio=pd.crosstab(df_b1[attr],df_b1['status'], normalize='columns').reset_index() 
    try:
        df_ratio=pd.merge(b_ratio,t_ratio,on=[attr],how='outer')
        df_ratio.fillna(0,inplace=True)
        df_ratio.set_index(attr,inplace=True)
    except:
        df_t[attr].unique().tolist()
        print('attribute changed,{0} has {1}, but not included in the baseline'.format(attr,str(df_t[attr].unique())))
    
    return df_ratio,df_b1
              
              
              
def cal_contribution(df_t,df_b,parameter,attr,attr_layer,shift,baseline_ratio,trigger_ratio,con):
    #attr_layer='R',shift use mean_shift,var_shift
    if con == "var":
        c=(trigger_ratio*(df_t[df_t[attr]==attr_layer].mean()[parameter]**2+df_t[df_t[attr]==attr_layer].var()[parameter])-(baseline_ratio*((df_b[df_b[attr]==attr_layer].mean()[parameter])**2+df_b[df_b[attr]==attr_layer].var()[parameter])))/shift
    else:
        c=(trigger_ratio*df_t[df_t[attr]==attr_layer].mean()[parameter]-baseline_ratio*df_b[df_b[attr]==attr_layer].mean()[parameter])/shift
    return c


def lvl_two_shift(parameter,attr,df_b,df_t):
    try:
        df_ratio,df_b1=cal_attr_ratio(df_b,df_t,attr)# baseline and Newdata need the same attr
        mean_shift=cal_mean_shift(df_t,df_b1,parameter)
        var_shift=cal_var_shift(df_t,df_b1,parameter)
        c_m=dict()
        c_v=dict()
        # parameter='FLANGE_HT_PROJECTION'
        # attr='MOTOR_LINE_NUM'
        for ind in df_ratio.index:
            attr_layer=len(df_ratio)
            baseline_ratio=df_ratio.loc[ind,'BASELINE']
            trigger_ratio=df_ratio.loc[ind,'NEWDATA']
            c_mean=cal_contribution(df_t,df_b1,parameter,attr,ind,mean_shift,baseline_ratio,trigger_ratio,"mean")
            c_var=cal_contribution(df_t,df_b1,parameter,attr,ind,var_shift,baseline_ratio,trigger_ratio,"var")
            c_m.update({ind:c_mean})
            c_v.update({ind:c_var})
    except:
        print ('lvl_two_shift is error because of attribute changed')
        
    return c_m,c_v

def gen_result(m_shift,v_shift):
    final=[]
    print(f"m_shift: {m_shift}")
    for attr in m_shift.keys():
        print(f"attr: {attr}")
        A=pd.Series(m_shift[attr]).dropna().to_frame()
        B=pd.Series(v_shift[attr]).dropna().to_frame()
        A.columns=['Mean Shift']
        B.columns=['Variance Shift']
        A.reset_index(inplace=True)
        B.reset_index(inplace=True)
        data_set=pd.merge(A,B,how='inner',on='index')
        data_set.rename(columns={'index':'atr'},inplace=True)
        data_set['attribute']=attr+'_ '+data_set['atr'].astype('str')
        data_set['col']=attr
        m_threshold=lvl_two_threshold(data_set,"mean")
        v_threshold=lvl_two_threshold(data_set,"var")
        data_set['Mean Threshold']=m_threshold[0]
        data_set['Variance Threshold']=v_threshold[0]
        final.append(data_set)
    final_result=pd.concat(final)
    return final_result

from scipy.stats import chi2
from scipy.optimize import fsolve
def lvl_two_threshold(data_set,con):
    df = len(data_set)-1
    vals=chi2.ppf([0.05],df)
    if con == "var":
        data_set['Variance Shift1']=[i if i >0 else 0 for i in data_set['Variance Shift'] ]
        func=lambda x:vals-sum((data_set['Variance Shift']-x)**2/x)
    else :
        data_set['Mean Shift1']=[i if i >0 else 0 for i in data_set['Mean Shift'] ]
        func=lambda x:vals-sum((data_set['Mean Shift1']-x)**2/x)
    threshold =fsolve(func,[10])
    return threshold




def plot(final_result,title,shift_layer='Mean Shift',threshold_layer='Mean Threshold'):
    import math
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.offline as offline  
    if shift_layer=='Mean Shift':
        final_result.sort_values(by='mean_thresold_ratio',ascending=False,inplace=True)
    else:
        final_result.sort_values(by='variance_thresold_ratio',ascending=False,inplace=True)
    attr_fac=final_result[final_result[shift_layer]-final_result[threshold_layer]>0]['col'].unique()
    ftitle=title+'_'+shift_layer

    fig = make_subplots(rows=len(attr_fac), cols=1,subplot_titles=[ftitle]*len(attr_fac)
                       ) 
    sub_title=[]
    for i,attr in enumerate(attr_fac):
        data_set=final_result[final_result['col']==attr]
        fig.add_trace(
            go.Bar(
                x=data_set.attribute.tolist(),
                y=data_set[shift_layer].tolist()
                ,name=shift_layer,legendgroup = str(i+1)
            ),
            row=i+1, col=1  # 1*1
        )
        fig.add_trace( go.Scatter(
            x=data_set.attribute.tolist(),
            y=data_set[threshold_layer].tolist()
            ,name=threshold_layer,legendgroup = str(i+1)
        ),
        row=i+1, col=1 # 1*1
    )
    fig.update_layout(height=2700,legend_tracegroupgap = 1550)
    offline.plot(fig,filename = './VCMA2/'+ftitle+'.html',  auto_open=False)


# def plot1(final_result,title,shift_layer='Mean Shift',threshold_layer='Mean Threshold'):
#     from PyPDF2 import PdfFileReader, PdfFileMerger
#     import math
#     from plotly.subplots import make_subplots
#     import plotly.graph_objects as go
#     import plotly.offline as offline
#     import matplotlib.pyplot as plt
#     if shift_layer=='Mean Shift':
#         final_result.sort_values(by='mean_thresold_ratio',ascending=False,inplace=True)
#     else:
#         final_result.sort_values(by='variance_thresold_ratio',ascending=False,inplace=True)
#     attr_fac=final_result[final_result[shift_layer]-final_result[threshold_layer]>0]['col'].unique()
#     ftitle=title+'_'+shift_layer
    
#     output = PdfFileMerger()
#     for i,attr in enumerate(attr_fac):
#         data_set=final_result[final_result['col']==attr]
#         fig=go.Figure(data=[go.Bar(
#         x=data_set.attribute.tolist(),
#         y=data_set[shift_layer].tolist()
#         ,name=shift_layer),
#         go.Scatter(
#         x=data_set.attribute.tolist(),
#         y=data_set[threshold_layer].tolist()
#         ,name=threshold_layer
#         )])

#         fig.update_layout(title=ftitle, height=1200, width=800, showlegend=True)
#         # fig.show()
#         fig.write_image(f'pdf_file.pdf')
#         pdf_file = PdfFileReader('pdf_file.pdf')
#         output.append(pdf_file) 
#     with open('./VCMA5/'+ftitle+".pdf", "wb") as output_stream:
#         output.write(output_stream)



def attr_par_corr(df_attr,k=4):
    """
    rawdata just have 1 attribute 
    
    """
    train_corr=df_attr.corr(method='kendall')
    cols=train_corr.nlargest(k,parameter)[parameter].index
    top_k=[(str(i)+' :'+str('{:.2f}'.format(train_corr.loc[i,parameter]))) for i in cols]
    if len(cols)>1:
        corr_topk=cols[1:]
        corr_topk=[i[:-13] if '_labelencoder' in i else i for i in corr_topk]
        print('Select top 3 Kendall correlation feature is:',top_k[1:]  )
    else:
        top_k=[]
        corr_topk=[]
#     cm=np.corrcoef(df_attr[cols].values.T)
#     hm=plt.subplots(figsize=(10,10))
#     hm=sns.heatmap(df_attr[cols].corr(method='kendall'), vmin=0, vmax=.8,square=True,annot=True)
#     plt.show()
    return top_k,corr_topk

def kbest(df_attr,con_attr,parameter):
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.feature_selection import f_regression
    df_attr.dropna(subset=con_attr,inplace=True)
    X1=np.array(df_attr[con_attr])
    y1=np.array(df_attr[[parameter]])
    # select feature by person coefficient
    skb = SelectKBest(score_func=f_regression, k=3)
    skb.fit(X1, y1.ravel())
    regression_topk=[df_attr[con_attr].columns[i] for i in skb.get_support(indices = True)]
    regression_topk=[i[:-13] if '_labelencoder' in i else i for i in regression_topk]
    
    print('Select f_regression feature is:', regression_topk)
    anova_filter = SelectKBest(f_classif, k=3)
    anova_filter.fit(X1, y1.ravel())
    anova_topk=[df_attr[con_attr].columns[i] for i in anova_filter.get_support(indices = True)]
    anova_topk=[i[:-13] if '_labelencoder' in i else i for i in anova_topk]
    print('Select anova feature is:', anova_topk)
    return regression_topk,anova_topk

def do_anova(result,t1,target):
    result2=[]    
    anova_results_list=[]
    for m in t1:#
#         Attribute_count=result.groupby(m).count()
        result1=result.copy()
#         count_filter=list(Attribute_count[Attribute_count['ORT_PRODUCT_NAME']>20].index)#ORT_PRODUCT_NAME is not null,so use ORT_PRODUCT_NAME best
#         result1=result[result[m].isin(count_filter)]
        result2.append(result1)
        string=str(target)+'~'   
        string+=str('+'.join([m]))
        anova_results=anova_lm(ols(string,result1).fit())
        anova_results_list.append(anova_results.iloc[:-1,:])                   
    if len(result2)>0:
        anova_results_list1=pd.concat(anova_results_list)
        anova_results_list1.sort_values(by=['PR(>F)','F'],ascending=[True,False],inplace=True)
        anova_results2=anova_results_list1.iloc[:3,:]
        anova_results2.reset_index(inplace=True)
        anova_results2.rename(columns={'index':'ATTR'},inplace=True)
        return anova_results2

# def select_top(corr_topk,regression_topk,anova_results):
#     corr_df=pd.DataFrame({'ATTR':corr_topk,'CORR_VALUE':[0.2]*len(corr_topk)})
# #     regression_topk,anova_topk= kbest(df_attr1,col_encoder,parameter)
#     regression_df=pd.DataFrame({'ATTR':regression_topk,'REGRESSION_VALUE':[0.2]*len(regression_topk)})
# #     anova_results=do_anova(df_attr1,col_encoder,parameter)
#     anova_results['ANOVA_VALUE']=0.6
#     merge_d=pd.merge(corr_df,regression_df,how='outer',on='ATTR')
#     merge_df=pd.merge(merge_d,anova_results,how='outer',on='ATTR')
#     merge_df.fillna(0,inplace=True)
#     merge_df['SCORE']=merge_df['CORR_VALUE']+merge_df['REGRESSION_VALUE']+merge_df['ANOVA_VALUE']
#     merge_df.sort_values(by=['SCORE'],ascending=[False],inplace=True)
#     return merge_df

def select_top(corr_topk,regression_topk,anova_topk):#anova_results):
    corr_df=pd.DataFrame({'ATTR':corr_topk,'CORR_VALUE':[0.2]*len(corr_topk)})
    #     regression_topk,anova_topk= kbest(df_attr1,col_encoder,parameter)
    regression_df=pd.DataFrame({'ATTR':regression_topk,'REGRESSION_VALUE':[0.2]*len(regression_topk)})
    anova_results=pd.DataFrame({'ATTR':anova_topk,'ANOVA_VALUE':[0.6]*len(anova_topk)})
    #     anova_results=do_anova(df_attr1,col_encoder,parameter)
    #     anova_results['ANOVA_VALUE']=0.6
    merge_d=pd.merge(corr_df,regression_df,how='outer',on='ATTR')
    merge_df=pd.merge(merge_d,anova_results,how='outer',on='ATTR')
    merge_df.fillna(0,inplace=True)
    merge_df['SCORE']=merge_df['CORR_VALUE']+merge_df['REGRESSION_VALUE']+merge_df['ANOVA_VALUE']
    merge_df.sort_values(by=['SCORE'],ascending=[False],inplace=True)
    return merge_df

def remove_single_factor_group(df_attr,con_attr):
    from sklearn.feature_selection import VarianceThreshold
    from sklearn import preprocessing
    cols_n=df_attr.columns.tolist()
    le = preprocessing.LabelEncoder()
    col_encoder=[]
    single_group=[]
    if df_attr.dtypes[df_attr.dtypes==object].shape[0]>0:
        for col in df_attr.dtypes[df_attr.dtypes==object].index:
            if col in con_attr:
                if df_attr[col].nunique()==1:
                    single_group.appemd(col)
                t=col+'_labelencoder'
                cols_n.remove(col)
                col_encoder.append(t)
                le.fit(df_attr[col].values)
                df_attr[t]=le.transform(df_attr[col].values)
    encoder_col=[i for i in df_attr.columns if '_labelencoder'in i]
    df_attr1=df_attr[encoder_col+cols_n]
    var_thres=VarianceThreshold(threshold=0)
    var_thres.fit(df_attr1)
    df_attr1.columns[var_thres.get_support()]
    constant_columns = [column for column in df_attr1.columns
    if column not in df_attr1.columns[var_thres.get_support()]]
    remove_constant_columns=[i[:-13] for i in constant_columns ]
    con_attr1=[i for i in con_attr if i not in remove_constant_columns]
    col_encoder=[i for i in col_encoder if i not in constant_columns]
    check_encoder=[i[:-13] for i in col_encoder]
    col_encoder=[i for i in con_attr1 if i not in check_encoder ]+col_encoder
    
    ###remove only '' in attribute
    
    con_attr1=[i if len(df_attr[i].unique())>1 else '' for i in con_attr1]
    col_encoder=[i if len(df_attr[i].unique())>1 else '' for i in col_encoder]
    if '' in con_attr1:
        con_attr1=[i for i in set(con_attr1)]
        con_attr1.remove('')
    if '' in col_encoder:
        col_encoder=[i for i in set(col_encoder)]
        col_encoder.remove('')
    return con_attr1,col_encoder,df_attr



def attr_encode(df_attr,con_attr):
    from sklearn.feature_selection import VarianceThreshold
    from sklearn import preprocessing
    cols_n=df_attr.columns.tolist()
    le = preprocessing.LabelEncoder()
    col_encoder=[]
    single_group=[]
    if df_attr.dtypes[df_attr.dtypes==object].shape[0]>0:
        for col in df_attr.dtypes[df_attr.dtypes==object].index:
            if col in con_attr:
                if df_attr[col].nunique()==1:###new add
                    single_group.append(col)
                else:
                    t=col+'_labelencoder'
                    cols_n.remove(col)
                    col_encoder.append(t)
                    le.fit(df_attr[col].values)
                    df_attr[t]=le.transform(df_attr[col].values)
    encoder_col=[i for i in df_attr.columns if '_labelencoder'in i]
    df_attr1=df_attr[encoder_col+cols_n]
#     var_thres=VarianceThreshold(threshold=0)
#     var_thres.fit(df_attr1)
#     df_attr1.columns[var_thres.get_support()]
#     constant_columns = [column for column in df_attr1.columns
#     if column not in df_attr1.columns[var_thres.get_support()]]
#     remove_constant_columns=[i[:-13] for i in constant_columns ]
    con_attr1=[i for i in con_attr if i not in single_group]
    check_encoder=[i[:-13] for i in col_encoder]
    col_encoder=[i for i in con_attr1 if i not in check_encoder and i not in single_group]+col_encoder
    
#     ###remove only '' in attribute
    
#     con_attr1=[i if len(df_attr[i].unique())>1 else '' for i in con_attr1]
#     col_encoder=[i if len(df_attr[i].unique())>1 else '' for i in col_encoder]
#     if '' in con_attr1:
#         con_attr1=[i for i in set(con_attr1)]
#         con_attr1.remove('')
#     if '' in col_encoder:
#         col_encoder=[i for i in set(col_encoder)]
#         col_encoder.remove('')
    return con_attr1,col_encoder,df_attr

def remove_single_group(df_attr,con_attr):
    single_group=[]
    for col in con_attr:
        if df_attr[col].nunique()==1:
            single_group.append(col)
    con_attr1=[i for i in con_attr if i not in single_group]
    return con_attr1

########



def isNaN(string):
    return string != string
def generate_greatest_boxplot(mod,df_motor,xaxis,yaxis,title,part_num):
    sample = copy.deepcopy(df_motor)
#     xaxis = 'GROUP_NUM'
#     yaxis = 'AXIAL_RRO'

    x_axis = 'Group_Num'
    mean_sample = sample.groupby([xaxis])[yaxis].mean().reset_index()
    if mod=='MOTOR':
        mod1='MO'
        real_list = []
        for i in list(map(int, mean_sample[xaxis])):
            real_list.append('202'+str(i%10)+'.'+str(i//10 - i//1000*100)+'.'+str(i//1000))
        mean_sample[x_axis] = real_list

        real_list = []
        for i in list(map(int, sample[xaxis])):
            real_list.append('202'+str(i%10)+'.'+str(i//10 - i//1000*100)+'.'+str(i//1000))
        sample[x_axis] = real_list
    else:
        mod1=mod
        mean_sample[x_axis] = mean_sample[xaxis]
        sample[x_axis] = sample[xaxis]


    mean_sample1 = mean_sample.sort_values(by=[x_axis]).reset_index()
    sample1 = sample.sort_values(by=[x_axis]).reset_index()

    aplot = ggplot() +\
    geom_boxplot(sample1,aes(x=x_axis,y = yaxis)) +\
    geom_line(mean_sample1, aes(x=x_axis,y = yaxis,group = 1), color='pink', size=1 )+\
    theme(panel_spacing = 0.45)+\
    ggtitle(title)+\
    theme(figure_size = (10, 5))+\
    theme(axis_text_x = element_text(angle=90, hjust=1,size=5)) 
    
    if mod=='MOTOR':
        aplot = aplot + scale_x_discrete(labels=lambda X: [i[8]+i[5]+i[6]+i[3] for i in X])

    
    if (part_num in SPEC[SPEC['COMMODITY'] == mod1]['PART_NUM'].unique().tolist()):
        SPEC1 = SPEC[SPEC['COMMODITY'] == mod1]
        SPEC2 = SPEC1[SPEC1['PART_NUM'] == part_num]
        if(yaxis in SPEC2['QPM_PARAMETER'].unique().tolist()):
            SPEC3 = SPEC2[SPEC2['QPM_PARAMETER'] == yaxis]
            TARGET = SPEC3['Target'].unique().tolist()[0]
            USL = SPEC3['USL'].unique().tolist()[0]
            LSL = SPEC3['LSL'].unique().tolist()[0]
            if not isNaN(TARGET):
                aplot = aplot + geom_hline(yintercept = float(TARGET), color='green')
            if not isNaN(USL):
                aplot = aplot + geom_hline(yintercept = float(USL), color='red')
            if not isNaN(LSL):
                aplot = aplot + geom_hline(yintercept = float(LSL), color='red')
    
    
    return (aplot)

def generate_newest_boxplot(mod,df_motor,zaxis,yaxis,title,part_num):
    sample = copy.deepcopy(df_motor)
    if mod == 'MOTOR':
        mod1 = 'MO'
    else:
        mod1 = mod

#     yaxis = 'AXIAL_RRO'
#     zaxis = 'MOTOR_LINE_NUM'
    z_axis = zaxis+' '
    
    real_list = []
    for i in list(map(int, sample[zaxis])):
        real_list.append(str(i))
    sample[z_axis] = real_list
    
    mean_sample = sample.groupby([zaxis])[yaxis].mean().reset_index()
    real_list = []
    for i in list(map(int, mean_sample[zaxis])):
        real_list.append(str(i))
    mean_sample[z_axis] = real_list

    mean_sample1 = mean_sample.sort_values(by=[z_axis]).reset_index()
    sample1 = sample.sort_values(by=[z_axis]).reset_index()

    aplot = ggplot() +\
    xlab(zaxis) +\
    geom_boxplot(sample1,aes(x=z_axis,y = yaxis)) +\
    scale_x_discrete(labels=lambda X: [z_axis+i for i in X])+\
    geom_line(mean_sample1, aes(x=z_axis,y = yaxis,group = 1), color='pink', size=1 )+\
    theme(panel_spacing = 0.45)+\
    ggtitle(title)+\
    theme(figure_size = (10, 7))+\
    theme(axis_text_x = element_text(angle=90, hjust=1,size=5))
    
    
    
    if (part_num in SPEC[SPEC['COMMODITY'] == mod1]['PART_NUM'].unique().tolist()):
        SPEC1 = SPEC[SPEC['COMMODITY'] == mod1]
        SPEC2 = SPEC1[SPEC1['PART_NUM'] == part_num]
        if(yaxis in SPEC2['QPM_PARAMETER'].unique().tolist()):
            SPEC3 = SPEC2[SPEC2['QPM_PARAMETER'] == yaxis]
            TARGET = SPEC3['Target'].unique().tolist()[0]
            USL = SPEC3['USL'].unique().tolist()[0]
            LSL = SPEC3['LSL'].unique().tolist()[0]
            if not isNaN(TARGET):
                aplot = aplot + geom_hline(yintercept = float(TARGET), color='green')
            if not isNaN(USL):
                aplot = aplot + geom_hline(yintercept = float(USL), color='red')
            if not isNaN(LSL):
                aplot = aplot + geom_hline(yintercept = float(LSL), color='red')
    
    
    return (aplot)

def generate_boxplot(df_motor,xaxis,yaxis,part_num):
    sample = copy.deepcopy(df_motor)
#     xaxis = 'GROUP_NUM'
#     yaxis = 'AXIAL_RRO'

    
    x_axis = 'Group_Num'
    count = 0
    real_list = []
    for i in df_motor['GROUP_NUM']:
        if i == latest_group_num:
            real_list.append('Latest_Group')
            count = count + 1
        else:
            real_list.append('Base_Group')
            
    sample[x_axis] = real_list
    
    mean_sample = sample.groupby([x_axis])[yaxis].mean().reset_index()
    

    mean_sample1 = mean_sample.sort_values(by=[zaxis,x_axis]).reset_index()
    sample1 = sample.sort_values(by=[zaxis,x_axis]).reset_index()

    aplot = ggplot() +\
    geom_boxplot(sample1,aes(x=x_axis,y = yaxis)) +\
    geom_line(mean_sample1, aes(x=x_axis,y = yaxis,group = 1), color='pink', size=1 )+\
    theme(panel_spacing = 0.45)+\
    theme(figure_size = (10, 25))+\
    theme(axis_text_x = element_text(angle=90, hjust=1,size=5))
    
    
    if (part_num in SPEC[SPEC['COMMODITY'] == 'MO']['PART_NUM'].unique().tolist()):
        SPEC1 = SPEC[SPEC['COMMODITY'] == 'MO']
        SPEC2 = SPEC1[SPEC1['PART_NUM'] == part_num]
        if(yaxis in SPEC2['QPM_PARAMETER'].unique().tolist()):
            SPEC3 = SPEC2[SPEC2['QPM_PARAMETER'] == yaxis]
            TARGET = SPEC3['Target'].unique().tolist()[0]
            USL = SPEC3['USL'].unique().tolist()[0]
            LSL = SPEC3['LSL'].unique().tolist()[0]
            if not isNaN(TARGET):
                aplot = aplot + geom_hline(yintercept = float(TARGET), color='green')
            if not isNaN(USL):
                aplot = aplot + geom_hline(yintercept = float(USL), color='red')
            if not isNaN(LSL):
                aplot = aplot + geom_hline(yintercept = float(LSL), color='red')
    
    
    return (aplot)


def plot2(report_date,mod,final_result,title,shift_layer='Mean Shift',threshold_layer='Mean Threshold'):
    from PyPDF2 import PdfFileReader, PdfFileMerger
    import math
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.offline as offline
    import matplotlib.pyplot as plt
    if shift_layer=='Mean Shift':
        final_result.sort_values(by='mean_thresold_ratio',ascending=False,inplace=True)
    else:
        final_result.sort_values(by='variance_thresold_ratio',ascending=False,inplace=True)
    attr_fac=final_result[final_result[shift_layer]-final_result[threshold_layer]>0]['col'].unique()
    ftitle=title+'_'+shift_layer
    
    output = PdfFileMerger()
    for i,attr in enumerate(attr_fac):
        data_set=final_result[final_result['col']==attr]
        fig=go.Figure(data=[go.Bar(
        x=data_set.attribute.tolist(),
        y=data_set[shift_layer].tolist()
        ,name=shift_layer),
        go.Scatter(
        x=data_set.attribute.tolist(),
        y=data_set[threshold_layer].tolist()
        ,name=threshold_layer
        )])
        fig.update_layout(title=ftitle, title_font_size = 12,title_x= 0.5, height=1067, width=800, showlegend=True)
        # fig.update_layout(title=ftitle, height=1200, width=800, showlegend=True)
        # fig.show()
        fig.write_image(f'pdf_file.pdf')
        pdf_file = PdfFileReader('pdf_file.pdf')
        output.append(pdf_file) 
    with open('./'+mod+'/'+report_date+'_'+ftitle+".pdf", "wb") as output_stream:
        output.write(output_stream)



# def plot1(report_date,mod,df_motor,parameter,partnum,new_groupnum,final_result,title,shift_layer='Mean Shift',threshold_layer='Mean Threshold'):
#     from PyPDF2 import PdfFileReader, PdfFileMerger
#     import math
#     from plotly.subplots import make_subplots
#     import plotly.graph_objects as go
#     import plotly.offline as offline
#     import matplotlib.pyplot as plt
#     if shift_layer=='Mean Shift':
#         final_result.sort_values(by='mean_thresold_ratio',ascending=False,inplace=True)
#     else:
#         final_result.sort_values(by='variance_thresold_ratio',ascending=False,inplace=True)
#     attr_fac=final_result[final_result[shift_layer]-final_result[threshold_layer]>0]['col'].unique()
#     ftitle=title+'_'+shift_layer
    
    
#     output = PdfFileMerger()
#     for i,attr in enumerate(attr_fac):
#         data_set=final_result[final_result['col']==attr]
#         greatest_line = data_set.atr.tolist()[0]
#         print('data_set.atr.tolist() is ',data_set.atr.tolist())
#         fig=go.Figure(data=[go.Bar(
#         x=data_set.attribute.tolist(),
#         y=data_set[shift_layer].tolist()
#         ,name=shift_layer),
#         go.Scatter(
#         x=data_set.attribute.tolist(),
#         y=data_set[threshold_layer].tolist()
#         ,name=threshold_layer
#         )])
#         fig.update_layout(title=ftitle, title_font_size = 12,title_x= 0.5, height=700, width=1000, showlegend=True)
#         # fig.show()
#         fig.write_image(f'pdf_file.pdf')
#         pdf_file = PdfFileReader('pdf_file.pdf')
#         output.append(pdf_file) 

#         df_motor_new = copy.deepcopy(df_motor)
#         df_motor_new = df_motor_new[df_motor_new['GROUP_NUM'] == new_groupnum]
# #         print('df_motor_new shape is {}'.format(df_motor_new.shape))
#         title2 = 'GROUP_NUM '+str(new_groupnum)
#         aplot = generate_newest_boxplot(mod,df_motor_new,attr,parameter,title2,str(partnum))
#         aplot.save(f'pdf_file2.pdf')
#         pdf_file = PdfFileReader('pdf_file2.pdf')
#         output.append(pdf_file)
        
#         df_motor_new1 = copy.deepcopy(df_motor)
#         df_motor_new1 = df_motor_new1[df_motor_new1[attr] == greatest_line]
# #         print('greatest_line: ',greatest_line)
# #         print('attr: ',attr)
# #         print()
# #         print('df_motor_new1 shape is {}'.format(df_motor_new1.shape))
#         title3 = str(attr)+' '+str(greatest_line)
#         aplot1 = generate_greatest_boxplot(mod,df_motor_new1,'GROUP_NUM',parameter,title3,str(partnum))
#         aplot1.save(f'pdf_file1.pdf')
#         pdf_file = PdfFileReader('pdf_file1.pdf')
#         output.append(pdf_file)  

#     with open('./'+mod+'/'+report_date+'_'+ftitle+".pdf", "wb") as output_stream:
#         output.write(output_stream)
def generate_new_boxplot(df_motor,df_motor3,zaxis,yaxis,fill_axis,title,part_num):
    
#     yaxis = 'AXIAL_RRO'
#     zaxis = 'MOTOR_LINE_NUM'
#     fill_axis = 'baseline'/'new plot'
    sample = copy.deepcopy(df_motor)

    z_axis = zaxis+' '

    real_list = []
    for i in list(map(int, sample[zaxis])):
        real_list.append(str(i))
    sample[z_axis] = real_list

    mean_sample = sample.groupby([zaxis,'Filter'])[yaxis].mean().reset_index()
    real_list = []
    for i in list(map(int, mean_sample[zaxis])):
        real_list.append(str(i))
    mean_sample[z_axis] = real_list

    fill_axis = []
    attr_list = mean_sample[zaxis].tolist()
    filter_list = mean_sample['Filter'].tolist()
    for i in range(len(mean_sample['Filter'])):
        fill_axis.append(str(zaxis)+' '+str(attr_list[i])+'\n'+str(filter_list[i]))
    mean_sample['Real_Filter'] = fill_axis

    mean_sample1 = mean_sample.sort_values(by=[z_axis]).reset_index()
    sample1 = sample.sort_values(by=[z_axis]).reset_index()

    sample3 = copy.deepcopy(df_motor3)

    real_list = []
    for i in list(map(int, sample3[zaxis])):
        real_list.append(str(i))
    sample3[z_axis] = real_list
    
# ,fill = fill_axis
    aplot = ggplot(aes(x='Real_Filter',y = yaxis)) +\
    geom_boxplot(sample1,fill = '#C6E2FF') +\
    scale_x_discrete(labels=lambda X: [i[len(i)-8:]+'\n'+i[:len(i)-9] for i in X])+\
    scale_fill_discrete(labels = ['baseline','new data'])+\
    xlab(z_axis)+\
    geom_boxplot(sample3,fill = '#EE9999') +\
    geom_point(mean_sample1,aes(group = 4), shape ='+', size=5 )+\
    geom_line(mean_sample1,aes(group = 4), color = 'pink', size=1 )+\
    geom_text(mean_sample1,aes(group = 4,label = "{0}".format(yaxis)), format_string='{:.4f}', va="bottom",ha = "left")+\
    theme(panel_spacing = 0.45)+\
    ggtitle(title)+\
    theme(figure_size = (10, 7))

    # geom_text(sample1,aes(y=-.5, label=attr),
    #              position=dodge_text,
    #              color=ccolor, size=8, angle=45, va='top')+\
    # geom_text(sample1,aes(label='Filter'),
    #              position=dodge_text,
    #              size=8, va='bottom', format_string='{}%')+\

    if (part_num in SPEC[SPEC['COMMODITY'] == 'MO']['PART_NUM'].unique().tolist()):
        SPEC1 = SPEC[SPEC['COMMODITY'] == 'MO']
        SPEC2 = SPEC1[SPEC1['PART_NUM'] == part_num]
        if(yaxis in SPEC2['QPM_PARAMETER'].unique().tolist()):
            SPEC3 = SPEC2[SPEC2['QPM_PARAMETER'] == yaxis]
            TARGET = SPEC3['Target'].unique().tolist()[0]
            USL = SPEC3['USL'].unique().tolist()[0]
            LSL = SPEC3['LSL'].unique().tolist()[0]
            if not isNaN(TARGET):
                aplot = aplot + geom_hline(yintercept = float(TARGET), color='green')
            if not isNaN(USL):
                aplot = aplot + geom_hline(yintercept = float(USL), color='red')
            if not isNaN(LSL):
                aplot = aplot + geom_hline(yintercept = float(LSL), color='red')
    
    
    return (aplot)


def plot1(df_motor,parameter,partnum,new_groupnum,final_result,title,shift_layer='Mean Shift',threshold_layer='Mean Threshold'):
    from PyPDF2 import PdfFileReader, PdfFileMerger
    import math
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.offline as offline
    import matplotlib.pyplot as plt
    if shift_layer=='Mean Shift':
        final_result.sort_values(by='mean_thresold_ratio',ascending=False,inplace=True)
    else:
        final_result.sort_values(by='variance_thresold_ratio',ascending=False,inplace=True)
    attr_fac=final_result[final_result[shift_layer]-final_result[threshold_layer]>0]['col'].unique()
    ftitle=title+'_'+shift_layer
    
    
    output = PdfFileMerger()
    for i,attr in enumerate(attr_fac):
        print('attr is ',attr)
        data_set=final_result[final_result['col']==attr]
        greatest_line = data_set.atr.tolist()[0]
        print('data_set.atr.tolist() is ',data_set.atr.tolist())
        fig=go.Figure(data=[go.Bar(
        x=data_set.attribute.tolist(),
        y=data_set[shift_layer].tolist()
         ,name=shift_layer
        ),
        go.Scatter(
        x=data_set.attribute.tolist(),
        y=data_set[threshold_layer].tolist()
         ,name=threshold_layer
        )])
        
        print('title is ',ftitle)
        print('shift is ',data_set[shift_layer].tolist())
        print('threshold is',data_set[threshold_layer].tolist())

        fig.update_layout(title=ftitle, height=700, width=1000, showlegend=True)
        # fig.show()
        fig.write_image(f'pdf_file.pdf')
        pdf_file = PdfFileReader('pdf_file.pdf')
        output.append(pdf_file) 
 
        df_motor_new = copy.deepcopy(df_motor)
        fill_axis = []
        for item in df_motor_new['GROUP_NUM']:
            if item  == new_groupnum:
                fill_axis.append('new data')
            else:
                fill_axis.append('baseline')
        df_motor_new['Filter'] = fill_axis

        fill_axis = []
        attr_list = df_motor_new[attr].tolist()
        filter_list = df_motor_new['Filter'].tolist()
        for i in range(len(df_motor_new['Filter'])):
            fill_axis.append(attr+' '+attr_list[i]+'\n'+filter_list[i])
        df_motor_new['Real_Filter'] = fill_axis


        df_motor_new_base = df_motor_new[df_motor_new['Filter'] =='baseline' ]
        df_motor_new_new = df_motor_new[df_motor_new['Filter'] =='new data' ]
        df_motor2 = pd.DataFrame(columns=df_motor.columns)
        df_motor3 = pd.DataFrame(columns=df_motor.columns)
        greatest_item = 0
        print(df_motor_new[attr].unique().tolist())
        for item in data_set.atr.tolist():  
            df_motor2 = df_motor2.append(df_motor_new_base[df_motor_new_base[attr] == item],ignore_index = True)
            df_motor2 = df_motor2.append(df_motor_new_new[df_motor_new_new[attr] == item],ignore_index = True)
            if greatest_item == 0:
                df_motor3 = df_motor3.append(df_motor_new_base[df_motor_new_base[attr] == item],ignore_index = True)
                df_motor3 = df_motor3.append(df_motor_new_new[df_motor_new_new[attr] == item],ignore_index = True)
                greatest_item = 1

#         df_motor_new = df_motor_new[df_motor_new['GROUP_NUM'] == new_groupnum]
        title2 = 'GROUP_NUM '+str(new_groupnum)
        aplot = generate_new_boxplot(df_motor2,df_motor3,attr,parameter,'Filter',title2,str(partnum))
        aplot.save(f'pdf_file2.pdf')
        pdf_file = PdfFileReader('pdf_file2.pdf')
        output.append(pdf_file)
        
#         df_motor_new1 = copy.deepcopy(df_motor)
#         df_motor_new1 = df_motor_new1[df_motor_new1[attr] == greatest_line]

#         title3 = str(attr)+' '+str(greatest_line)
#         aplot1 = generate_greatest_boxplot(df_motor_new1,'GROUP_NUM',parameter,title3,str(partnum))
#         aplot1.save(f'pdf_file1.pdf')
#         pdf_file = PdfFileReader('pdf_file1.pdf')
#         output.append(pdf_file)  
     
    with open('./MOTOR9/'+ftitle+".pdf", "wb") as output_stream:
        output.write(output_stream)

def attr_ana(mod):
    if mod=='MOTOR':
        attr_list=['VENDOR_ID',
    # 'SERIAL_SHORT_NUM',
    'BASEPLATE_LOT_NUM',
    # 'BASEPLATE_SERIAL_NUM',
    'BASEPLATE_MOLD_NUM',
    'BASEPLATE_CAVITY_NUM',
    'MOTOR_LINE_NUM',
    'RAMP_DATE_CODE',
    'RAMP_MOLD_NUM',
    'RAMP_CAVITY_NUM',
    'DIVERTER_DATE_CODE',
    'DIVERTER_MOLD_NUM',
    'DIVERTER_CAVITY_NUM',
    'CSID_SUPPLIER',
    'CSID_RAWMAT_DATE_CODE',
    'CSID_MOLDING_MACHINES',
    'CSID_MOLDING_DATE_CODE']
    if mod=='VCMA':
        attr_list=['BONDING_MACHINE_NUM',
                    'SUPPLIER_NAME',
                    'MAGNET_LOT_NUM',
                    'POLE_LOT_NUM',
                    'MAGNATIZING_CAVITY',
                    'LINE',
                    'DATECODE',
                    'SUBTIER_MAGNET_NAME',
                    'SUBTIER_MAGNET_ID',
                    'SUBTIER_POLE_NAME',
                    'SUBTIER_POLE_ID',
                    'SUBTIER_LATCH_NAME',
                    'SUBTIER_LATCH_ID',
                    'SUBTIER_LATCH_PIN_NA ME',
                    'SUBTIER_LATCH_PIN_ID',
                    'SUBTIER_STOP_PIN_NAME',
                    'SUBTIER_STOP_PIN_ID',
                    'SUBTIER_ODSPACER_NAME',
                    'SUBTIER_ODSPACER_ID',
                    'SUBTIER_ODSPACER_LOT_NUM',
                    'SUBTIER_IDSPACER_NAME',
                    'SUBTIER_IDSPACER_ID',
                    'SUBTIER_IDSPACER_LOT_NUM',
                    'SUBTIER_REAR_SPACER_NAME',
                    'SUBTIER_REAR_SPACER_ID',
                    'SUBTIER_REAR_SPACER_LOT_NUM',
                    'SUBTIER_CLOCK_PIN_NAME',
                    'SUBTIER_CLOCK_PIN_ID',
                    'SUBTIER_ODSTOP_PIN_N AME',
                    'SUBTIER_ODSTOP_PIN_ID',
                    'SUBTIER_ODSTOP_PIN_LOT_NUM',
                    'SUBTIER_IDSTOP_PIN_NAME',
                    'SUBTIER_IDSTOP_PIN_ID',
                    'SUBTIER_IDSTOP_PIN_LOT_NUM',
                    'SUBTIER_ODCRASH_STOP_NAME',
                    'SUBTIER_ODCRASH_STOP_ID',
                    'SUBTIER_IDCRASH_STOP_NAME',
                    'SUBTIER_IDCRASH_STOP_ID']
    if mod=='DSP':
        attr_list=['MOLD_NUM',
                    'CAVITY_NUM']
    if mod=='CLAMP':
        attr_list=['CLAMP_DATECODE',
                    'TYPE_PROCESS',
                    'MACHINE_TOOL_LINE_NO',
                    'CLAMP_LOT_TYPE',
                    'CLAMP_SAMPLE_NUM']
    return attr_list

def expert_attr_list(mod):
    xls = pd.ExcelFile('Sheet.xlsx')
    newsheet = pd.read_excel(xls,mod)
    par_name = newsheet.columns.values.tolist()[0]
    sub_ctq = newsheet[newsheet[par_name] == parameter]
    for j in newsheet.columns.values.tolist():
        if not ('Yes' in str(sub_ctq[j])):
            if not ('yes' in str(sub_ctq[j])):
                continue
        expert_attr.append(j)
    return expert_attr

if __name__ == "__main__":
    day = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    print(day)
    qpm_trigger_id=pd.read_csv("/srv/shiny-server/QPM_config/qpm_trigger_id.csv")
    qpm_trigger_id['PART_NUM']=qpm_trigger_id['PART_NUM'].astype('str')
    SPEC=pyreadr.read_r('/home/r_server1/Documents/software/data/QPM_SPEC.rda')[None]
    store_col=['SUPPLIER_NAME',
     'PART_NUM',
     'PARAMETER','ATTR','ATTR_INTIAL','AFTER_REMOVE_ONLY1_GROUP','ATTR_FINAL',
     'CORR_VALUE',
     'REGRESSION_VALUE',
     'ANOVA_VALUE',
     'SCORE']

    for mod,sup,partnum,parameter,new_groupnum in zip(qpm_trigger_id['COMMODITY'],qpm_trigger_id['SUPPLIER_NAME'],qpm_trigger_id['PART_NUM'],qpm_trigger_id['PARAMETER'],qpm_trigger_id['GROUP_NUM']):
        print (mod,sup,partnum,parameter,new_groupnum)
        if mod=='MOTOR':
        	group_num_s=new_groupnum.split('-')
        	new_groupnum=np.int64(group_num_s[-1]+group_num_s[1]+group_num_s[0])
        rawdata=pyreadr.read_r('/home/r_server1/Documents/software/data/INP_QPM_DASHBOARD_'+mod+'.rda')[None]
	    rawdata['PART_NUM']=rawdata['PART_NUM'].astype('str')
	    rawdata1=rawdata[rawdata['SUPPLIER_NAME']==sup]
        rawdata2=rawdata1[rawdata1['PART_NUM']==partnum]
        rawdata3=rawdata2.dropna(subset=[parameter])  
        if mod=='CLAMP':
            rawdata3=rawdata3[rawdata3['CLAMP_DATA_SOURCE_TYPE']=='PMP']
        if mod=='DSP':
            rawdata3=rawdata3[rawdata3['DATA_SOURCE_TYPE']=='PMP']     
        tttt=[]
        attr_compare=[]
        t=pd.read_csv('vcma_useful_attr.csv',index_col=0)
        t.drop(t.index,inplace=True)
        attr_list=attr_ana(mod)
        con_attr=[i for i in rawdata.columns if i in attr_list and i!='SUPPLIER_NAME']
        ####Step1
        # con_attr1,col_encoder,df_attr=remove_single_factor_group(rawdata3,con_attr)
        con_attr1,col_encoder,df_attr=attr_encode(rawdata3,con_attr)
        #                 print('after  attr_encode, the length of attrinute:{0},{1}'.format(len(con_attr1),con_attr1))
        df_attr_encoder=df_attr[col_encoder+[parameter]]
        df_attr1=df_attr[con_attr1+[parameter]]
        df_attr1=df_attr1[df_attr1[parameter].isin([''])==False]
        if len(df_attr1)>0:
            try:                    
                top_k,corr_topk=attr_par_corr(df_attr1,4)
                regression_topk,anova_topk= kbest(df_attr_encoder,col_encoder,parameter)
        #                     anova_results=do_anova(df_attr1,con_attr1,parameter)
                merge_df=select_top(corr_topk,regression_topk,anova_topk)
                merge_df['SUPPLIER_NAME']=sup
                merge_df['PART_NUM']=partnum
                merge_df['PARAMETER']=parameter
                merge_df['ATTR_INTIAL']='the length of attrinute:{0},{1}'.format(len(con_attr),con_attr)
                merge_df['AFTER_REMOVE_ONLY1_GROUP']='the length of attrinute:{0},{1}'.format(len(con_attr1),con_attr1)
                attr_compare=merge_df['ATTR'].tolist()
                merge_df['ATTR_FINAL']='the length of final attrinute:{0},{1}'.format(len(attr_compare),attr_compare)
        #         merge_df[store_col].to_csv('/home/r_server1/Documents/software/result/vcma/'+sup+'_'+str(partnum)+'_'+parameter+'.csv')
                t=t.append(merge_df[store_col],ignore_index=True)                     
            except :
                tttt.append(sup+'_'+str(partnum)+'_'+parameter)
    #                     print ('{} is error'.format(sup+'_'+str(partnum)+'_'+parameter))


        df_motor=rawdata3.copy()
        m_shift=dict()
        v_shift=dict()
        final=[]
        split_con=[]
        check_attr=[]
        attr_detail=[]
        expert_attr=[]

        if mod in ('VCMA','MOTOR'):
            expert_attr=expert_attr_list(mod)
            exp_attr_list= [i for i in set(expert_attr+ attr_compare) if 'SUPPLIER_NAME' not in i and i in df_motor.columns]
            total=remove_single_group(df_motor[exp_attr_list+[parameter]],exp_attr_list)
        else:
            total=attr_compare.copy()
        if len(total)>0:
            df_b=df_motor[total+[parameter]][df_motor['GROUP_NUM'].isin([new_groupnum])==False]
            df_t=df_motor[total+[parameter]][df_motor['GROUP_NUM'].isin([new_groupnum])]
            df_b['status']='BASELINE'
            df_t['status']='NEWDATA'
            df_t.to_csv('./'+mod+'/'+day+'_newdata_'+sup+'('+str(partnum)+')_'+parameter+'_('+str(new_groupnum)+').csv')
            df_b.to_csv('./'+mod+'/'+day+'_baseline_'+sup+'('+str(partnum)+')_'+parameter+'_('+str(new_groupnum)+').csv')

            for attr in total:
                try:
                    c_m,c_v=lvl_two_shift(parameter,attr,df_b,df_t)
                    print(attr,parameter,c_m)
                    m_shift.update({attr:c_m})
                    v_shift.update({attr:c_v})
                    if len(c_m)==0 or len(c_v)==0:
                        del m_shift[attr]
                        del v_shift[attr]

                except:
                    split_con.append(sup+'_'+str(partnum)+'_'+parameter)
                    check_attr.append(attr)
                    attr_detail.append('attribute changed,{0} has {1}, but not included in the baseline'.format(attr,str(df_t[attr].unique())))

            if len(m_shift)>0:
                final_result=gen_result(m_shift,v_shift)
                final_result['mean_thresold_ratio']=(final_result['Mean Shift']-final_result['Mean Threshold'])/final_result['Mean Threshold']
                final_result['variance_thresold_ratio']=(final_result['Variance Shift']-final_result['Variance Threshold'])/final_result['Variance Threshold']
                title=sup+'('+str(partnum)+')_'+parameter+'_('+str(new_groupnum)+')'
                print(title,final_result)
                final_result.to_csv('./'+mod+'/'+day+'_'+title+".csv",index=False)
                if final_result.shape[0]>1:
                    if final_result[final_result['Mean Shift']-final_result['Mean Threshold']>0].shape[0]>0:
                        if mod in ('VCMA','MOTOR'):
                    	    plot1(df_motor,parameter,partnum,new_groupnum,final_result,title,'Mean Shift','Mean Threshold')
                        else:
                            plot2(report_date,mod,final_result,title,'Mean Shift','Mean Threshold')
                    if final_result[final_result['Variance Shift']-final_result['Variance Threshold']>0].shape[0]>0:
                    	if mod in ('VCMA','MOTOR'):
                            plot1(df_motor,parameter,partnum,new_groupnum,final_result,title,'Variance Shift','Variance Threshold')
                        else :
                            plot2(report_date,mod,final_result,title,'Variance Shift','Variance Threshold')
          
                print ('{0}_{1}_{2} config anaysis already done'.format(parameter,partnum,new_groupnum))
    t1=pd.pivot_table(t, values='SCORE', index=['SUPPLIER_NAME', 'PART_NUM','PARAMETER','ATTR_INTIAL','AFTER_REMOVE_ONLY1_GROUP','ATTR_FINAL'],
                    columns=['ATTR'], aggfunc=np.sum)
    t1.to_csv('/home/r_server1/Documents/software/useful_attr_0324.csv')  