#!/usr/bin/env python
# coding: utf-8

# In[1]:


# IMPORT LIBRARY

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cx_Oracle
import time
import seaborn as sns
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.preprocessing import MinMaxScaler
from io import StringIO
pd.set_option('display.float_format', lambda x: '%.2f' % x)
# cx_Oracle.init_oracle_client(lib_dir=r"C:\Users\NFNawawi\anaconda3\instantclient_21_3")

# Data Connection

def connection():
    dsn_tns = cx_Oracle.makedsn('10.23.0.41', '1521', service_name='PMS')
    conn = cx_Oracle.connect(user='report', password='report', dsn=dsn_tns)
    return conn

conn = connection()
cursor = conn.cursor()


# ## PEMUPUKAN

# In[2]:


query_job_rawat="""SELECT 
A.TYPE, 
b.jobgroupcode AS JOBGROUP_CODE, 
B.description AS JOBGROUP_DESC, 
C.JOBCODE as "KODE JOB LEVEL 3", 
c.description AS JOB_DESC 
FROM mekanisasi.m_jobgroupcat_rwttpl A, mekanisasi.m_jobgroup_rwttpl B, mekanisasi.m_job_rwttpl C
WHERE a.jobgroupcatcode = b.jobgroupcatcode
AND b.jobgroupcode = c.jobgroupcode"""

pemupukan=pd.read_sql(query_job_rawat,con=conn)


# In[3]:


dff_clustering=pemupukan[(pemupukan['JOBGROUP_DESC'].str.contains("Pemupukan"))&(pemupukan['TYPE']=='TM')]
dff_clustering['Clustering']=dff_clustering['JOB_DESC'].str.split(expand=True)[0]


# In[4]:


filter_jobcode = ['250150','250128','680501','250151','250324','250343']
dataclustering=dff_clustering[~dff_clustering['KODE JOB LEVEL 3'].isin(filter_jobcode)]


# # EMPQTYACT

# In[5]:


q_empqtyact  = """SELECT
    mekanisasi.empqtyact.tdate,
    mekanisasi.empqtyact.sitecode,
    mekanisasi.empqtyact.location,
    mekanisasi.empqtyact.maturetype,
    mekanisasi.empqtyact.jobcode,
    mekanisasi.empqtyact.mandorcode,
    mekanisasi.empqtyact.empmandor,
    mekanisasi.empqtyact.empno,
    mekanisasi.empqtyact.empcode,
    mekanisasi.empqtyact.qtyact,
    mekanisasi.empqtyact.material1,
    mekanisasi.empqtyact.qtymaterial1,
    mekanisasi.empqtyact.material2,
    mekanisasi.empqtyact.qtymaterial2,
    mekanisasi.empqtyact.material3,
    mekanisasi.empqtyact.qtymaterial3,
    mekanisasi.empqtyact.jobgroupcode
FROM
    mekanisasi.empqtyact
WHERE
        mekanisasi.empqtyact.tdate >= TO_TIMESTAMP('15-12-2022 00:00:00', 'dd-mm-yyyy hh24:mi:ss')
    AND mekanisasi.empqtyact.tdate <= TO_TIMESTAMP('31-12-2023 00:00:00', 'dd-mm-yyyy hh24:mi:ss')"""

df_empqtyact = pd.read_sql(q_empqtyact, con=conn)


# In[6]:


empqtyact = df_empqtyact[~df_empqtyact['JOBCODE'].isin(filter_jobcode)]


# # M_JOB_RAWAT

# In[7]:


q_m_job_rawat  = """SELECT JOBCODE, DESCRIPTION FROM MEKANISASI.M_JOB_RWT"""

df_m_job_rawat = pd.read_sql(q_m_job_rawat, con=conn)


# In[8]:


m_job_rawat=df_m_job_rawat[~df_m_job_rawat['JOBCODE'].isin(filter_jobcode)]


# In[9]:


m_job_rawat=m_job_rawat.drop_duplicates()
m_job_rawat.head()


# # T_RWT_MATERIAL

# In[10]:


QUERY_t_rwt_material  = """SELECT
    mekanisasi.t_rwt_material.tdate,
    mekanisasi.t_rwt_material.sitecode,
    mekanisasi.t_rwt_material.spvcode,
    mekanisasi.t_rwt_material.jobcode,
    mekanisasi.t_rwt_material.mat1,
    mekanisasi.m_material.mcname,
    mekanisasi.t_rwt_material.qtymat1,
    mekanisasi.t_rwt_material.mat2,
    mekanisasi.t_rwt_material.qtymat2,
    mekanisasi.t_rwt_material.mat3,
    mekanisasi.m_material.mgname
FROM
         mekanisasi.t_rwt_material
    INNER JOIN mekanisasi.m_material ON mekanisasi.m_material.materialcode = mekanisasi.t_rwt_material.mat1
WHERE
        mekanisasi.t_rwt_material.tdate >= TO_TIMESTAMP('15-12-2022 00:00:00', 'dd-mm-yyyy hh24:mi:ss')
    AND mekanisasi.t_rwt_material.tdate <= TO_TIMESTAMP('31-12-2023 00:00:00', 'dd-mm-yyyy hh24:mi:ss')"""

df_t_rwt_material = pd.read_sql(QUERY_t_rwt_material, con=conn)
df_t_rwt_material.head()


# In[11]:


t_rwt_material=df_t_rwt_material[~df_t_rwt_material['JOBCODE'].isin(filter_jobcode)]


# # EDA

# In[12]:


# copy material re run from here
t_rwt_material = t_rwt_material.copy()
empqtyact = empqtyact.copy()
m_job_rawat = m_job_rawat.copy()


# In[13]:


# add job level
df_rawat_description=t_rwt_material.merge(m_job_rawat,on='JOBCODE', how= 'left')


# In[14]:


df_emp_desc_job = empqtyact.merge(m_job_rawat,on='JOBCODE', how='left')
df_emp_desc_job.head()


# ### CLEANSING AND CLUSTERING

# In[15]:


df_cluster = dataclustering[["KODE JOB LEVEL 3", "JOB_DESC","Clustering"]]


# In[23]:


df_rawat_description_1= df_rawat_description.merge(df_cluster, how = 'left', left_on='JOBCODE', right_on="KODE JOB LEVEL 3").fillna(0).drop(columns=['MAT2','QTYMAT2','MAT3'],axis=1)


# In[24]:


df_rawat_description_1.info()


# In[25]:


drop_repl = df_rawat_description_1[df_rawat_description_1['DESCRIPTION'].str.contains('REPL')==True]
drop_tbm = df_rawat_description_1[df_rawat_description_1['DESCRIPTION'].str.contains('TBM')==True]
drop_bbt =df_rawat_description_1[df_rawat_description_1['DESCRIPTION'].str.contains('BBT')==True]
drop_tkks =df_rawat_description_1[df_rawat_description_1['DESCRIPTION'].str.contains('TKKS')==True]
drop_abu =df_rawat_description_1[df_rawat_description_1['DESCRIPTION'].str.contains('ABU')==True]
drop_grabber =df_rawat_description_1[df_rawat_description_1['DESCRIPTION'].str.contains('Graber')==True]

df_drop = pd.concat([drop_repl,drop_tbm,drop_bbt,drop_tkks,drop_abu,drop_grabber])


# In[27]:


df_rawat_description_1.drop(df_drop.index, inplace=True)


# In[28]:


dff=df_rawat_description_1[df_rawat_description_1['MGNAME']!='CHEMICAL TANAMAN']


# In[29]:


df_pemupukan=dff[(dff.MGNAME=='CHEMIST')|(dff.MGNAME=='PUPUK')|(dff['MCNAME']=='Starane')|(dff['MCNAME']=='Garlon')|(dff['MCNAME']=='Parakuat')|(dff['MCNAME']=='24D-Amine')|(dff['MCNAME']=='Gliphosate')]


# In[30]:


df_pemupukan =df_pemupukan[['SITECODE','TDATE', 'SPVCODE', 'JOBCODE', 'MAT1', 'MCNAME', 'QTYMAT1',
       'MGNAME', 'DESCRIPTION','Clustering']].rename(columns={'MCNAME':'NAMA','MGNAME':'CLUSTERING2','Clustering':'CLUSTERING'})


# In[31]:


df_pemupukan['KET'] = 0
keywords = ['NPK', 'BORATE', 'KIESERITE', 'KAPTAN', 'UREA', 'DOLOMITE', 'MOP', 'Zn EDTA', 'Cu EDTA','RP','TSP']
for i in range(len(keywords)):
    df_pemupukan['KET'] = np.where(df_pemupukan['NAMA'].str.contains(keywords[i], case=False), keywords[i], df_pemupukan['KET'])


# In[32]:


keywords = ['Starane', 'Garlon', 'Parakuat', '24D-Amine', 'Gliphosate', 'PRORODENT', 'Emulan LVA']
for keyword in keywords:
    df_pemupukan.loc[(df_pemupukan['CLUSTERING2'] == 'CHEMIST') & (df_pemupukan['NAMA'] == keyword), 'KET'] = "CHEMIST"


# In[33]:


df_pemupukan_1 =df_pemupukan[['SITECODE','TDATE','SPVCODE','JOBCODE','MAT1','NAMA','QTYMAT1','KET','CLUSTERING2','DESCRIPTION']].rename(columns={'KET':'CLUSTERINGG'})


# ### ADD LOCATION

# In[34]:


df_empqtyact_description=df_emp_desc_job[['SITECODE', 'TDATE', 'LOCATION', 'MATURETYPE', 'JOBCODE', 'MANDORCODE',
       'EMPMANDOR', 'EMPNO', 'EMPCODE', 'QTYACT', 'MATERIAL1', 'QTYMATERIAL1',
       'MATERIAL2', 'QTYMATERIAL2', 'MATERIAL3', 'QTYMATERIAL3',
       'JOBGROUPCODE', 'DESCRIPTION']]


# In[35]:


location =df_empqtyact_description.iloc[:,0:5].drop(columns=['MATURETYPE'],axis=1)
location_drop = location.drop_duplicates()


# In[36]:


location_drop_new = location_drop[['SITECODE','TDATE','JOBCODE','LOCATION']]


# ### MERGE

# In[37]:


df_pemupukan_merge=df_pemupukan_1.merge(location_drop_new, on=['SITECODE','TDATE','JOBCODE'],how="left").drop_duplicates()


# ### EXPORT

# In[38]:


df_pemupukan_merge['PTAFD_PIMS'] = df_pemupukan_merge['SITECODE'] + df_pemupukan_merge['LOCATION'].str[:2]
df_pemupukan_merge['AFD'] = df_pemupukan_merge['LOCATION'].str[:2]
df_pemupukan_merge['BLOCKCODE'] = df_pemupukan_merge['LOCATION'].str[2:]


# In[49]:


df_final=df_pemupukan_merge[['SITECODE','PTAFD_PIMS','TDATE','BLOCKCODE','AFD','SPVCODE','JOBCODE','CLUSTERINGG','CLUSTERING2','DESCRIPTION','MAT1','QTYMAT1']]


# In[50]:


df_final_groupby  = df_final.groupby(['SITECODE','SPVCODE','JOBCODE','TDATE','CLUSTERINGG','CLUSTERING2','DESCRIPTION','MAT1'])['QTYMAT1'].count().reset_index()


# In[52]:


final=df_final.merge(df_final_groupby, left_on=['SITECODE','SPVCODE','JOBCODE','TDATE','CLUSTERINGG','CLUSTERING2','DESCRIPTION','MAT1'],right_on=['SITECODE','SPVCODE','JOBCODE','TDATE','CLUSTERINGG','CLUSTERING2','DESCRIPTION','MAT1'], how ='left').rename(columns={'QTYMAT1_x':'QTYMAT','QTYMAT1_y':'COUNT'})


# In[53]:


final['SUM_QTYMAT1'] = final['QTYMAT'] / final['COUNT']


# In[55]:


final= final[['PTAFD_PIMS','TDATE', 'AFD','BLOCKCODE','SPVCODE', 'JOBCODE',
       'CLUSTERINGG', 'CLUSTERING2', 'DESCRIPTION', 'MAT1',
       'SUM_QTYMAT1']].rename(columns={'SPVCODE':'MANDORCODE','CLUSTERINGG':'Clustering','CLUSTERING2':'Clustering2','SUM_QTYMAT1':'Sum_QTYMAT1'})


# In[56]:


final['Sum_QTYMAT1']=final['Sum_QTYMAT1'].round().astype(int)


# In[58]:


final.to_csv("pemupukan.csv")


# In[ ]:




