---
title: Discrimination
nav_include: 4
notebook: Discrimination_Analysis.ipynb
---


{:.no_toc}
*  
{: toc}







Discrimination is the act of someone being prejudice towards another. Forms of discrimination include nationalist, race, gender, religions, etc. 

As in modern world, non-discrimination has been stated on varies websites. On the bottom of each page of Lending Club, “Equal Housing Lender” logo appears, which means that "the bank makes loans without regard to race, color, religion, national origin, sex, handicap, or familial status." However, this logo doesn’t mean the web is 100% fair. In this context, we want to do more analysis  to test discrimination.

However, due to the limitation of the data set, we can only focus analysis on state level. Our final goal is to find the relationship between loan return rate and loan accepting rate for each state.



```python
import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.plotly as py

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score

%matplotlib inline

import seaborn as sns
sns.set(style='whitegrid')
pd.set_option('display.width', 1500)
pd.set_option('display.max_columns', 100)
```


## Discrimination Analysis 

To investigate whether there exists discrimination in LandingClub, we simplify our question as: whether applicants from a certain state were less likely to receive a loan.    
One way we think about this question is that for a given state if the rejection rate is higher than the proportion of loans of that state among total loans, then that state is likely to be underprivileged.     
The other way is that there should be a postive associate between the acceptance rate and the probability of fully paid. So we want to use data of 50 states to fit a linear regression model, the states under the trend line are likely to be underprivileged while those above the trend line should be considered as being privileged.

### Data Summary



```python
acc_data = pd.read_csv('cdf_acc2016.csv')
acc_data.head()
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>emp_length</th>
      <th>home_ownership</th>
      <th>annual_inc</th>
      <th>verification_status</th>
      <th>issue_d</th>
      <th>loan_status</th>
      <th>purpose</th>
      <th>zip_code</th>
      <th>addr_state</th>
      <th>dti</th>
      <th>delinq_2yrs</th>
      <th>earliest_cr_line</th>
      <th>inq_last_6mths</th>
      <th>open_acc</th>
      <th>pub_rec</th>
      <th>revol_bal</th>
      <th>revol_util</th>
      <th>total_acc</th>
      <th>initial_list_status</th>
      <th>out_prncp</th>
      <th>out_prncp_inv</th>
      <th>total_pymnt</th>
      <th>total_pymnt_inv</th>
      <th>total_rec_prncp</th>
      <th>total_rec_int</th>
      <th>last_pymnt_d</th>
      <th>last_pymnt_amnt</th>
      <th>last_credit_pull_d</th>
      <th>application_type</th>
      <th>acc_now_delinq</th>
      <th>tot_coll_amt</th>
      <th>tot_cur_bal</th>
      <th>total_rev_hi_lim</th>
      <th>acc_open_past_24mths</th>
      <th>avg_cur_bal</th>
      <th>bc_open_to_buy</th>
      <th>bc_util</th>
      <th>chargeoff_within_12_mths</th>
      <th>delinq_amnt</th>
      <th>mo_sin_old_il_acct</th>
      <th>mo_sin_old_rev_tl_op</th>
      <th>mo_sin_rcnt_rev_tl_op</th>
      <th>mo_sin_rcnt_tl</th>
      <th>mort_acc</th>
      <th>mths_since_recent_bc</th>
      <th>num_accts_ever_120_pd</th>
      <th>num_actv_bc_tl</th>
      <th>num_actv_rev_tl</th>
      <th>num_bc_sats</th>
      <th>num_bc_tl</th>
      <th>num_il_tl</th>
      <th>num_op_rev_tl</th>
      <th>num_rev_accts</th>
      <th>num_rev_tl_bal_gt_0</th>
      <th>num_sats</th>
      <th>num_tl_120dpd_2m</th>
      <th>num_tl_30dpd</th>
      <th>num_tl_90g_dpd_24m</th>
      <th>num_tl_op_past_12m</th>
      <th>pct_tl_nvr_dlq</th>
      <th>percent_bc_gt_75</th>
      <th>pub_rec_bankruptcies</th>
      <th>tax_liens</th>
      <th>tot_hi_cred_lim</th>
      <th>total_bal_ex_mort</th>
      <th>total_bc_limit</th>
      <th>total_il_high_credit_limit</th>
      <th>hardship_flag</th>
      <th>disbursement_method</th>
      <th>debt_settlement_flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>35000</td>
      <td>35000</td>
      <td>60</td>
      <td>21.18</td>
      <td>950.42</td>
      <td>E</td>
      <td>10+</td>
      <td>MORTGAGE</td>
      <td>195000.0</td>
      <td>1</td>
      <td>Mar-2016</td>
      <td>1</td>
      <td>debt_consolidation</td>
      <td>871xx</td>
      <td>NM</td>
      <td>15.56</td>
      <td>0</td>
      <td>Aug-2004</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>32223</td>
      <td>0.934</td>
      <td>28</td>
      <td>w</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>47748.356466</td>
      <td>47748.36</td>
      <td>35000.0</td>
      <td>12748.36</td>
      <td>Mar-2018</td>
      <td>26612.62</td>
      <td>Jul-2018</td>
      <td>Individual</td>
      <td>0</td>
      <td>0</td>
      <td>644712.0</td>
      <td>34500</td>
      <td>12</td>
      <td>71635</td>
      <td>878</td>
      <td>91.2</td>
      <td>0</td>
      <td>0</td>
      <td>122</td>
      <td>95</td>
      <td>7</td>
      <td>7</td>
      <td>4</td>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>17</td>
      <td>4</td>
      <td>7</td>
      <td>4</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>88.5</td>
      <td>100.0</td>
      <td>0</td>
      <td>0</td>
      <td>671522</td>
      <td>166084</td>
      <td>10000</td>
      <td>138709</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16000</td>
      <td>16000</td>
      <td>36</td>
      <td>5.32</td>
      <td>481.84</td>
      <td>A</td>
      <td>8</td>
      <td>RENT</td>
      <td>105000.0</td>
      <td>0</td>
      <td>Mar-2016</td>
      <td>1</td>
      <td>debt_consolidation</td>
      <td>206xx</td>
      <td>MD</td>
      <td>15.02</td>
      <td>1</td>
      <td>Nov-2000</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>6219</td>
      <td>0.279</td>
      <td>21</td>
      <td>w</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>16098.340000</td>
      <td>16098.34</td>
      <td>16000.0</td>
      <td>98.34</td>
      <td>May-2016</td>
      <td>16107.80</td>
      <td>Nov-2016</td>
      <td>Individual</td>
      <td>0</td>
      <td>0</td>
      <td>23525.0</td>
      <td>22300</td>
      <td>3</td>
      <td>3361</td>
      <td>13632</td>
      <td>29.7</td>
      <td>0</td>
      <td>0</td>
      <td>124</td>
      <td>184</td>
      <td>22</td>
      <td>8</td>
      <td>0</td>
      <td>52</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>5</td>
      <td>7</td>
      <td>6</td>
      <td>14</td>
      <td>3</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>95.2</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>47543</td>
      <td>23525</td>
      <td>19400</td>
      <td>25243</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9000</td>
      <td>9000</td>
      <td>36</td>
      <td>5.32</td>
      <td>271.04</td>
      <td>A</td>
      <td>10+</td>
      <td>MORTGAGE</td>
      <td>90000.0</td>
      <td>0</td>
      <td>Mar-2016</td>
      <td>1</td>
      <td>home_improvement</td>
      <td>581xx</td>
      <td>ND</td>
      <td>17.97</td>
      <td>0</td>
      <td>Dec-1989</td>
      <td>1</td>
      <td>20</td>
      <td>0</td>
      <td>19386</td>
      <td>0.239</td>
      <td>42</td>
      <td>w</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9361.741129</td>
      <td>9361.74</td>
      <td>9000.0</td>
      <td>361.74</td>
      <td>Feb-2017</td>
      <td>6927.70</td>
      <td>Feb-2017</td>
      <td>Individual</td>
      <td>0</td>
      <td>0</td>
      <td>151359.0</td>
      <td>81100</td>
      <td>6</td>
      <td>7966</td>
      <td>48684</td>
      <td>28.3</td>
      <td>0</td>
      <td>0</td>
      <td>180</td>
      <td>315</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>5</td>
      <td>10</td>
      <td>16</td>
      <td>12</td>
      <td>16</td>
      <td>27</td>
      <td>5</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>100.0</td>
      <td>10.0</td>
      <td>0</td>
      <td>0</td>
      <td>288144</td>
      <td>61072</td>
      <td>67900</td>
      <td>54244</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13550</td>
      <td>13550</td>
      <td>36</td>
      <td>10.75</td>
      <td>442.01</td>
      <td>B</td>
      <td>2</td>
      <td>MORTGAGE</td>
      <td>79000.0</td>
      <td>1</td>
      <td>Mar-2016</td>
      <td>1</td>
      <td>credit_card</td>
      <td>743xx</td>
      <td>OK</td>
      <td>19.22</td>
      <td>0</td>
      <td>Apr-2000</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>7787</td>
      <td>0.666</td>
      <td>23</td>
      <td>w</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>15537.174857</td>
      <td>15537.17</td>
      <td>13550.0</td>
      <td>1987.17</td>
      <td>Feb-2018</td>
      <td>6261.07</td>
      <td>Sep-2018</td>
      <td>Individual</td>
      <td>0</td>
      <td>2552</td>
      <td>135472.0</td>
      <td>11700</td>
      <td>3</td>
      <td>15052</td>
      <td>1966</td>
      <td>77.9</td>
      <td>0</td>
      <td>0</td>
      <td>138</td>
      <td>114</td>
      <td>24</td>
      <td>13</td>
      <td>2</td>
      <td>24</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>4</td>
      <td>5</td>
      <td>15</td>
      <td>5</td>
      <td>6</td>
      <td>5</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>95.7</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>163583</td>
      <td>38420</td>
      <td>8900</td>
      <td>47151</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10000</td>
      <td>10000</td>
      <td>36</td>
      <td>6.49</td>
      <td>306.45</td>
      <td>A</td>
      <td>4</td>
      <td>MORTGAGE</td>
      <td>70000.0</td>
      <td>1</td>
      <td>Mar-2016</td>
      <td>1</td>
      <td>debt_consolidation</td>
      <td>478xx</td>
      <td>IN</td>
      <td>13.66</td>
      <td>0</td>
      <td>May-1999</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>24472</td>
      <td>0.665</td>
      <td>19</td>
      <td>w</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10924.386145</td>
      <td>10924.39</td>
      <td>10000.0</td>
      <td>924.39</td>
      <td>May-2018</td>
      <td>3576.80</td>
      <td>Sep-2018</td>
      <td>Individual</td>
      <td>0</td>
      <td>0</td>
      <td>110357.0</td>
      <td>36800</td>
      <td>1</td>
      <td>22071</td>
      <td>1921</td>
      <td>91.8</td>
      <td>0</td>
      <td>0</td>
      <td>159</td>
      <td>202</td>
      <td>40</td>
      <td>20</td>
      <td>2</td>
      <td>71</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>4</td>
      <td>10</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>0</td>
      <td>0</td>
      <td>137157</td>
      <td>24472</td>
      <td>23300</td>
      <td>0</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
</div>





```python
# Calculating proportion of fully paid
def success_rate(df):
    succ_rate = list()
    succ_rate.append(len(df[df['loan_status']==1])/len(df))
    return succ_rate

succ_rate_df = acc_data.groupby('addr_state').apply(success_rate)
succ_rate = pd.DataFrame({'State': succ_rate_df.index, 'Success Rate': succ_rate_df.values})
for i in range(len(succ_rate)):
    succ_rate['Success Rate'][i] = succ_rate['Success Rate'][i][0]
```




```python
df = pd.read_csv('if_loan_is_rejected.csv')
df = df.drop(df[df['State'] == 'IA'].index)
rej = df[df['ifrej']== 1]
acp = df[df['ifrej']== 0]
```




```python
acc_count = acp['State'].value_counts()
acc_count = pd.DataFrame({'State': acc_count.index, 'Accept Count': acc_count.values})
rej_count = rej['State'].value_counts()
rej_count = pd.DataFrame({'State': rej_count.index, 'Reject Count': rej_count.values})
```




```python
model = pd.read_csv('cdf_discrimination.csv')
model_roi = pd.read_csv('roi_discrimination.csv')
model_iroi = pd.read_csv('iroi_discrimination.csv')
model.head()
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>purpose</th>
      <th>emp_length</th>
      <th>num_tl_120dpd_2m</th>
      <th>acc_now_delinq</th>
      <th>annual_inc</th>
      <th>revol_bal</th>
      <th>avg_cur_bal</th>
      <th>mo_sin_old_il_acct</th>
      <th>mort_acc</th>
      <th>pct_tl_nvr_dlq</th>
      <th>addr_state</th>
      <th>funded_amnt</th>
      <th>total_pymnt</th>
      <th>loan_status</th>
      <th>Random_Forest</th>
      <th>Neural_Network</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>debt_consolidation</td>
      <td>10+</td>
      <td>0</td>
      <td>0</td>
      <td>195000.0</td>
      <td>32223</td>
      <td>71635</td>
      <td>122</td>
      <td>4</td>
      <td>88.5</td>
      <td>NM</td>
      <td>35000</td>
      <td>47748.356466</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>debt_consolidation</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>105000.0</td>
      <td>6219</td>
      <td>3361</td>
      <td>124</td>
      <td>0</td>
      <td>95.2</td>
      <td>MD</td>
      <td>16000</td>
      <td>16098.340000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>home_improvement</td>
      <td>10+</td>
      <td>0</td>
      <td>0</td>
      <td>90000.0</td>
      <td>19386</td>
      <td>7966</td>
      <td>180</td>
      <td>3</td>
      <td>100.0</td>
      <td>ND</td>
      <td>9000</td>
      <td>9361.741129</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>credit_card</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>79000.0</td>
      <td>7787</td>
      <td>15052</td>
      <td>138</td>
      <td>2</td>
      <td>95.7</td>
      <td>OK</td>
      <td>13550</td>
      <td>15537.174857</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>debt_consolidation</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>70000.0</td>
      <td>24472</td>
      <td>22071</td>
      <td>159</td>
      <td>2</td>
      <td>100.0</td>
      <td>IN</td>
      <td>10000</td>
      <td>10924.386145</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>





```python
rf_rej = model[model['Random_Forest']== 0][['addr_state', 'Random_Forest']]
rf_acp = model[model['Random_Forest']== 1][['addr_state', 'Random_Forest']]
nn_rej = model[model['Neural_Network']== 0][['addr_state', 'Neural_Network']]
nn_acp = model[model['Neural_Network']== 1][['addr_state', 'Neural_Network']]
roi_rf_rej = model_roi[model_roi['Random_Forest']== 0][['addr_state', 'Random_Forest']]
roi_rf_acp = model_roi[model_roi['Random_Forest']== 1][['addr_state', 'Random_Forest']]
roi_nn_rej = model_roi[model_roi['Neural_Network']== 0][['addr_state', 'Neural_Network']]
roi_nn_acp = model_roi[model_roi['Neural_Network']== 1][['addr_state', 'Neural_Network']]
iroi_rf_rej = model_iroi[model_iroi['Random_Forest']== 0][['addr_state', 'Random_Forest']]
iroi_rf_acp = model_iroi[model_iroi['Random_Forest']== 1][['addr_state', 'Random_Forest']]
iroi_nn_rej = model_iroi[model_iroi['Neural_Network']== 0][['addr_state', 'Neural_Network']]
iroi_nn_acp = model_iroi[model_iroi['Neural_Network']== 1][['addr_state', 'Neural_Network']]

acc_rf_counts = rf_acp['addr_state'].value_counts()
acc_rf_counts_df = pd.DataFrame({'State': acc_rf_counts.index, 'RF Accept Counts': acc_rf_counts.values})
acc_nn_counts = nn_acp['addr_state'].value_counts()
acc_nn_counts_df = pd.DataFrame({'State': acc_nn_counts.index, 'NN Accept Counts': acc_nn_counts.values})
roi_acc_rf_counts = roi_rf_acp['addr_state'].value_counts()
roi_acc_rf_counts_df = pd.DataFrame({'State': roi_acc_rf_counts.index, 'ROI RF Accept Counts': roi_acc_rf_counts.values})
roi_acc_nn_counts = roi_nn_acp['addr_state'].value_counts()
roi_acc_nn_counts_df = pd.DataFrame({'State': roi_acc_nn_counts.index, 'ROI NN Accept Counts': roi_acc_nn_counts.values})
iroi_acc_rf_counts = iroi_rf_acp['addr_state'].value_counts()
iroi_acc_rf_counts_df = pd.DataFrame({'State': iroi_acc_rf_counts.index, 'IROI RF Accept Counts': iroi_acc_rf_counts.values})
iroi_acc_nn_counts = iroi_nn_acp['addr_state'].value_counts()
iroi_acc_nn_counts_df = pd.DataFrame({'State': iroi_acc_nn_counts.index, 'IROI NN Accept Counts': iroi_acc_nn_counts.values})
```




```python
# Summarise successfully-paid rate, accept rate, accept count and reject count
state_rate = pd.merge(succ_rate, acc_count, how='left', on=['State'])
state_rate = pd.merge(state_rate, rej_count, how='left', on=['State'])
state_rate = pd.merge(state_rate, acc_rf_counts_df, how='left', on=['State'])
state_rate = pd.merge(state_rate, acc_nn_counts_df, how='left', on=['State'])
state_rate = pd.merge(state_rate, roi_acc_rf_counts_df, how='left', on=['State'])
state_rate = pd.merge(state_rate, roi_acc_nn_counts_df, how='left', on=['State'])
state_rate = pd.merge(state_rate, iroi_acc_rf_counts_df, how='left', on=['State'])
state_rate = pd.merge(state_rate, iroi_acc_nn_counts_df, how='left', on=['State'])
state_rate['Total Loan'] = (state_rate['Accept Count'] + state_rate['Reject Count'])
state_rate['Accept Rate'] = state_rate['Accept Count']/state_rate['Total Loan'] 
state_rate['RF Accept Rate'] = state_rate['RF Accept Counts']/state_rate['Total Loan']
state_rate['NN Accept Rate'] = state_rate['NN Accept Counts']/state_rate['Total Loan']
state_rate['ROI RF Accept Rate'] = state_rate['ROI RF Accept Counts']/state_rate['Total Loan']
state_rate['ROI NN Accept Rate'] = state_rate['ROI NN Accept Counts']/state_rate['Total Loan']
state_rate['IROI RF Accept Rate'] = state_rate['IROI RF Accept Counts']/state_rate['Total Loan']
state_rate['IROI NN Accept Rate'] = state_rate['IROI NN Accept Counts']/state_rate['Total Loan']
state_rate.head()
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Success Rate</th>
      <th>Accept Count</th>
      <th>Reject Count</th>
      <th>RF Accept Counts</th>
      <th>NN Accept Counts</th>
      <th>ROI RF Accept Counts</th>
      <th>ROI NN Accept Counts</th>
      <th>IROI RF Accept Counts</th>
      <th>IROI NN Accept Counts</th>
      <th>Total Loan</th>
      <th>Accept Rate</th>
      <th>RF Accept Rate</th>
      <th>NN Accept Rate</th>
      <th>ROI RF Accept Rate</th>
      <th>ROI NN Accept Rate</th>
      <th>IROI RF Accept Rate</th>
      <th>IROI NN Accept Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AK</td>
      <td>0.714597</td>
      <td>1006</td>
      <td>10984</td>
      <td>131</td>
      <td>115</td>
      <td>20</td>
      <td>32</td>
      <td>131</td>
      <td>109</td>
      <td>11990</td>
      <td>0.083903</td>
      <td>0.010926</td>
      <td>0.009591</td>
      <td>0.001668</td>
      <td>0.002669</td>
      <td>0.010926</td>
      <td>0.009091</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AL</td>
      <td>0.683806</td>
      <td>5329</td>
      <td>85421</td>
      <td>681</td>
      <td>609</td>
      <td>62</td>
      <td>102</td>
      <td>677</td>
      <td>556</td>
      <td>90750</td>
      <td>0.058722</td>
      <td>0.007504</td>
      <td>0.006711</td>
      <td>0.000683</td>
      <td>0.001124</td>
      <td>0.007460</td>
      <td>0.006127</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AR</td>
      <td>0.659658</td>
      <td>3335</td>
      <td>51294</td>
      <td>392</td>
      <td>370</td>
      <td>41</td>
      <td>53</td>
      <td>439</td>
      <td>324</td>
      <td>54629</td>
      <td>0.061048</td>
      <td>0.007176</td>
      <td>0.006773</td>
      <td>0.000751</td>
      <td>0.000970</td>
      <td>0.008036</td>
      <td>0.005931</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AZ</td>
      <td>0.733703</td>
      <td>10462</td>
      <td>100749</td>
      <td>1654</td>
      <td>1592</td>
      <td>223</td>
      <td>315</td>
      <td>1773</td>
      <td>1440</td>
      <td>111211</td>
      <td>0.094073</td>
      <td>0.014873</td>
      <td>0.014315</td>
      <td>0.002005</td>
      <td>0.002832</td>
      <td>0.015943</td>
      <td>0.012948</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CA</td>
      <td>0.718838</td>
      <td>57888</td>
      <td>567707</td>
      <td>8368</td>
      <td>7773</td>
      <td>1084</td>
      <td>1635</td>
      <td>8914</td>
      <td>7198</td>
      <td>625595</td>
      <td>0.092533</td>
      <td>0.013376</td>
      <td>0.012425</td>
      <td>0.001733</td>
      <td>0.002614</td>
      <td>0.014249</td>
      <td>0.011506</td>
    </tr>
  </tbody>
</table>
</div>





```python
def add_subplot(factor, subplot,ylim, color):
    plt.subplot(subplot)
    plt.title(factor)
    #state_rate[factor].plot.bar(x='State', y=factor)
    plt.bar(x = state_rate['State'], height = state_rate[factor], color=color)
    plt.ylim(ylim)
    plt.xticks(rotation=90, fontsize=8)
fig, ax = plt.subplots(figsize=(12, 8))
fig.suptitle('Summary of Acceptance and Rejection Status of 50 States', fontsize=16)
add_subplot('Accept Count', 221,[0,60000], "#a072fc")
add_subplot('Reject Count', 222, [0,600000], "#fc9272")
add_subplot('Success Rate', 223, [0.65,0.85], '#a1d99b')
add_subplot('Accept Rate', 224, [0.05,0.12], '#9ecae1')

plt.subplots_adjust(top=0.89, bottom=0, left=0, right=1.0)
```



![png](Discrimination_Analysis_files/Discrimination_Analysis_12_0.png)


The accepted loan count and rejected loan count are similar for each state. However, the accepting rate and success return loan rate seems have non-constant relationship.

### Bias Analysis with Linear Regression



```python
# Fit linear regression model to check relationship between success_rate and accept_rate
X = state_rate['Success Rate'].to_frame()
y = state_rate['Accept Rate']
succ_to_accept = LinearRegression().fit(X.values, y.values)

X_rf = state_rate['Success Rate'].to_frame()
y_rf = state_rate['RF Accept Rate']
succ_to_accept_rf = LinearRegression().fit(X_rf.values, y_rf.values)

X_nn = state_rate['Success Rate'].to_frame()
y_nn = state_rate['NN Accept Rate']
succ_to_accept_nn = LinearRegression().fit(X_rf.values, y_rf.values)

X_roi_rf = state_rate['Success Rate'].to_frame()
y_roi_rf = state_rate['ROI RF Accept Rate']
succ_to_accept_roi_rf = LinearRegression().fit(X_roi_rf.values, y_roi_rf.values)

X_roi_nn = state_rate['Success Rate'].to_frame()
y_roi_nn = state_rate['ROI NN Accept Rate']
succ_to_accept_roi_nn = LinearRegression().fit(X_roi_rf.values, y_roi_rf.values)

X_iroi_rf = state_rate['Success Rate'].to_frame()
y_iroi_rf = state_rate['IROI RF Accept Rate']
succ_to_accept_iroi_rf = LinearRegression().fit(X_iroi_rf.values, y_iroi_rf.values)

X_iroi_nn = state_rate['Success Rate'].to_frame()
y_iroi_nn = state_rate['IROI NN Accept Rate']
succ_to_accept_iroi_nn = LinearRegression().fit(X_iroi_rf.values, y_iroi_rf.values)

# r2 score
r2 = r2_score(y, succ_to_accept.predict(X.values))
rf_r2 = r2_score(y_rf, succ_to_accept_rf.predict(X_rf.values))
nn_r2 = r2_score(y_nn, succ_to_accept_nn.predict(X_nn.values))
roi_rf_r2 = r2_score(y_roi_rf, succ_to_accept_roi_rf.predict(X_roi_rf.values))
roi_nn_r2 = r2_score(y_roi_nn, succ_to_accept_roi_nn.predict(X_roi_nn.values))
iroi_rf_r2 = r2_score(y_iroi_rf, succ_to_accept_iroi_rf.predict(X_iroi_rf.values))
iroi_nn_r2 = r2_score(y_iroi_nn, succ_to_accept_iroi_nn.predict(X_iroi_nn.values))

# Consider the distance between estimated accept rate and true rate as bias
state_rate['BIAS'] = pd.Series(succ_to_accept.predict(X.values)) - y
state_rate.head()
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Success Rate</th>
      <th>Accept Count</th>
      <th>Reject Count</th>
      <th>RF Accept Counts</th>
      <th>NN Accept Counts</th>
      <th>ROI RF Accept Counts</th>
      <th>ROI NN Accept Counts</th>
      <th>IROI RF Accept Counts</th>
      <th>IROI NN Accept Counts</th>
      <th>Total Loan</th>
      <th>Accept Rate</th>
      <th>RF Accept Rate</th>
      <th>NN Accept Rate</th>
      <th>ROI RF Accept Rate</th>
      <th>ROI NN Accept Rate</th>
      <th>IROI RF Accept Rate</th>
      <th>IROI NN Accept Rate</th>
      <th>BIAS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AK</td>
      <td>0.714597</td>
      <td>1006</td>
      <td>10984</td>
      <td>131</td>
      <td>115</td>
      <td>20</td>
      <td>32</td>
      <td>131</td>
      <td>109</td>
      <td>11990</td>
      <td>0.083903</td>
      <td>0.010926</td>
      <td>0.009591</td>
      <td>0.001668</td>
      <td>0.002669</td>
      <td>0.010926</td>
      <td>0.009091</td>
      <td>-0.002699</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AL</td>
      <td>0.683806</td>
      <td>5329</td>
      <td>85421</td>
      <td>681</td>
      <td>609</td>
      <td>62</td>
      <td>102</td>
      <td>677</td>
      <td>556</td>
      <td>90750</td>
      <td>0.058722</td>
      <td>0.007504</td>
      <td>0.006711</td>
      <td>0.000683</td>
      <td>0.001124</td>
      <td>0.007460</td>
      <td>0.006127</td>
      <td>0.018300</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AR</td>
      <td>0.659658</td>
      <td>3335</td>
      <td>51294</td>
      <td>392</td>
      <td>370</td>
      <td>41</td>
      <td>53</td>
      <td>439</td>
      <td>324</td>
      <td>54629</td>
      <td>0.061048</td>
      <td>0.007176</td>
      <td>0.006773</td>
      <td>0.000751</td>
      <td>0.000970</td>
      <td>0.008036</td>
      <td>0.005931</td>
      <td>0.012694</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AZ</td>
      <td>0.733703</td>
      <td>10462</td>
      <td>100749</td>
      <td>1654</td>
      <td>1592</td>
      <td>223</td>
      <td>315</td>
      <td>1773</td>
      <td>1440</td>
      <td>111211</td>
      <td>0.094073</td>
      <td>0.014873</td>
      <td>0.014315</td>
      <td>0.002005</td>
      <td>0.002832</td>
      <td>0.015943</td>
      <td>0.012948</td>
      <td>-0.010273</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CA</td>
      <td>0.718838</td>
      <td>57888</td>
      <td>567707</td>
      <td>8368</td>
      <td>7773</td>
      <td>1084</td>
      <td>1635</td>
      <td>8914</td>
      <td>7198</td>
      <td>625595</td>
      <td>0.092533</td>
      <td>0.013376</td>
      <td>0.012425</td>
      <td>0.001733</td>
      <td>0.002614</td>
      <td>0.014249</td>
      <td>0.011506</td>
      <td>-0.010752</td>
    </tr>
  </tbody>
</table>
</div>






![png](Discrimination_Analysis_files/dis.png)


For the first plot, under non-discrimination assumption, we hope the successful return loan rate and accepting rate have positive linear relationship that successful return loan rate is the only cause of accepting rate. However, in reality, some states are off the line indicating there might be potential discrimination. For example, some states, such as OR and ME, have relative high return loan rate compare to CO and DC, but they also have smaller accepting rate. On the other hand, some states, like NY and NJ, have relative small successful return loan rate but have accepting rate.     
The second and third plots represent the relationship between predicted accept rate by our random forest and neural network models and the actual success rate. We can see that the dots are more centered towards the trend line, indicating that the accept rate depend more on the successfully paid rate. Also, the $R^2$ of these two linear models are higher than the original model, which suggests that using the acceptance rate predicted by our models, the linear regression model can explain more variability of the response data. Therefore, we can conclude that if adopting the investment strategy based on our random forest or neural network models, there would be less potential demographical discrimination.



```python
state_rate = state_rate.sort_values(by = 'BIAS')
fig, ax = plt.subplots(figsize=(15, 6))
plt.bar(x = state_rate['State'], height = state_rate['BIAS'])
plt.xticks(rotation=70)
y_ticks = ax.get_yticks()
ax.set_ylabel('BIAS')
ax.set_xlabel('State')
ax.set_title('Bias (Estimated Accept Rate - True Accept Rate)', fontsize=16)
plt.show()
```



![png](Discrimination_Analysis_files/Discrimination_Analysis_18_0.png)


Similar to previous plot, in reality, some states have high accepting rate than expected.



```python
scl=[[0.0, 'rgb(165,0,38)'], [0.2, 'rgb(244,109,67)'], [0.4, 'rgb(254,224,144)'], 
            [0.6, 'rgb(171,217,233)'], [0.8, 'rgb(69,117,180)'], [1.0, 'rgb(49,54,149)']]
data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = state_rate['State'],
        z = state_rate['BIAS'].astype(float),
        locationmode = 'USA-states',
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Bias")
        ) ]

layout = dict(
        title = 'Difference Between Actual and Estimated Accept Rate Based on Linear Regression',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='lendingclub-cloropleth-map-2' )
```





<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~jingyichen/4.embed" height="525px" width="100%"></iframe>



From the above US map, most northern states have lower rejection rates while southern states have higher rejection rates. This trend may show some discrimination in Lending Club.

### Difference Between Actual and Expect Number of Rejects



```python
# The proportion of number of total loans for each state
p_all = df['State'].value_counts()/len(df)
p_all_df = pd.DataFrame({'State': p_all.index, 'All_Prop': p_all.values})
# The proportion of number of rejected loans for every state
p_rej = rej['State'].value_counts()/len(rej)
p_rej_df = pd.DataFrame({'State': p_rej.index, 'Rej_Prop': p_rej.values})
# The proportion of number of accepted loans for every state
p_acp = acp['State'].value_counts()/len(acp)
p_acp_df = pd.DataFrame({'State': p_acp.index, 'Acc_Prop': p_acp.values})
```




```python
count_indices = np.arange(len(p_rej_df['State']))
width = np.min(np.diff(count_indices))/3.
names = p_rej_df['State']
fig, ax = plt.subplots(figsize=(15, 6))
ax.bar(count_indices-width/2.,p_rej_df['Rej_Prop'],width,color='Steelblue',
       tick_label = p_rej_df['State'],label='Rejection Proportion')
ax.bar(count_indices+width/2.,p_acp_df['Acc_Prop'],width,color='tomato',label='Acceptance Proportion')
#ax.axes.set_xticklabels(names)
ax.set_xticklabels(names, rotation=70)
ax.set_xlabel('State')
y_ticks = ax.get_yticks()
ax.set_yticklabels(['{:3.2f}%'.format(k*100) for k in y_ticks])
ax.legend()
ax.set_title('Rejection Proportion vs. Acceptance Proportion', fontsize=16)
plt.show()
```



![png](Discrimination_Analysis_files/Discrimination_Analysis_24_0.png)


Some states (CA, FL, AZ, etc) have higher acceptance proportion than rejection proportion.



```python
state_prop = pd.merge(p_all_df, p_rej_df, how='left', on=['State'])
state_prop['Diff_Prop'] = state_prop['Rej_Prop'] - state_prop['All_Prop']
state_prop['Diff_num'] = state_prop['Diff_Prop']*len(rej)
```




```python
state_prop = state_prop.sort_values(by = 'Diff_num')
fig, ax = plt.subplots(figsize=(15, 6))
plt.bar(x = state_prop['State'], height = state_prop['Diff_num'])
plt.xticks(rotation=70)
y_ticks = ax.get_yticks()
ax.set_ylabel('Counts')
ax.set_xlabel('State')
ax.set_title('Difference Between Actual and Expect Number of Rejects', fontsize=16)
plt.show()
```



![png](Discrimination_Analysis_files/Discrimination_Analysis_27_0.png)


In reality, some states have lower rejected-loan counts than expected (states on the left) while many other states have higher rejected-loan counts than expected (states on the right).



```python
import plotly.plotly as py

#ploty.tools.set_credentials_file(username='jingyichen', api_key='ithtgZkbeInxasauGeMe')

scl=[[0.0, 'rgb(165,0,38)'], [0.2, 'rgb(244,109,67)'], [0.4, 'rgb(254,224,144)'], 
            [0.6, 'rgb(171,217,233)'], [0.8, 'rgb(69,117,180)'], [1.0, 'rgb(49,54,149)']]
data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = state_prop['State'],
        z = state_prop['Diff_num'].astype(float),
        locationmode = 'USA-states',
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Bias(counts)")
        ) ]

layout = dict(
        title = 'Difference Between Actual and Expected Number of Rejects',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='lendingclub-cloropleth-map' )
```





<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~jingyichen/2.embed" height="525px" width="100%"></iframe>



This graph shows that applications from areas with darker blue are more likely to be rejected, while applications from areas with darker red are less likely to be accepted, comparing with our estimated rejection number. The dark blue areas are mostly southern states (Florida, Texas, Alabama, Tennessee, etc), and the red areas include California, New York, Massachusetts, and etc. This indicates a general pattern that applicants from states which have worse economic status (e.g. lower median household incomes) are more likely to be rejected than expected, and applicants from states which have better economic status face lower probability of rejection than expected. 
