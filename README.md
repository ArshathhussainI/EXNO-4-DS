# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```python
from scipy import stats
import pandas as pd
import numpy as np
df = pd.read_csv("bmi.csv")
df
```





<div id="df-5bc8f9ba-3150-4808-95dd-27cac6ec6575" class="colab-df-container">
<div>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
<th></th>
<th>Gender</th>
<th>Height</th>
<th>Weight</th>
<th>Index</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>Male</td>
<td>174</td>
<td>96</td>
<td>4</td>
</tr>
<tr>
<th>1</th>
<td>Male</td>
<td>189</td>
<td>87</td>
<td>2</td>
</tr>
<tr>
<th>2</th>
<td>Female</td>
<td>185</td>
<td>110</td>
<td>4</td>
</tr>
<tr>
<th>3</th>
<td>Female</td>
<td>195</td>
<td>104</td>
<td>3</td>
</tr>
<tr>
<th>4</th>
<td>Male</td>
<td>149</td>
<td>61</td>
<td>3</td>
</tr>
<tr>
<th>...</th>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
</tr>
<tr>
<th>495</th>
<td>Female</td>
<td>150</td>
<td>153</td>
<td>5</td>
</tr>
<tr>
<th>496</th>
<td>Female</td>
<td>184</td>
<td>121</td>
<td>4</td>
</tr>
<tr>
<th>497</th>
<td>Female</td>
<td>141</td>
<td>136</td>
<td>5</td>
</tr>
<tr>
<th>498</th>
<td>Male</td>
<td>150</td>
<td>95</td>
<td>5</td>
</tr>
<tr>
<th>499</th>
<td>Male</td>
<td>173</td>
<td>131</td>
<td>5</td>
</tr>
</tbody>
</table>
<p>500 rows × 4 columns</p>
</div>
<div class="colab-df-buttons">

<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-5bc8f9ba-3150-4808-95dd-27cac6ec6575')"
title="Convert this dataframe to an interactive table."
style="display:none;">

</button>


</div>


<div id="id_59ebd83e-c91a-47c5-b957-0fcdc0c646bd">
<button class="colab-df-generate" onclick="generateWithVariable('df')"
title="Generate code using this dataframe."
style="display:none;">

</button>
</div>

</div>
</div>





```python
df.isnull()
```





<div id="df-6b8aba40-dd2c-4820-9b93-f744200eaab9" class="colab-df-container">
<div>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
<th></th>
<th>Gender</th>
<th>Height</th>
<th>Weight</th>
<th>Index</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
</tr>
<tr>
<th>1</th>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
</tr>
<tr>
<th>2</th>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
</tr>
<tr>
<th>3</th>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
</tr>
<tr>
<th>4</th>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
</tr>
<tr>
<th>...</th>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
</tr>
<tr>
<th>495</th>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
</tr>
<tr>
<th>496</th>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
</tr>
<tr>
<th>497</th>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
</tr>
<tr>
<th>498</th>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
</tr>
<tr>
<th>499</th>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
</tr>
</tbody>
</table>
<p>500 rows × 4 columns</p>
</div>
<div class="colab-df-buttons">

<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-6b8aba40-dd2c-4820-9b93-f744200eaab9')"
title="Convert this dataframe to an interactive table."
style="display:none;">

</button>


</div>


</div>
</div>





```python
df.isnull().sum()
```




<div>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
<th></th>
<th>0</th>
</tr>
</thead>
<tbody>
<tr>
<th>Gender</th>
<td>0</td>
</tr>
<tr>
<th>Height</th>
<td>0</td>
</tr>
<tr>
<th>Weight</th>
<td>0</td>
</tr>
<tr>
<th>Index</th>
<td>0</td>
</tr>
</tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>




```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df[["h1-standardscaler","w1-standardscaler"]] = sc.fit_transform(df[["Height","Weight"]])
df.head(10)
```





<div id="df-4e378ea4-9862-41c6-923a-1f5e112b9947" class="colab-df-container">
<div>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
<th></th>
<th>Gender</th>
<th>Height</th>
<th>Weight</th>
<th>Index</th>
<th>h1-standardscaler</th>
<th>w1-standardscaler</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>Male</td>
<td>174</td>
<td>96</td>
<td>4</td>
<td>0.247939</td>
<td>-0.309117</td>
</tr>
<tr>
<th>1</th>
<td>Male</td>
<td>189</td>
<td>87</td>
<td>2</td>
<td>1.164872</td>
<td>-0.587322</td>
</tr>
<tr>
<th>2</th>
<td>Female</td>
<td>185</td>
<td>110</td>
<td>4</td>
<td>0.920357</td>
<td>0.123647</td>
</tr>
<tr>
<th>3</th>
<td>Female</td>
<td>195</td>
<td>104</td>
<td>3</td>
<td>1.531645</td>
<td>-0.061823</td>
</tr>
<tr>
<th>4</th>
<td>Male</td>
<td>149</td>
<td>61</td>
<td>3</td>
<td>-1.280283</td>
<td>-1.391027</td>
</tr>
<tr>
<th>5</th>
<td>Male</td>
<td>189</td>
<td>104</td>
<td>3</td>
<td>1.164872</td>
<td>-0.061823</td>
</tr>
<tr>
<th>6</th>
<td>Male</td>
<td>147</td>
<td>92</td>
<td>5</td>
<td>-1.402541</td>
<td>-0.432764</td>
</tr>
<tr>
<th>7</th>
<td>Male</td>
<td>154</td>
<td>111</td>
<td>5</td>
<td>-0.974639</td>
<td>0.154559</td>
</tr>
<tr>
<th>8</th>
<td>Male</td>
<td>174</td>
<td>90</td>
<td>3</td>
<td>0.247939</td>
<td>-0.494587</td>
</tr>
<tr>
<th>9</th>
<td>Female</td>
<td>169</td>
<td>103</td>
<td>4</td>
<td>-0.057706</td>
<td>-0.092735</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">

<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-4e378ea4-9862-41c6-923a-1f5e112b9947')"
title="Convert this dataframe to an interactive table."
style="display:none;">

</button>


</div>


</div>
</div>





```python
from sklearn.preprocessing import MinMaxScaler
mMs = MinMaxScaler()
df[["h3-MinMaxScaler","w3-MinMaxScaler"]] = mMs.fit_transform(df[["Height","Weight"]])
df.head(10)
```





<div id="df-107dd755-4cc7-476d-a0cd-72182c12b161" class="colab-df-container">
<div>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
<th></th>
<th>Gender</th>
<th>Height</th>
<th>Weight</th>
<th>Index</th>
<th>h1-standardscaler</th>
<th>w1-standardscaler</th>
<th>h3-MinMaxScaler</th>
<th>w3-MinMaxScaler</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>Male</td>
<td>174</td>
<td>96</td>
<td>4</td>
<td>0.247939</td>
<td>-0.309117</td>
<td>0.576271</td>
<td>0.418182</td>
</tr>
<tr>
<th>1</th>
<td>Male</td>
<td>189</td>
<td>87</td>
<td>2</td>
<td>1.164872</td>
<td>-0.587322</td>
<td>0.830508</td>
<td>0.336364</td>
</tr>
<tr>
<th>2</th>
<td>Female</td>
<td>185</td>
<td>110</td>
<td>4</td>
<td>0.920357</td>
<td>0.123647</td>
<td>0.762712</td>
<td>0.545455</td>
</tr>
<tr>
<th>3</th>
<td>Female</td>
<td>195</td>
<td>104</td>
<td>3</td>
<td>1.531645</td>
<td>-0.061823</td>
<td>0.932203</td>
<td>0.490909</td>
</tr>
<tr>
<th>4</th>
<td>Male</td>
<td>149</td>
<td>61</td>
<td>3</td>
<td>-1.280283</td>
<td>-1.391027</td>
<td>0.152542</td>
<td>0.100000</td>
</tr>
<tr>
<th>5</th>
<td>Male</td>
<td>189</td>
<td>104</td>
<td>3</td>
<td>1.164872</td>
<td>-0.061823</td>
<td>0.830508</td>
<td>0.490909</td>
</tr>
<tr>
<th>6</th>
<td>Male</td>
<td>147</td>
<td>92</td>
<td>5</td>
<td>-1.402541</td>
<td>-0.432764</td>
<td>0.118644</td>
<td>0.381818</td>
</tr>
<tr>
<th>7</th>
<td>Male</td>
<td>154</td>
<td>111</td>
<td>5</td>
<td>-0.974639</td>
<td>0.154559</td>
<td>0.237288</td>
<td>0.554545</td>
</tr>
<tr>
<th>8</th>
<td>Male</td>
<td>174</td>
<td>90</td>
<td>3</td>
<td>0.247939</td>
<td>-0.494587</td>
<td>0.576271</td>
<td>0.363636</td>
</tr>
<tr>
<th>9</th>
<td>Female</td>
<td>169</td>
<td>103</td>
<td>4</td>
<td>-0.057706</td>
<td>-0.092735</td>
<td>0.491525</td>
<td>0.481818</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">

<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-107dd755-4cc7-476d-a0cd-72182c12b161')"
title="Convert this dataframe to an interactive table."
style="display:none;">

</button>


</div>


</div>
</div>





```python
from sklearn.preprocessing import Normalizer
n = Normalizer()
df[["h2-Normalizer","w2-Normalizer"]] = n.fit_transform(df[["Height","Weight"]])
df.head(10)
```





<div id="df-9fa4f38c-90e5-4856-9f45-01f4b9840f2e" class="colab-df-container">
<div>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
<th></th>
<th>Gender</th>
<th>Height</th>
<th>Weight</th>
<th>Index</th>
<th>h1-standardscaler</th>
<th>w1-standardscaler</th>
<th>h3-MinMaxScaler</th>
<th>w3-MinMaxScaler</th>
<th>h2-Normalizer</th>
<th>w2-Normalizer</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>Male</td>
<td>174</td>
<td>96</td>
<td>4</td>
<td>0.247939</td>
<td>-0.309117</td>
<td>0.576271</td>
<td>0.418182</td>
<td>0.875578</td>
<td>0.483077</td>
</tr>
<tr>
<th>1</th>
<td>Male</td>
<td>189</td>
<td>87</td>
<td>2</td>
<td>1.164872</td>
<td>-0.587322</td>
<td>0.830508</td>
<td>0.336364</td>
<td>0.908381</td>
<td>0.418144</td>
</tr>
<tr>
<th>2</th>
<td>Female</td>
<td>185</td>
<td>110</td>
<td>4</td>
<td>0.920357</td>
<td>0.123647</td>
<td>0.762712</td>
<td>0.545455</td>
<td>0.859536</td>
<td>0.511075</td>
</tr>
<tr>
<th>3</th>
<td>Female</td>
<td>195</td>
<td>104</td>
<td>3</td>
<td>1.531645</td>
<td>-0.061823</td>
<td>0.932203</td>
<td>0.490909</td>
<td>0.882353</td>
<td>0.470588</td>
</tr>
<tr>
<th>4</th>
<td>Male</td>
<td>149</td>
<td>61</td>
<td>3</td>
<td>-1.280283</td>
<td>-1.391027</td>
<td>0.152542</td>
<td>0.100000</td>
<td>0.925448</td>
<td>0.378875</td>
</tr>
<tr>
<th>5</th>
<td>Male</td>
<td>189</td>
<td>104</td>
<td>3</td>
<td>1.164872</td>
<td>-0.061823</td>
<td>0.830508</td>
<td>0.490909</td>
<td>0.876118</td>
<td>0.482097</td>
</tr>
<tr>
<th>6</th>
<td>Male</td>
<td>147</td>
<td>92</td>
<td>5</td>
<td>-1.402541</td>
<td>-0.432764</td>
<td>0.118644</td>
<td>0.381818</td>
<td>0.847674</td>
<td>0.530517</td>
</tr>
<tr>
<th>7</th>
<td>Male</td>
<td>154</td>
<td>111</td>
<td>5</td>
<td>-0.974639</td>
<td>0.154559</td>
<td>0.237288</td>
<td>0.554545</td>
<td>0.811234</td>
<td>0.584721</td>
</tr>
<tr>
<th>8</th>
<td>Male</td>
<td>174</td>
<td>90</td>
<td>3</td>
<td>0.247939</td>
<td>-0.494587</td>
<td>0.576271</td>
<td>0.363636</td>
<td>0.888218</td>
<td>0.459423</td>
</tr>
<tr>
<th>9</th>
<td>Female</td>
<td>169</td>
<td>103</td>
<td>4</td>
<td>-0.057706</td>
<td>-0.092735</td>
<td>0.491525</td>
<td>0.481818</td>
<td>0.853906</td>
<td>0.520428</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">

<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-9fa4f38c-90e5-4856-9f45-01f4b9840f2e')"
title="Convert this dataframe to an interactive table."
style="display:none;">

</button>


</div>


</div>
</div>





```python
import pandas as pd
d  = pd.read_csv("titanic_dataset.csv")
d
```





<div id="df-6f4af63e-815a-4726-b21f-290c8d563089" class="colab-df-container">
<div>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
<th></th>
<th>PassengerId</th>
<th>Survived</th>
<th>Pclass</th>
<th>Name</th>
<th>Sex</th>
<th>Age</th>
<th>SibSp</th>
<th>Parch</th>
<th>Ticket</th>
<th>Fare</th>
<th>Cabin</th>
<th>Embarked</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>1</td>
<td>0</td>
<td>3</td>
<td>Braund, Mr. Owen Harris</td>
<td>male</td>
<td>22.0</td>
<td>1</td>
<td>0</td>
<td>A/5 21171</td>
<td>7.2500</td>
<td>NaN</td>
<td>S</td>
</tr>
<tr>
<th>1</th>
<td>2</td>
<td>1</td>
<td>1</td>
<td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
<td>female</td>
<td>38.0</td>
<td>1</td>
<td>0</td>
<td>PC 17599</td>
<td>71.2833</td>
<td>C85</td>
<td>C</td>
</tr>
<tr>
<th>2</th>
<td>3</td>
<td>1</td>
<td>3</td>
<td>Heikkinen, Miss. Laina</td>
<td>female</td>
<td>26.0</td>
<td>0</td>
<td>0</td>
<td>STON/O2. 3101282</td>
<td>7.9250</td>
<td>NaN</td>
<td>S</td>
</tr>
<tr>
<th>3</th>
<td>4</td>
<td>1</td>
<td>1</td>
<td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
<td>female</td>
<td>35.0</td>
<td>1</td>
<td>0</td>
<td>113803</td>
<td>53.1000</td>
<td>C123</td>
<td>S</td>
</tr>
<tr>
<th>4</th>
<td>5</td>
<td>0</td>
<td>3</td>
<td>Allen, Mr. William Henry</td>
<td>male</td>
<td>35.0</td>
<td>0</td>
<td>0</td>
<td>373450</td>
<td>8.0500</td>
<td>NaN</td>
<td>S</td>
</tr>
<tr>
<th>...</th>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
</tr>
<tr>
<th>886</th>
<td>887</td>
<td>0</td>
<td>2</td>
<td>Montvila, Rev. Juozas</td>
<td>male</td>
<td>27.0</td>
<td>0</td>
<td>0</td>
<td>211536</td>
<td>13.0000</td>
<td>NaN</td>
<td>S</td>
</tr>
<tr>
<th>887</th>
<td>888</td>
<td>1</td>
<td>1</td>
<td>Graham, Miss. Margaret Edith</td>
<td>female</td>
<td>19.0</td>
<td>0</td>
<td>0</td>
<td>112053</td>
<td>30.0000</td>
<td>B42</td>
<td>S</td>
</tr>
<tr>
<th>888</th>
<td>889</td>
<td>0</td>
<td>3</td>
<td>Johnston, Miss. Catherine Helen "Carrie"</td>
<td>female</td>
<td>NaN</td>
<td>1</td>
<td>2</td>
<td>W./C. 6607</td>
<td>23.4500</td>
<td>NaN</td>
<td>S</td>
</tr>
<tr>
<th>889</th>
<td>890</td>
<td>1</td>
<td>1</td>
<td>Behr, Mr. Karl Howell</td>
<td>male</td>
<td>26.0</td>
<td>0</td>
<td>0</td>
<td>111369</td>
<td>30.0000</td>
<td>C148</td>
<td>C</td>
</tr>
<tr>
<th>890</th>
<td>891</td>
<td>0</td>
<td>3</td>
<td>Dooley, Mr. Patrick</td>
<td>male</td>
<td>32.0</td>
<td>0</td>
<td>0</td>
<td>370376</td>
<td>7.7500</td>
<td>NaN</td>
<td>Q</td>
</tr>
</tbody>
</table>
<p>891 rows × 12 columns</p>
</div>
<div class="colab-df-buttons">

<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-6f4af63e-815a-4726-b21f-290c8d563089')"
title="Convert this dataframe to an interactive table."
style="display:none;">

</button>


</div>


<div id="id_3660a5b6-aa06-4c22-a529-2b8e5cb215f0">
<button class="colab-df-generate" onclick="generateWithVariable('d')"
title="Generate code using this dataframe."
style="display:none;">

</button>
</div>

</div>
</div>





```python
d.isnull()
```





<div id="df-0caed14f-76e2-4e90-973f-cabb68edf51a" class="colab-df-container">
<div>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
<th></th>
<th>PassengerId</th>
<th>Survived</th>
<th>Pclass</th>
<th>Name</th>
<th>Sex</th>
<th>Age</th>
<th>SibSp</th>
<th>Parch</th>
<th>Ticket</th>
<th>Fare</th>
<th>Cabin</th>
<th>Embarked</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>True</td>
<td>False</td>
</tr>
<tr>
<th>1</th>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
</tr>
<tr>
<th>2</th>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>True</td>
<td>False</td>
</tr>
<tr>
<th>3</th>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
</tr>
<tr>
<th>4</th>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>True</td>
<td>False</td>
</tr>
<tr>
<th>...</th>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
</tr>
<tr>
<th>886</th>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>True</td>
<td>False</td>
</tr>
<tr>
<th>887</th>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
</tr>
<tr>
<th>888</th>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>True</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>True</td>
<td>False</td>
</tr>
<tr>
<th>889</th>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
</tr>
<tr>
<th>890</th>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>False</td>
<td>True</td>
<td>False</td>
</tr>
</tbody>
</table>
<p>891 rows × 12 columns</p>
</div>
<div class="colab-df-buttons">

<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-0caed14f-76e2-4e90-973f-cabb68edf51a')"
title="Convert this dataframe to an interactive table."
style="display:none;">

</button>


</div>


</div>
</div>





```python
d.info()
```

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
#   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
0   PassengerId  891 non-null    int64  
1   Survived     891 non-null    int64  
2   Pclass       891 non-null    int64  
3   Name         891 non-null    object 
4   Sex          891 non-null    object 
5   Age          714 non-null    float64
6   SibSp        891 non-null    int64  
7   Parch        891 non-null    int64  
8   Ticket       891 non-null    object 
9   Fare         891 non-null    float64
10  Cabin        204 non-null    object 
11  Embarked     889 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB



```python
d.isnull().sum()
```




<div>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
<th></th>
<th>0</th>
</tr>
</thead>
<tbody>
<tr>
<th>PassengerId</th>
<td>0</td>
</tr>
<tr>
<th>Survived</th>
<td>0</td>
</tr>
<tr>
<th>Pclass</th>
<td>0</td>
</tr>
<tr>
<th>Name</th>
<td>0</td>
</tr>
<tr>
<th>Sex</th>
<td>0</td>
</tr>
<tr>
<th>Age</th>
<td>177</td>
</tr>
<tr>
<th>SibSp</th>
<td>0</td>
</tr>
<tr>
<th>Parch</th>
<td>0</td>
</tr>
<tr>
<th>Ticket</th>
<td>0</td>
</tr>
<tr>
<th>Fare</th>
<td>0</td>
</tr>
<tr>
<th>Cabin</th>
<td>687</td>
</tr>
<tr>
<th>Embarked</th>
<td>2</td>
</tr>
</tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>




```python
d = d.dropna()
```


```python
d.isnull().sum()
```




<div>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
<th></th>
<th>0</th>
</tr>
</thead>
<tbody>
<tr>
<th>PassengerId</th>
<td>0</td>
</tr>
<tr>
<th>Survived</th>
<td>0</td>
</tr>
<tr>
<th>Pclass</th>
<td>0</td>
</tr>
<tr>
<th>Name</th>
<td>0</td>
</tr>
<tr>
<th>Sex</th>
<td>0</td>
</tr>
<tr>
<th>Age</th>
<td>0</td>
</tr>
<tr>
<th>SibSp</th>
<td>0</td>
</tr>
<tr>
<th>Parch</th>
<td>0</td>
</tr>
<tr>
<th>Ticket</th>
<td>0</td>
</tr>
<tr>
<th>Fare</th>
<td>0</td>
</tr>
<tr>
<th>Cabin</th>
<td>0</td>
</tr>
<tr>
<th>Embarked</th>
<td>0</td>
</tr>
</tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>




```python
from sklearn.feature_selection import SelectKBest,f_classif
x = d[['PassengerId','Pclass','Age','SibSp','Parch','Fare']]
y = d['Survived']
selector = SelectKBest(score_func=f_classif,k=4)
x_n = selector.fit_transform(x,y)
select_feature_indices = selector.get_support(indices=True)
selected_features = x.columns[select_feature_indices]
print("Selected features")
print(selected_features)
```

Selected features
Index(['PassengerId', 'Age', 'SibSp', 'Fare'], dtype='object')



```python
import pandas as pd
import numpy as np
import seaborn as sns

tip = sns.load_dataset('tips')
tip.head()
```





<div id="df-8fc1b4ec-83e6-4742-a81c-3d24e81560e8" class="colab-df-container">
<div>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
<th></th>
<th>total_bill</th>
<th>tip</th>
<th>sex</th>
<th>smoker</th>
<th>day</th>
<th>time</th>
<th>size</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>16.99</td>
<td>1.01</td>
<td>Female</td>
<td>No</td>
<td>Sun</td>
<td>Dinner</td>
<td>2</td>
</tr>
<tr>
<th>1</th>
<td>10.34</td>
<td>1.66</td>
<td>Male</td>
<td>No</td>
<td>Sun</td>
<td>Dinner</td>
<td>3</td>
</tr>
<tr>
<th>2</th>
<td>21.01</td>
<td>3.50</td>
<td>Male</td>
<td>No</td>
<td>Sun</td>
<td>Dinner</td>
<td>3</td>
</tr>
<tr>
<th>3</th>
<td>23.68</td>
<td>3.31</td>
<td>Male</td>
<td>No</td>
<td>Sun</td>
<td>Dinner</td>
<td>2</td>
</tr>
<tr>
<th>4</th>
<td>24.59</td>
<td>3.61</td>
<td>Female</td>
<td>No</td>
<td>Sun</td>
<td>Dinner</td>
<td>4</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">

<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-8fc1b4ec-83e6-4742-a81c-3d24e81560e8')"
title="Convert this dataframe to an interactive table."
style="display:none;">

</button>


</div>


</div>
</div>





```python
tip.isnull().sum()
```




<div>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
<th></th>
<th>0</th>
</tr>
</thead>
<tbody>
<tr>
<th>total_bill</th>
<td>0</td>
</tr>
<tr>
<th>tip</th>
<td>0</td>
</tr>
<tr>
<th>sex</th>
<td>0</td>
</tr>
<tr>
<th>smoker</th>
<td>0</td>
</tr>
<tr>
<th>day</th>
<td>0</td>
</tr>
<tr>
<th>time</th>
<td>0</td>
</tr>
<tr>
<th>size</th>
<td>0</td>
</tr>
</tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>




```python
contigency_table = pd.crosstab(tip['sex'],tip['time'])
print(contigency_table)
```

time    Lunch  Dinner
sex                  
Male       33     124
Female     35      52



```python
from scipy.stats import chi2_contingency
chi2,p,_,_ = chi2_contingency(contigency_table)
print(f"chi square statistics: {chi2}")
print(f"P-value: {p}")
```

chi square statistics: 9.343808982970623
P-value: 0.002237400118075248



```python
from sklearn.feature_selection import SelectKBest,mutual_info_regression,f_classif
d = {
"FEATURE1":[1,2,3,4,5],
"FEATURE2":['A','B','C','A','B'],
"FEATURE3":[0,1,1,0,1],
"FEATURE4":[0,1,1,0,1]
}
selector = SelectKBest(score_func=f_classif,k=4)
x_n = selector.fit_transform(x,y)
select_feature_indices = selector.get_support(indices=True)
selected_features = x.columns[select_feature_indices]
print("Selected features")
print(selected_features)
```

Selected features
Index(['PassengerId', 'Age', 'SibSp', 'Fare'], dtype='object')



# RESULT:
 Thus, Feature selection and Feature scaling has been used on the given dataset.
