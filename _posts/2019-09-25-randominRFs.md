---
title: "The Random in Random Forests"
last_modified_at: 2019-10-15T15:19:02-05:00
layout: post
categories:
  - Blog
tags:
  - ML
  - Fish
---
# 1. The Random in Random Forests
Random forests (RF) are my default starting point for most of the data science I do at Texas A&M. Random forests are the honey badgers of machine learning. **They don't care**. 

![dont care](http://giphygifs.s3.amazonaws.com/media/f8k6R32qjJGV2/giphy.gif "RF doesn't care")

They don't care about normalizing to the mean and standard deviation. They don't care about hyperparameters or tuning. You may not be able to get that last 1% of accuracy versus other methods, but RFs are easy to understand and visualize.

## 1.2 An Observation for Motivation
There's this prevailing idea in a number of [foundational](https://ieeexplore.ieee.org/document/598994) [papers](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) about Random Forests.

>Average a bunch of *bad* trees, and you get a *good* prediction.

From a purely intuitive sense, that's a weird concept. We show the machine lots of different data sets, and ask it to make the best possible prediction, which ends up being lots of not so good predictions. Let's take a closer look with a really simple [dataset](https://www.kaggle.com/aungpyaeap/fish-market) and the [fastai](https://www.fast.ai/) library.

***
Quick and shameless plug for [fastai](https://www.fast.ai/). If you're even a little bit interested in using ML, this is such a great resource. The top down lecturing approach is much more intuitive than the statistics up approach I've seen at university. Lots of discussion about broader impacts of the technology too. This whole post is based on something I've learned while working through the fastai courses.

# 2. Setup
***


```python
%load_ext autoreload
%autoreload 2

%matplotlib inline
```
<!-- 
    <!-- The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload --> 
    


```python
from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics

import pandas as pd
import graphviz
print("Setup Complete.")
```

## 2.1 Read data


```python
PATH = "data/fish-market/"
```


```python
df_raw = pd.read_csv('data/fish-market/Fish.csv') #read the raw data
df_raw.head() #display the first few rows, verification
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
      <th>Species</th>
      <th>Weight</th>
      <th>Length1</th>
      <th>Length2</th>
      <th>Length3</th>
      <th>Height</th>
      <th>Width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bream</td>
      <td>242.0</td>
      <td>23.2</td>
      <td>25.4</td>
      <td>30.0</td>
      <td>11.5200</td>
      <td>4.0200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bream</td>
      <td>290.0</td>
      <td>24.0</td>
      <td>26.3</td>
      <td>31.2</td>
      <td>12.4800</td>
      <td>4.3056</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bream</td>
      <td>340.0</td>
      <td>23.9</td>
      <td>26.5</td>
      <td>31.1</td>
      <td>12.3778</td>
      <td>4.6961</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bream</td>
      <td>363.0</td>
      <td>26.3</td>
      <td>29.0</td>
      <td>33.5</td>
      <td>12.7300</td>
      <td>4.4555</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bream</td>
      <td>430.0</td>
      <td>26.5</td>
      <td>29.0</td>
      <td>34.0</td>
      <td>12.4440</td>
      <td>5.1340</td>
    </tr>
  </tbody>
</table>
</div>



This is a happy dataset. We've got 7 species of fish with relevant measurements. Here's a Bream fish for example. We can target weight as our output variable for the time being.
![alt text](https://cdn.pixabay.com/photo/2018/05/17/16/33/sea-bream-3409033_960_720.png "Bream Fish")
This is an especially nice set because there's no time interaction between the variables. Variable interaction alone is worth a whole separate post but  note that most datasets need more feature engineering before they can be learned. Since this is such an easy data set, I'm not going to focus on getting maximum accuracy, but rather keeping the forest small so we can visualize it easily.

## 2.2 Prepare data


```python
df_raw.Weight = np.log(df_raw.Weight) #convert target variable to log scale, usually plots nicer
```


```python
os.makedirs('tmp', exist_ok=True) #create a folder for raw feather files
df_raw.to_feather('tmp/fish-raw') #write raw data to feather so we don't have to keep reading a slow csv file
```


```python
df_raw = pd.read_feather('tmp/fish-raw') #sushis and sashimis
df_raw.head() #make sure the feather was read correctly
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
      <th>Species</th>
      <th>Weight</th>
      <th>Length1</th>
      <th>Length2</th>
      <th>Length3</th>
      <th>Height</th>
      <th>Width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bream</td>
      <td>5.488938</td>
      <td>23.2</td>
      <td>25.4</td>
      <td>30.0</td>
      <td>11.5200</td>
      <td>4.0200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bream</td>
      <td>5.669881</td>
      <td>24.0</td>
      <td>26.3</td>
      <td>31.2</td>
      <td>12.4800</td>
      <td>4.3056</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bream</td>
      <td>5.828946</td>
      <td>23.9</td>
      <td>26.5</td>
      <td>31.1</td>
      <td>12.3778</td>
      <td>4.6961</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bream</td>
      <td>5.894403</td>
      <td>26.3</td>
      <td>29.0</td>
      <td>33.5</td>
      <td>12.7300</td>
      <td>4.4555</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bream</td>
      <td>6.063785</td>
      <td>26.5</td>
      <td>29.0</td>
      <td>34.0</td>
      <td>12.4440</td>
      <td>5.1340</td>
    </tr>
  </tbody>
</table>
</div>



## 2.3 Visualize the data


```python
fig, axs = plt.subplots(1, 6, figsize=(15, 5)) #visualize the initial data
axs[0].hist(df_raw.Length1) 
axs[0].set_title('Length1')
axs[1].hist(df_raw.Length2)
axs[1].set_title('Length2')
axs[2].hist(df_raw.Length3)
axs[2].set_title('Length3')
axs[3].hist(df_raw.Height)
axs[3].set_title('Height')
axs[4].hist(df_raw.Width)
axs[4].set_title('Width')
axs[5].hist(df_raw.Weight)
axs[5].set_title('Weight')
```
<!-- 



    Text(0.5, 1.0, 'Weight') -->




![png](/assets/images/output_12_1.png)


Super manageable data set. You could probably get away with some kind of [5 dimension linear hyperplane](https://www.kaggle.com/akdagmelih/multiplelinear-regression-fish-weight-estimation) if you wanted. 

## 2.4 Spit data into test and training sets


```python
train_cats(df_raw) #enumerate categorical variables
df, y, nas = proc_df(df_raw, 'Weight') #split target variable from training data
```


```python
msk = np.random.rand(len(df)) < 0.8 #set ratio for training and test set sizes (80% of sample in training)
X_train = df[msk] 
y_train=y[msk]
X_valid = df[~msk]
y_valid = y[~msk]
X_train.shape, y_train.shape, X_valid.shape, y_valid.shape #verify sizes of datasets
```




    ((135, 6), (135,), (24, 6), (24,))




```python
def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
```

# 3. Train the Model


```python
m = RandomForestRegressor(n_jobs=-1, n_estimators=10 , max_depth=4) #limit the size of the tree for visualization
%time m.fit(X_train, y_train)
print_score(m)
```

    Wall time: 114 ms
    [0.105640573238319, 0.18335970391783124, 0.9936340002864559, 0.9798625436302886]
    


```python
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
```

# 4. Analysis and Results
Let's take a look at a single tree visually. Notice that this particular tree is splitting on Length3, the diagonal length of the fish, a number of times. This suggests that variable is particularly important. The specifics of the tree aren't super critical here, it's mostly a nice visualization.


```python
s=export_graphviz(m.estimators_[0], out_file=None, feature_names=X_train.columns, filled=True,
                      special_characters=True, rotate=True, precision=3)
graph = graphviz.Source(s) 
graph
```




![svg](/assets/images/output_22_0.svg)




```python
preds = np.stack([t.predict(X_valid) for t in m.estimators_])
final_out=(preds[:,0], np.mean(preds[:,0]), y_valid[0]) # this is the key metric for the point I'm trying to make
final_out #an object with each of the predictions for a single sample, combined with the final estimate and the truth
```




    (array([5.64993, 5.85184, 5.65803, 5.56792, 5.57952, 5.57955, 5.78348, 5.34795, 5.88255, 5.80122]),
     5.670200821131428,
     5.66988092298052)



There's the result. Let's look at this graphically.


```python
preds = np.stack([t.predict(X_valid) for t in m.estimators_])
final_out=(preds[:,0])
final_out=np.append(final_out,np.mean(preds[:,0]))
final_out=np.append(final_out,y_valid[0]) #CS folks, how to concat scalars w/ 1D array?
final_delta=final_out-y_valid[0]
labels = ['T1', 'T2', 'T3', 'T4', 'T5','T6', 'T7', 'T8', 'T9', 'T10','Forest','Truth']
```

# 5. Main Conclusions


```python
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].grid(zorder=0)
ax[0].bar(range(len(final_out)), final_out, width=0.8, color='steelblue', zorder=3)
ax[0].set_xticks(range(len(final_out)))
ax[0].set_xticklabels(labels)
ax[1].grid(zorder=0)
ax[1].bar(range(len(final_delta)), final_delta, width=0.8, align='center', color='tomato', zorder=3)
ax[1].set_xticks(range(len(final_delta)))
ax[1].set_xticklabels(labels);
```


![png](/assets/images/output_27_0.png)


<!-- 
```python
def color_table(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    if val < 0:
        color = 'red' 
    elif 0 < val < .01:
        color = 'blue'
    elif 0 < val < 1:
        color = 'green'
    else:
        color = 'black'
    return 'color: %s' % color
``` -->


```python
result_table = pd.read_csv('data/fish-market/results.csv', index_col=0)
s = result_table.style.applymap(color_table)
s
```




<style  type="text/css" >
    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow0_col0 {
            color:  black;
        }    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow0_col1 {
            color:  black;
        }    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow0_col2 {
            color:  red;
        }    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow1_col0 {
            color:  black;
        }    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow1_col1 {
            color:  black;
        }    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow1_col2 {
            color:  green;
        }    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow2_col0 {
            color:  black;
        }    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow2_col1 {
            color:  black;
        }    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow2_col2 {
            color:  red;
        }    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow3_col0 {
            color:  black;
        }    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow3_col1 {
            color:  black;
        }    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow3_col2 {
            color:  red;
        }    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow4_col0 {
            color:  black;
        }    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow4_col1 {
            color:  black;
        }    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow4_col2 {
            color:  red;
        }    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow5_col0 {
            color:  black;
        }    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow5_col1 {
            color:  black;
        }    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow5_col2 {
            color:  red;
        }    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow6_col0 {
            color:  black;
        }    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow6_col1 {
            color:  black;
        }    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow6_col2 {
            color:  green;
        }    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow7_col0 {
            color:  black;
        }    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow7_col1 {
            color:  black;
        }    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow7_col2 {
            color:  red;
        }    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow8_col0 {
            color:  black;
        }    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow8_col1 {
            color:  black;
        }    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow8_col2 {
            color:  green;
        }    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow9_col0 {
            color:  black;
        }    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow9_col1 {
            color:  black;
        }    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow9_col2 {
            color:  green;
        }    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow10_col0 {
            color:  black;
        }    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow10_col1 {
            color:  black;
        }    #T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow10_col2 {
            color:  blue;
        }</style><table id="T_0fae72ac_ef62_11e9_9c38_001583f72bbf" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Prediction</th>        <th class="col_heading level0 col1" >Truth</th>        <th class="col_heading level0 col2" >delta_truth</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_0fae72ac_ef62_11e9_9c38_001583f72bbflevel0_row0" class="row_heading level0 row0" >Tree 1</th>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow0_col0" class="data row0 col0" >5.64993</td>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow0_col1" class="data row0 col1" >5.66988</td>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow0_col2" class="data row0 col2" >-0.0199509</td>
            </tr>
            <tr>
                        <th id="T_0fae72ac_ef62_11e9_9c38_001583f72bbflevel0_row1" class="row_heading level0 row1" >Tree 2</th>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow1_col0" class="data row1 col0" >5.85184</td>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow1_col1" class="data row1 col1" >5.66988</td>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow1_col2" class="data row1 col2" >0.181959</td>
            </tr>
            <tr>
                        <th id="T_0fae72ac_ef62_11e9_9c38_001583f72bbflevel0_row2" class="row_heading level0 row2" >Tree 3</th>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow2_col0" class="data row2 col0" >5.65803</td>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow2_col1" class="data row2 col1" >5.66988</td>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow2_col2" class="data row2 col2" >-0.0118509</td>
            </tr>
            <tr>
                        <th id="T_0fae72ac_ef62_11e9_9c38_001583f72bbflevel0_row3" class="row_heading level0 row3" >Tree 4</th>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow3_col0" class="data row3 col0" >5.56792</td>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow3_col1" class="data row3 col1" >5.66988</td>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow3_col2" class="data row3 col2" >-0.101961</td>
            </tr>
            <tr>
                        <th id="T_0fae72ac_ef62_11e9_9c38_001583f72bbflevel0_row4" class="row_heading level0 row4" >Tree 5</th>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow4_col0" class="data row4 col0" >5.57952</td>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow4_col1" class="data row4 col1" >5.66988</td>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow4_col2" class="data row4 col2" >-0.0903609</td>
            </tr>
            <tr>
                        <th id="T_0fae72ac_ef62_11e9_9c38_001583f72bbflevel0_row5" class="row_heading level0 row5" >Tree 6</th>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow5_col0" class="data row5 col0" >5.57955</td>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow5_col1" class="data row5 col1" >5.66988</td>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow5_col2" class="data row5 col2" >-0.0903309</td>
            </tr>
            <tr>
                        <th id="T_0fae72ac_ef62_11e9_9c38_001583f72bbflevel0_row6" class="row_heading level0 row6" >Tree 7</th>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow6_col0" class="data row6 col0" >5.78348</td>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow6_col1" class="data row6 col1" >5.66988</td>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow6_col2" class="data row6 col2" >0.113599</td>
            </tr>
            <tr>
                        <th id="T_0fae72ac_ef62_11e9_9c38_001583f72bbflevel0_row7" class="row_heading level0 row7" >Tree 8</th>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow7_col0" class="data row7 col0" >5.34795</td>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow7_col1" class="data row7 col1" >5.66988</td>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow7_col2" class="data row7 col2" >-0.321931</td>
            </tr>
            <tr>
                        <th id="T_0fae72ac_ef62_11e9_9c38_001583f72bbflevel0_row8" class="row_heading level0 row8" >Tree 9</th>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow8_col0" class="data row8 col0" >5.88255</td>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow8_col1" class="data row8 col1" >5.66988</td>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow8_col2" class="data row8 col2" >0.212669</td>
            </tr>
            <tr>
                        <th id="T_0fae72ac_ef62_11e9_9c38_001583f72bbflevel0_row9" class="row_heading level0 row9" >Tree 10</th>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow9_col0" class="data row9 col0" >5.80122</td>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow9_col1" class="data row9 col1" >5.66988</td>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow9_col2" class="data row9 col2" >0.131339</td>
            </tr>
            <tr>
                        <th id="T_0fae72ac_ef62_11e9_9c38_001583f72bbflevel0_row10" class="row_heading level0 row10" >Average (Forest)</th>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow10_col0" class="data row10 col0" >5.6702</td>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow10_col1" class="data row10 col1" >5.66988</td>
                        <td id="T_0fae72ac_ef62_11e9_9c38_001583f72bbfrow10_col2" class="data row10 col2" >0.000319898</td>
            </tr>
    </tbody></table>



We're looking at a single estimate of the weight for our forest with 10 trees in it. The first 10 bars show the weight prediction of a **single** fish. The 11th bar is the average of all the predicted weights from our trees, which makes the forest. The last bar is the actual weight. 

Notice every tree is a little off. Tree 8 is predicting way too low, while trees 2, 9, and 10 are predicting too high. Think of this as each tree exploiting a different trend in the data. 

However, the average across all trees gives an estimate that is incredibly accurate. So much so that in this example the difference between the true weight and the forest predicted weight is invisible on the second plot. 

Let's revisit our initial motivation:

>Average a bunch of *bad* trees, and you get a *good* prediction.

Perhaps this statement is misleading. Maybe what I really mean to say is:
>The average of random error is **zero**. 

If all the trees exploit different paths through the data, the model generalizes and yields the best prediction. It is *randomly* exploring the trends in the data, putting the random in random forest. You can download this notebook [here](https://drive.google.com/file/d/1oUDjM0dYWXoESoxgQIwU1MeaQiOKMlBM/view?usp=sharing).

