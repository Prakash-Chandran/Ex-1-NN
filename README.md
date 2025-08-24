<H3>ENTER YOUR NAME : PRAKASH C</H3>
<H3>ENTER YOUR REGISTER NO : 212223240122</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
````
Name : Prakash C
Reg No : 212223240122
````
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("Churn_Modelling.csv")
data
data.head()

X=data.iloc[:,:-1].values
X

y=data.iloc[:,-1].values
y

data.isnull().sum()

data.duplicated()

data.describe()

data = data.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

X_train

X_test

print("Lenght of X_test ",len(X_test))


```


## OUTPUT:

### Dataset:
<img width="1111" height="215" alt="Screenshot 2025-08-24 073626" src="https://github.com/user-attachments/assets/b289ebe8-9a1f-4ff8-8d33-537ca35af9b2" />

### X Values:
<img width="563" height="136" alt="Screenshot 2025-08-24 073632" src="https://github.com/user-attachments/assets/756a1997-5fb2-438a-8872-adeca242a802" />

### Y Values:
<img width="265" height="37" alt="Screenshot 2025-08-24 073637" src="https://github.com/user-attachments/assets/a31d57f6-71ef-4239-a5e0-68e2416a6860" />

### Null Values:
<img width="228" height="270" alt="Screenshot 2025-08-24 073644" src="https://github.com/user-attachments/assets/24be3936-c1a6-4f8f-948a-b1b3722b5175" />

### Duplicated Values:
<img width="252" height="214" alt="Screenshot 2025-08-24 073651" src="https://github.com/user-attachments/assets/27f0240a-71a6-40c4-b5c6-008e1320185c" />

### Description:
<img width="1111" height="302" alt="Screenshot 2025-08-24 073704" src="https://github.com/user-attachments/assets/2e9ba9c5-2eeb-4b5c-b349-0a5d7a60ec29" />

### Normalized Dataset:
<img width="926" height="189" alt="Screenshot 2025-08-24 073712" src="https://github.com/user-attachments/assets/099ff21d-6c10-4580-8a91-70b86b0aa671" />

### Testing Data:
<img width="725" height="468" alt="Screenshot 2025-08-24 073721" src="https://github.com/user-attachments/assets/3017d367-b918-4560-a6e4-e1b9f80798c6" />

<img width="210" height="26" alt="Screenshot 2025-08-24 073733" src="https://github.com/user-attachments/assets/37eb8b79-9c97-470f-b1f6-1d2ed6cf22b5" />



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


