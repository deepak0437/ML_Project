from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

def index(request):
    return render(request,'index.html')
##-------------------------------------------------------------------------
def predict(request):
    return render(request, 'predict.html')
##--------------------------------------------------------------------------
def result(request):
    data = pd.read_csv(r"E:\datasets\diabetes11.csv")

    x = data.drop('Outcome', axis=1)
    y = data['Outcome']
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25, random_state = 0)

    model = LogisticRegression()
    model.fit(x_train,y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])
    
    result1 = ""
    if pred == [1]:
        result1 = "Your Diabetes report is positive, Please Concern to your nearby doctor."
    else:
        result1 = "Your Diabetes report is negative, you have no issue related diabetes."

    return render(request, 'result.html', {"result2": result1})
##------------------------------------------------------------------------------

def marks(request):
    return render(request, 'marks.html')
##------------------------------------------------------------------------------

def result_marks(request):
    df = pd.read_csv(r"E:\datasets\Student_marks.csv")
    x = df.drop('Result', axis=1)
    y = df['Result']
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25, random_state = 0)

    
    classifier = DecisionTreeClassifier()
    classifier.fit(x_train,y_train)

    val_1 = int(request.GET['n1'])
    val_2 = int(request.GET['n2'])
    val_3 = int(request.GET['n3'])

    pred1 = classifier.predict([[val_1, val_2, val_3]])
    

    result_m = ""
    if pred1 == [1]:
        result_m = "Your Maths marks is:",val_1,",Physics marks is:",val_2,"and Chemistry marks is:",val_3,".So, you are PASS."
    else:
        result_m = "Your Maths marks is:",val_1,",Physics marks is:",val_2,"and Chemistry marks is:",val_3,".So, you are FAIL."

    return render(request, 'result_marks.html', {"result3": result_m})
##------------------------------------------------------------------------------

def wine(request):
    return render(request, 'wine.html')
##------------------------------------------------------------------------------

def result_wine(request):
    df = pd.read_csv(r"E:\datasets\wine.csv")
    df.isna().sum()
    #####
    mean = df['fixed acidity'].mean()
    df['fixed acidity'].fillna(mean , inplace = True)
    df['fixed acidity'].isna().sum()
    #####
    mean = df['volatile acidity'].mean()
    df['volatile acidity'].fillna(mean , inplace = True)
    df['volatile acidity'].isna().sum()
    #####
    mean = df['citric acid'].mean()
    df['citric acid'].fillna(mean , inplace = True)
    df['citric acid'].isna().sum()
    #####
    mean = df['residual sugar'].mean()
    df['residual sugar'].fillna(mean , inplace = True)
    df['residual sugar'].isna().sum()
    #####
    mean = df['chlorides'].mean()
    df['chlorides'].fillna(mean , inplace = True)
    df['chlorides'].isna().sum()
    #####
    mean = df['pH'].mean()
    df['pH'].fillna(mean , inplace = True)
    df['pH'].isna().sum()
    #####
    mean = df['sulphates'].mean()
    df['sulphates'].fillna(mean , inplace = True)
    df['sulphates'].isna().sum()

    df.isna().sum()
    from sklearn.preprocessing import LabelEncoder
    model = LabelEncoder()
    df['type'] = model.fit_transform(df['type'])

    mapping = {3: "Low" , 4: "Low" , 5: "Medium" , 6: "Medium" , 7:"Medium" , 8:"High" , 9:"High"}
    df['quality'] = df['quality'].map(mapping)

    mapping = {"Low" : 0 , "Medium" : 1 , "High" : 2}
    df['quality'] = df['quality'].map(mapping)

    x = df.drop('quality', axis=True)
    y = df['quality']

    x_train, x_test, y_train, y_test = train_test_split(x , y , test_size = 0.25 , random_state = 0)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    regressor1 = LogisticRegression(multi_class='ovr')
    regressor1.fit(x_train,y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])
    val9 = float(request.GET['n9'])
    val10 = float(request.GET['n10'])
    val11 = float(request.GET['n11'])
    val12 = float(request.GET['n12'])

    pred2 = regressor1.predict([[val1, val2, val3, val4, val5, val6, val7, val8, val9,val10, val11, val12]])
    
    result_w = ""
    if pred2 == [0]:
        result_w = "Your wine quality is LOW."
    elif pred2 == [1]:
        result_w = "Your wine quality is MEDIUM."
    elif pred2 == [2]:
        result_w = "Your wine quality is HIGH."

    return render(request, 'result_wine.html', {"result4": result_w})

##------------------------------------------------------------------------------

def house(request):
    return render(request, 'house.html')
##------------------------------------------------------------------------------

def result_house(request):
    df = pd.read_csv(r"E:\datasets\USA_Housing.csv")
    df = df.drop(['Address'], axis = 1)

    x = df.drop('Price', axis=1)
    y = df['Price']

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25, random_state = 0)

    model = LinearRegression()
    model.fit(x_train,y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])

    pred3 = model.predict(np.array([val1, val2, val3, val4, val5]).reshape(1,-1))
    pred3 = round(pred3[0])

    price = "Your predicted house price is Rs."+str(pred3)
    return render(request, 'result_house.html', {"result5": price})

##------------------------------------------------------------------------------

def insurance(request):
    return render(request, 'insurance.html')
##------------------------------------------------------------------------------

def result_insurance(request):
    df = pd.read_csv(r"E:\datasets\insurance.csv")
    df['sex'].unique()
    df['sex'] = df['sex'].map({'female':0, 'male':1})

    df['smoker'].unique()
    df['smoker'] = df['smoker'].map({'no':0, 'yes':1})

    df['region'].unique()
    df['region'] = df['region'].map({'southwest':1, 'southeast':2, 'northwest':3, 'northeast':4})
    
    x = df.drop('expenses', axis=1)
    y = df['expenses']

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25, random_state = 0)

    from sklearn.ensemble import GradientBoostingRegressor
    gr = GradientBoostingRegressor()
    gr.fit(x_train,y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])

    pred4 = gr.predict(np.array([val1, val2, val3, val4, val5,val6]).reshape(1,-1))
    pred4 = round(pred4[0])

    insurance = "The predicted Insurance of premium policy is Rs."+str(pred4)

    return render(request, 'result_insurance.html', {"result6": insurance})

##------------------------------------------------------------------------------

def cement(request):
    return render(request, 'cement.html')
##------------------------------------------------------------------------------

def result_cement(request):
    df = pd.read_csv(r"E:\datasets\concrete_data.csv")
    
    x = df.drop('concrete_compressive_strength', axis=1)
    y = df['concrete_compressive_strength']

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25, random_state = 0)

    from sklearn.ensemble import GradientBoostingRegressor
    gr = GradientBoostingRegressor()
    gr.fit(x_train,y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    pred5 = gr.predict(np.array([val1, val2, val3, val4, val5,val6, val7, val8]).reshape(1,-1))
    pred5 = round(pred5[0])

    cement = "The average compressive strength of the concrete is "+str(pred5)+" MPa"

    return render(request, 'result_cement.html', {"result7": cement})

##------------------------------------------------------------------------------

def require(request):
    return render(request, 'require.html')
##------------------------------------------------------------------------------