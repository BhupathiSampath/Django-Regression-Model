from django.shortcuts import render
from regression.forms import InputForm
import requests
# Create your views here.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def home_view(request):
    context ={}
    # context['form']= InputForm()
    # name = request.POST.get("name")
    # email = request.POST.get("email")
    # message = request.POST.get("message")
    
    # form = InputForm(request.POST)
    # # name = form['name'].value()
    # email = form['name'].email()
    # message = form['message'].value()
    return render(request, "home.html")
    # return render(request, "home.html", {"context": context, "name": form['name'].value()})


def results(request):
    df=pd.read_csv("C:/Users/BhupathiSampath/OneDrive - JMAN Group Ltd/Documents/House_Rent_Prediction/test.csv")
    df['Rent'] = df['Rent'].replace(['-'],'0')
    df['Rent']=pd.to_numeric(df['Rent'])
    df['MaintenanceCharge'] = df['MaintenanceCharge'].replace(['No'],'0')
    df['MaintenanceCharge']=pd.to_numeric(df['MaintenanceCharge'])
    df[["Area","Measure"]]= df["Area"].str.split(" ",expand = True)
    df['Area'] = df['Area'].str.replace(',','')
    df['Area']=pd.to_numeric(df['Area'])
    df = df.drop(["Unnamed: 0","Title", "Address", "AvailableFrom", "Measure"], axis=1)
    df["Area"] = df["Area"].apply(lambda x: np.log(x) if not x==0 else x)
    df["Rent"] = df["Rent"].apply(lambda x: np.log(x) if not x==0 else x)
    for i in ["Rent","Area"]:
        Q1 = np.percentile(df[i], 25, interpolation = 'midpoint')

        Q3 = np.percentile(df[i], 75, interpolation = 'midpoint')
        IQR = Q3 - Q1

        # print("Old Shape: ", df.shape)

        # Upper bound
        upper = np.where(df[i] >= (Q3+1.5*IQR))
        # Lower bound
        lower = np.where(df[i] <= (Q1-1.5*IQR))

        ''' Removing the Outliers '''
        df.drop(upper[0], inplace = True)
        df.drop(lower[0], inplace = True)
    df.drop(["Place"], axis=1, inplace=True)
    df = pd.get_dummies(df)
    X = df.drop(["Rent"], axis=1)
    print(X.columns)
    y = df['Rent']
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state=5)
    lin_model = LinearRegression()
    lin_model.fit(X_train, Y_train)

    area = float(request.GET["area"])
    deposit = float(request.GET["deposit"])
    MaintenanceCharge = float(request.GET["MaintenanceCharge"])
    Furnished = request.GET["Furnished"]
    ApartmentType = request.GET["ApartmentType"]
    TenantType = request.GET["TenantType"]

    print("Area",area)
    Furnished_Fully = 0
    Furnished_Semi = 0
    Furnished_Unfurnished = 0
    if Furnished == "Fully furnished":
        Furnished_Fully = 1
        Furnished_Semi = 0
        Furnished_Unfurnished = 0
    if Furnished == "Semi furnished":
        Furnished_Fully = 0
        Furnished_Semi = 1
        Furnished_Unfurnished = 0
    if Furnished == "Unfurnished":
        Furnished_Fully = 0
        Furnished_Semi = 0
        Furnished_Unfurnished = 1
    ApartmentType_1rk = 0
    ApartmentType_1 = 0
    ApartmentType_2 = 0
    ApartmentType_3 = 0
    ApartmentType_4 = 0
    ApartmentType_4p = 0
    if ApartmentType == '1 RK':
        ApartmentType_1rk = 1
        ApartmentType_1 = 0
        ApartmentType_2 = 0
        ApartmentType_3 = 0
        ApartmentType_4 = 0
        ApartmentType_4p = 0
    if ApartmentType == '1 BHK':
        ApartmentType_1rk = 0
        ApartmentType_1 = 1
        ApartmentType_2 = 0
        ApartmentType_3 = 0
        ApartmentType_4 = 0
        ApartmentType_4p = 0
    if ApartmentType == '2 BHK':
        ApartmentType_1rk = 0
        ApartmentType_1 = 0
        ApartmentType_2 = 1
        ApartmentType_3 = 0
        ApartmentType_4 = 0
        ApartmentType_4p = 0
    if ApartmentType == '3 BHK':
        ApartmentType_1rk = 0
        ApartmentType_1 = 0
        ApartmentType_2 = 0
        ApartmentType_3 = 1
        ApartmentType_4 = 0
        ApartmentType_4p = 0
    if ApartmentType == '4 BHK':
        ApartmentType_1rk = 0
        ApartmentType_1 = 0
        ApartmentType_2 = 0
        ApartmentType_3 = 0
        ApartmentType_4 = 1
        ApartmentType_4p = 0
    if ApartmentType == '4+ BHK':
        ApartmentType_1rk = 0
        ApartmentType_1 = 0
        ApartmentType_2 = 0
        ApartmentType_3 = 0
        ApartmentType_4 = 0
        ApartmentType_4p = 1
    TenantType_Family = 0
    TenantType_Company = 0
    TenantType_Bachelor = 0
    TenantType_All = 0
    if TenantType == "Bachelor":
        TenantType_All = 0
        TenantType_Family = 0
        TenantType_Company = 0
        TenantType_Bachelor = 1
    if TenantType == "Family":
        TenantType_All = 0
        TenantType_Family = 1
        TenantType_Company = 0
        TenantType_Bachelor = 0
    if TenantType == "Company":
        TenantType_All = 0
        TenantType_Family = 0
        TenantType_Company = 1
        TenantType_Bachelor = 0
    if TenantType == "All":
        TenantType_All = 1
        TenantType_Family = 0
        TenantType_Company = 0
        TenantType_Bachelor = 0
        

    test_data = {
        "area": area,
        "deposit": deposit,
        "MaintenanceCharge": MaintenanceCharge,
        "Furnished_Fully furnished": Furnished_Fully,
        "Furnished_Semi furnished": Furnished_Semi,
        "Furnished_Unfurnished": Furnished_Unfurnished,
        "ApartmentType_1 BHK": ApartmentType_1, 
        "ApartmentType_1 RK": ApartmentType_1rk,
        "ApartmentType_2 BHK": ApartmentType_2, 
        "ApartmentType_3 BHK": ApartmentType_3, 
        "ApartmentType_4 BHK": ApartmentType_4,
        "ApartmentType_4+ BHK": ApartmentType_4p,
        "TenantType_All": TenantType_All,
        "TenantType_Family": TenantType_Family,
        "TenantType_Company": TenantType_Company,
        "TenantType_Bachelor": TenantType_Bachelor,
    }

    test_data["area"] = np.log(test_data["area"])
    y_test_predict = lin_model.predict([list(test_data.values())])
    # test_data["rent"] = np.log(test_data["rent"])
    print("predicted_value",y_test_predict[0])

    return render(request,'home.html', {"Result": np.exp(y_test_predict[0])})