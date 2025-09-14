import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import datetime
from sklearn.decomposition import PCA
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans

df = pd.read_csv("GlobalLandTemperaturesByCityEdited.csv")

#finding null values
null_value = df.isna().sum()
print(null_value)
Shape_df = df.shape
print(Shape_df)

#Deleting null entries
df.dropna(inplace=True)
Shape_df = df.shape
print(Shape_df)

#finding Duplicates
duplicates = df[df.duplicated()]
print(duplicates)
# No data duplicated, if there is any, use : df.drop_duplicated(keep=False)


#Finding the datatype of each column :
column = df.columns
for x in column :
    if df[x].dtype== "O":
        print(x," is a object")
    else :
        print(x," is a numerical data")

#Changing the type as dt as date format
print(type(df.dt[0]))
df['dt'] = df['dt'].apply(lambda x: datetime.datetime.strptime(x, "%d-%m-%Y"))
print(type(df.dt[0]))

# Converting the object columns into numerical : Label Encoding
label_convert = LabelEncoder()
column_to_encode = ["dt",'City',"Country",'Latitude','Longitude']

for col in column_to_encode :
    df[col] = label_convert.fit_transform(df[col])
                                          
print(df["Country"].loc[:10])

# EDA : plotting latitude and temperature to check their co - relation :
x_to_plot = df["Latitude"]
y_to_plot = df["AverageTemperature"]

plt.plot(x_to_plot[:140],y_to_plot[:140],marker="*",color="red",label="Temperature variation with latitude")
plt.xlabel("Latitude")
plt.ylabel("AverageTemperature")
plt.legend()
plt.show()


#Unsupervised Learning : Dimensionality reduction -- PCA
X = df[["Country","City","Latitude","Longitude"]]
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Variance ratio : ", pca.explained_variance_ratio_)
print("Total variance captured:", sum(pca.explained_variance_ratio_))

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', edgecolor='k', alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection')
plt.grid(True)
plt.show()

#Detecting anomalies in the Average temperature :

def detect_anomaly_data(data, threshold = 3) :
    z_scores = np.abs(stats.zscore(data))
    anomalies = z_scores > threshold

    return z_scores,anomalies

data = df["AverageTemperature"]
z_scores, anomalies = detect_anomaly_data(data,threshold = 3)

print("Total data points : ", len(data))
print("No of Anomalies :", np.sum(anomalies))
print(f"Anomalies Percentage: {np.sum(anomalies)/len(data)*100:.2f}%")

anomalies_values = data[anomalies]
z_score_anomalies = z_scores[anomalies]

for i,(values, z_scores) in enumerate(zip(anomalies_values,z_score_anomalies)) :
    print(f"Anomalies :{values:.2f}")
    print(f"z_scores : {z_scores:.2f}")
    

#Linear  Regression - Supervised Learning Algorithm :
x = df[["dt","City", "Country", "Latitude", "Longitude"]]
y = df["AverageTemperature"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=40)
model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print(y_pred)

print("\nAccuracy Score:")
accuracy_r2 = r2_score(y_test,y_pred)
print("R2 accuracy score :", accuracy_r2)
accuracy_mae = mean_absolute_error(y_test,y_pred)
print("MAE score :",accuracy_mae)
accuracy_mse = mean_squared_error(y_test,y_pred)
print("MSE score :",accuracy_mse)

#As we had a poor accuracy score in Linear Regression as it is a non-linear model, handling with decision tree model.
dt_model = DecisionTreeRegressor()
dt_x = df[["dt","City", "Country", "Latitude", "Longitude"]]
dt_y = df["AverageTemperature"]

dt_x_train, dt_x_test, dt_y_train, dt_y_test = train_test_split(dt_x, dt_y, test_size=0.2, random_state=42)
dt_model.fit(dt_x_train,dt_y_train)

y_prediction = dt_model.predict(dt_x_test)
accuracy_scoring = mean_squared_error(dt_y_test,y_prediction)
print("accuracy_scoring:",accuracy_scoring)

#cross validation for Decision tree model
cross_validation = cross_val_score(dt_model,dt_x,dt_y,cv=5,scoring=None)
print(cross_validation)

#Implementing k-means clustering to cluster average temperature based on city & country :
X_cluster = df[["City","Country"]]
Y_cluster = df["AverageTemperature"]

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_cluster)

labels = kmeans.labels_
print("Labels :",labels)

plt.figure(figsize=(8,6))
plt.scatter(X_cluster.iloc[:,0],X_cluster.iloc[:,1],c=labels,cmap="rainbow",marker="*")
plt.show()








