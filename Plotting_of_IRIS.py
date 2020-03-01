
import pandas as pd 
import matplotlib.pyplot as plt 

data = pd.read_csv("Iris.csv")  ## Loading Iris dataset as dataframe

#=======================================================

setosa=data[data['Species']=='Iris-setosa']           ## Data points which has Setosa as labels
versicolor=data[data['Species']=='Iris-versicolor']## Data points which has Versicolor as labels
virginica =data[data['Species']=='Iris-virginica'] ## Data points which has Verginica as labels

plt.figure()             
fig,ax=plt.subplots(1,1)  ## to plot all three labels in ax(same plot)

setosa.plot(x="SepalLengthCm", y="SepalWidthCm", kind="scatter",ax=ax,label='setosa',color='r')
versicolor.plot(x="SepalLengthCm",y="SepalWidthCm",kind="scatter",ax=ax,label='versicolor',color='b')
virginica.plot(x="SepalLengthCm", y="SepalWidthCm", kind="scatter", ax=ax, label='virginica', color='g')

plt.figure()
fig,ax=plt.subplots(1,1)  ## Search how plt.subplots works for more understanding

setosa.plot(x="PetalLengthCm", y="PetalWidthCm", kind="scatter",ax=ax,label='setosa',color='r')
versicolor.plot(x="PetalLengthCm",y="PetalWidthCm",kind="scatter",ax=ax,label='versicolor',color='b')
virginica.plot(x="PetalLengthCm", y="PetalWidthCm", kind="scatter", ax=ax, label='virginica', color='g')

