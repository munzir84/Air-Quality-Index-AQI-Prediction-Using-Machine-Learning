#from livelossplot import PlotLossesKeras
# import module from tkinter for UI
from matplotlib.pyplot import *
    # from array import *
from numpy import *
from tkinter import *
import os
root = Tk()
img_size = 48
batch_size = 64
def datapreprocessing():
    os.system("python dp.py")
def dttraining():
    os.system("python decisiontree.py")
def rftraining():
    os.system("python randomforest.py")
def lrtraining():
    os.system("python linearreg.py")
def plotacc():
    os.system("python acc.py")
def function6():
    root.destroy()
def appopen():
    os.system("python web.py")
root.configure(background="white")
root.title("Air Quality Index Prediction using Machine Learning")
# creating a text label
Label(root, text="Air Quality Index Prediction using Machine Learning", font=("times new roman", 20), fg="white", bg="#1A3C40",
      height=2).grid(row=0, columnspan=2, sticky=N + E + W + S, padx=75, pady=15)
Button(root, text="Data Preprocessing", font=('times new roman', 20), bg="#EDE6DB", fg="#3e2723", command=datapreprocessing).grid(
    row=1, columnspan=2, sticky=N + E + W + S,padx=75, pady=15)
Button(root, text="Linear Regression Model Training", font=('times new roman', 20), bg="#EDE6DB", fg="#3e2723", command=lrtraining).grid(
    row=2, columnspan=2, sticky=N + E + W + S, padx=75, pady=15)
Button(root, text="Decision Tree Model Training", font=('times new roman', 20), bg="#EDE6DB", fg="#3e2723", command=dttraining).grid(
    row=3, columnspan=2, sticky=N + E + W + S, padx=75, pady=15)
Button(root, text="Random Forest Model Training", font=('times new roman', 20), bg="#EDE6DB", fg="#3e2723", command=rftraining).grid(
    row=4, columnspan=2, sticky=N + E + W + S, padx=75, pady=15)
Button(root, text="Model Accuracy Comparison", font=('times new roman', 20), bg="#EDE6DB", fg="#3e2723", command=plotacc).grid(
    row=5, columnspan=2, sticky=N + E + W + S, padx=75, pady=15)
# creating second button
Button(root, text="Air Quality Index Prediction Web App", font=('times new roman', 20), bg="#EDE6DB", fg="#3e2723", command=appopen).grid(
    row=6, columnspan=2, sticky=N + E + W + S, padx=75, pady=15)
Button(root, text="Exit", font=('times new roman', 20), bg="#EDE6DB", fg="#3e2723", command=function6).grid(
    row=7, columnspan=2, sticky=N + E + W + S, padx=75, pady=15)
root.mainloop()
