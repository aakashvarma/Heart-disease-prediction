import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_train = pd.read_csv("final_mitbih_train.csv")

y = df_train.iloc[1]
x = list(range(1,189))

plt.plot(x, y)
plt.show()










