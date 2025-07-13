# %%
import pandas as pd

df = pd.read_excel("data/dados_frutas.xlsx")
df

# %%
from sklearn import tree

arvore = tree.DecisionTreeClassifier()

# %%
y = df['Fruta']

caracteristicas = ['Arredondada','Suculenta','Vermelha','Doce']

x = df[caracteristicas]

# %%
arvore.fit(x, y)

# %%
arvore.predict([[0,1,0,0]])

# %%
import matplotlib.pyplot as plt

tree.plot_tree(arvore,
               feature_names=caracteristicas,
               class_names=arvore.classes_,
               filled=True)

# %%
