# %%
import pandas as pd

df = pd.read_parquet("data/dados_clones.parquet")
df.head()

# %%
from sklearn import tree

arvore = tree.DecisionTreeClassifier()

# %%

caracteristicas = ['Massa(em kilos)','Estatura(cm)']

y = df['Status ']
x = df[caracteristicas]

x = x.replace({
    'Tipo 1': 1, 'Tipo 2': 2,
    'Tipo 3': 3, 'Tipo 4': 4,
    'Tipo 5': 5, 'Defeituoso': 1,
    'Apto': 0,
})

# %%
arvore.fit(x, y)

# %%
import matplotlib.pyplot as plt

tree.plot_tree(arvore,
               feature_names=caracteristicas,
               class_names=arvore.classes_,
               filled=True,
               max_depth=4)

# %%
