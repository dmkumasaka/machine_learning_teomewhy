# %%
import pandas as pd

df = pd.read_excel("data/dados_cerveja.xlsx")
df

# %%
from sklearn import tree

arvore = tree.DecisionTreeClassifier()

# %%
caracteristicas = ['temperatura','copo','espuma','cor']

y = df['classe']
x = df[caracteristicas]

# %%

x = x.replace({
    'mud': 1, 'pint': 2,
    'sim': 1, 'não': 0,
    'escura': 1, 'clara': 0,
})

# %%
arvore.fit(x, y)

# %%
arvore.predict([[0,1,0,0]])

# %%
import matplotlib.pyplot as plt

plt.figure(dpi=1080)

tree.plot_tree(arvore,
               feature_names=caracteristicas,
               class_names=arvore.classes_,
               filled=True)


# %%
