# para dividir la data en 2 para evitar el overfitting

import numpy as np
from sklearn.model_selection import train_test_split

a = np.arange(1, 101)
b = np.arange(501, 600)

train_test_split(a) # x defecto divide 75/25 y lo hace al azar
#ya lo podemos guardar en 2 variables distintas

a_train, a_test = train_test_split(a, test_size=0.25, shuffle=True) # asi seria x defecto
a_train, a_test = train_test_split(a, test_size=0.2, shuffle=False) # asi cambiamos la proporcion y lo hacemos q no sea al azar
a_train, a_test = train_test_split(a, test_size=0.2, random_state=42) #asi hacemos que sea al azar, pero siempre igual

a_train, a_test, b_train, b_test = train_test_split(a,b, test_size=0.2, random_state=42) # podemos hacer eso mismo para
# dos variables a la vez, pero es importante q al randomizarlas, las haga juntas, porq sino cuando hacemos la regresion, manda cualquiera


