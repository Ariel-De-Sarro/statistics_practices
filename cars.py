import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


raw_data = pd.read_csv('1.04. Real-life example.csv')
raw_data.head()
raw_data.describe(include = 'all') # por defecto, solo pone las variables numericas

data = raw_data.drop(['Model'], axis = 1) #queremos sacar la columna de modelo, q esta sobre el eje horizontal q es el 1
data.describe(include='all')

# segun la data, hay obvervaciones que tienen variables vacias, y hay q eliminarlas
print(data.isnull().sum()) # aca me da la suma de observaciones con variables vacias, xq null es True, entonces es 1, y no nulas es 0

# por regla general, si vamos a eliminar menos del 5% de las observaciones, las podemos eliminar todas

data_no_mv = data.dropna(axis=0) #eliminamos las observaciones con valores vacios y la guardamos en una nueva variable,
# en este caso, el eje es 0, xq es el de las obvservaciones

print(data_no_mv.describe(include='all'))

#ahora queremos ver como se comporta cada variable, lo buscado es q sean distribuciones normales


#print(sns.distplot(data_no_mv['Price'])) # aca construimos el grafico de distribucion de la variable 'price'
#plt.show() # tenemos q poner eso para q nos muestre el grafico

# vemos en el grafico, y en la tabla, q hay valores muy altos (outliers), asi q los vamos a eliminar

q = data_no_mv['Price'].quantile(0.99) # eliminamos el 1% mas alto
data_1 = data_no_mv[data_no_mv['Price']<q] # hacemos otro dataframe, con los valores que contengan un valor de 'Price'
# menor a q
print(data_1.describe(include='all'))

q = data_1['Mileage'].quantile(0.99) # eliminamos el 1% mas alto
data_2 = data_1[data_1['Mileage']<q]

data_3 = data_2[data_2['EngineV']<6.5] # eliminamos valores de volumen superiores a 6.5 l

q = data_3['Year'].quantile(0.01) # eliminamos el 1% mas bajo
data_4 = data_3[data_1['Year']>q]

data_cleaned = data_4.reset_index(drop= 'True') #aca reseteamos los valores de indices, para q se repartan entre las
# observaciones que forman nuestra data limpia
print(data_cleaned.describe(include='all'))

# aca vemos los graficos de Price versus 3 variables, y ninguna tiene una respuesta lineal
#f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15,3))
#ax1.scatter(data_cleaned['Year'], data_cleaned['Price'])
#ax1.set_title('Price and Year')
#ax2.scatter(data_cleaned['EngineV'], data_cleaned['Price'])
#ax2.set_title('Price and EngineV')
#ax3.scatter(data_cleaned['Mileage'], data_cleaned['Price'])
#ax3.set_title('Price and Mileage')
# plt.show()

log_price = np.log(data_cleaned['Price']) # aca transformamos a la variable Precio logaritmicamente
data_cleaned['log_Price'] = log_price

#print(sns.distplot(data_cleaned['log_Price'])) # aca vemos el grafico de distibucion de log_price y si es normal ahora
#plt.show()

#ahora probamos hacer log_price vs 3 variables, y ahora si dan una respuesta lineal
#f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15,3))
#ax1.scatter(data_cleaned['Year'], data_cleaned['log_Price'])
#ax1.set_title('log_Price and Year')
#ax2.scatter(data_cleaned['EngineV'], data_cleaned['log_Price'])
#ax2.set_title('log_Price and EngineV')
#ax3.scatter(data_cleaned['Mileage'], data_cleaned['log_Price'])
#ax3.set_title('log_Price and Mileage')
#plt.show()

data_cleaned = data_cleaned.drop(['Price'], axis=1) # eliminamos la variable precio, xq ya no la necesitamos

#luego hay q chequear multicolinealidad

from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_cleaned[['Mileage', 'Year', 'EngineV']]
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['Features'] = variables.columns
print(vif)

# si el VIF es = 1, no hay nada de colinealidad. SI VIF entre 1 y 5, esta perfectamente bien. Pero el
# limite es arbitrario, algunos consideran 5, otros 6, o 7. En este caso, descartamos Year, ya q es la mas correlacionada
# con las otras variables

data_no_mc = data_cleaned.drop(['Year'], axis=1)

# luego tenemos que convertir las variables categoricas en dummies. Si tenemos N categorias para una variable, se crean
# N-1 dummies (xq la otra estaria creada automaticamente x descarte, si la inculyesemos, incluiriamos multicolinealidad)

data_with_dummies = pd.get_dummies(data_no_mc, drop_first=True) # para q no se creen dummies para todos, solo para N-1

#luego podemos reacomodar las columnas

print(data_with_dummies.columns.values)
# aca cambio el orden de las columnas que quiero
cols = ['log_Price', 'Mileage', 'EngineV', 'Brand_BMW', 'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
        'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other', 'Body_sedan', 'Body_vagon', 'Body_van',
        'Engine Type_Gas', 'Engine Type_Other', 'Engine Type_Petrol','Registration_yes']

data_preprocessed = data_with_dummies[cols]

#ahora ya estamos para hacer la regresion
#primero tenemos q definir el target y los imputs

targets = data_preprocessed['log_Price']
data_no_log_Price = data_preprocessed.drop(['log_Price'], axis=1) #todos menos log_Price
inputs = data_no_log_Price

#luego tenemos que escalar

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(inputs)
input_scaled = scaler.transform(inputs)

#luego tenemos q dividir la data en train y test

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(input_scaled, targets, test_size=0.2, random_state=365)

# ahora creamos la regresion
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)

# una forma de chequear el resultado es hacer un grafico de y_train y y_predecido

y_hat = reg.predict(x_train) # y predecido, o y sombrero

#plt.scatter(y_train, y_hat)
plt.show()

# otra forma de comprobar es hacer el grafico de distribucion de los residuales (errores, o diferencias entre los
# tagets y los predecidos)

sns.distplot(y_train - y_hat)
plt.show()

#calculamos el r2
reg.score(x_train, y_train)

# calculamos los weights y el bias (coeficientes y ordenada al origen)

reg.intercept_ #bias

reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
print(reg_summary)

# viendo los datos, podemos decir que los weights positivos muestran q al aumentar esas caracteristicas hacen aumentar
# el log_Price y el price. Los negativos muestan q al aumentar, hacen disminuir el precio y el log.
# con respecto a los Dummies, un weight positivo quiere decir que esa categoria es mas cara que la de referencia (Audi)
# y viceversa

# para ver cual es el valor de referencia para categoria de dummie, hacer
# data_cleaned['nombre de categoria'].unique() y ver la que falta
# si se quiere cambiar esto, antes de hacer los dummies, reordenarlas

#ahora hay q testear
y_hat_test = reg.predict(x_test)
plt.scatter(y_test, y_hat_test, alpha=0.2) # el alpha hace tipo un mapa de calor
plt.show()

# ahora hacemos un dataframe para ver los resultados y analizar

df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Predictions']) # hacemos la exponencial de los datos, para volver a Precios
y_test = y_test.reset_index(drop=True) # tenemos q resetear los indices, xq sino no se pueden aparear

df_pf['Target'] = np.exp(y_test) # aca le agregamos la columna al df para comparar los datos predecidos con los de la data
df_pf['Residual'] = df_pf['Target'] - df_pf['Preductions'] # aca hacemos otra columna para ver la diferencia
df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100) # aca hacemos otra columna para ver la diferencia en % y absoluta

pd.options.display.max_rows =999 #esto para que muestre todas las filas
pd.set_option('display.float.format', lambda x: '%.2f' % x) # esto para que solo muestre 2 digitos despues del 0
df_pf.sort_values(by=['Difference%']) # para que lo ordene por valores de Diferencia porcentual

# asi podemos ver los datos manualmente y ver como se comporto el modelo
# en este caso al ver los valores de mas diferencia, se ve q son pocos.
# puede ser que estos hayan estado dañados, o modelos baratos, etc, ya que los predecidos son mas altos que los valores

