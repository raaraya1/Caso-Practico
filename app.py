import streamlit as st
import pandas as pd
import numpy as np
from model import model
import matplotlib.pyplot as plt


st.write('''
    ## **PROBLEMA DE ASIGNACIÓN DE AUDITORES**

    ## **Contexto**

    El problema surge de la necesidad de buscar la asignación optima
    entre los precios que ofrecen los auditores para auditar un conjunto
    de empresas. Así, se espera que la solución al problema contemple una
    asignación en donde cada empresa tiene asignado un auditor de forma de
    reducir los costos totales del problema.

    ### **Datos**

''')

# Cargar datos
st.sidebar.write('**Datos**')
file_cost_matrix = st.sidebar.file_uploader('Selecciona el archivo con la matriz de costos (.csv)')

# Cargar descuentos
modelo_con_descuentos = st.sidebar.checkbox(label='Modelo con descuentos', value=False)
file_discounts = None
if modelo_con_descuentos:
    file_discounts = st.sidebar.file_uploader('Seleccione el archivo con los descuentos (.csv)')
    if file_discounts != None:
        df_discounts = pd.read_csv(file_discounts, sep=';')
        df_disc = df_discounts[['Auditor', 'Empresa', 'Opcion', 'Descuento']].copy()
        D_ijd = {(df_disc['Auditor'][i], df_disc['Empresa'][i], df_disc['Opcion'][i]):float(df_disc['Descuento'][i]) for i in df_disc.index}


# Cargar matriz de costos
if file_cost_matrix:
    with st.expander('Datos cargados'):
        df_cost_matrix = pd.read_csv(file_cost_matrix, sep=';')
        st.dataframe(df_cost_matrix)

    # Grafico de Datos
    with st.expander('Visualización de costos'):
        auditores = df_cost_matrix.columns[1:]
        aud_selct = st.multiselect('Auditores', options=auditores, default=[i for i in auditores])
        df_cost_matrix.index = [i+1 for i in range(len(df_cost_matrix))]
        empresas = df_cost_matrix.index
        emp_selct = st.multiselect('Empresas', options=empresas, default=[i for i in empresas])

        # Ahora cambiamos arreglamos el DataFrame
        df_new = df_cost_matrix[aud_selct]
        df_new = df_new.loc[emp_selct]

        # Generamos el grafico
        if aud_selct != [] and emp_selct != []:
            fig = df_new.plot(kind='bar', title='Costos de auditores por empresa', xlabel='Empresas', ylabel='Valores').get_figure()
            st.pyplot(fig)

        # En caso de cargar el archivo de descuentos
        c1, c2, c3 = st.columns([3, 10, 1])
        c2.write('''
        #### **RESUMEN DE COSTOS CON DESCUENTOS**
        ''')

        if file_discounts != None:
            D_1jd = {'sin desc':[], 'desc 1':[]}
            D_2jd = {'sin desc':[], 'desc 1':[], 'desc 2':[], 'desc 3':[], 'desc 4':[]}
            D_3jd = {'sin desc':[], 'desc 1':[], 'desc 2':[], 'desc 3':[], 'desc 4':[], 'desc 5':[], 'desc 6':[], 'desc 7':[], 'desc 8':[], 'desc 9':[], 'desc 10':[]}
            D_4jd = {'sin desc':[], 'desc 1':[], 'desc 2':[]}
            D_5jd = {'sin desc':[], 'desc 1':[], 'desc 2':[], 'desc 3':[], 'desc 4':[]}

            # Casos sin descuentos
            for i in df_cost_matrix.index:
                D_1jd['sin desc'].append(df_cost_matrix['Auditor 1'][i])
                D_2jd['sin desc'].append(df_cost_matrix['Auditor 2'][i])
                D_3jd['sin desc'].append(df_cost_matrix['Auditor 3'][i])
                D_4jd['sin desc'].append(df_cost_matrix['Auditor 4'][i])
                D_5jd['sin desc'].append(df_cost_matrix['Auditor 5'][i])

            for i in D_ijd:
                # Descuentos auditor 1
                if i[0] == 1:
                ## Descuentos auditor 1 opcion 1
                    if i[2] == 1: D_1jd['desc 1'].append(D_1jd['sin desc'][i[1]-1] - D_ijd[i])

                # Descuentos auditor 2
                elif i[0] == 2:
                    if i[2] == 1: D_2jd['desc 1'].append(D_2jd['sin desc'][i[1]-1] - D_ijd[i])
                    elif i[2] == 2: D_2jd['desc 2'].append(D_2jd['sin desc'][i[1]-1] - D_ijd[i])
                    elif i[2] == 3: D_2jd['desc 3'].append(D_2jd['sin desc'][i[1]-1] - D_ijd[i])
                    elif i[2] == 4: D_2jd['desc 4'].append(D_2jd['sin desc'][i[1]-1] - D_ijd[i])

                # Descuentos auditor 3
                elif i[0] == 3:
                    if i[2] == 1: D_3jd['desc 1'].append(D_3jd['sin desc'][i[1]-1] - D_ijd[i])
                    elif i[2] == 2: D_3jd['desc 2'].append(D_3jd['sin desc'][i[1]-1] - D_ijd[i])
                    elif i[2] == 3: D_3jd['desc 3'].append(D_3jd['sin desc'][i[1]-1] - D_ijd[i])
                    elif i[2] == 4: D_3jd['desc 4'].append(D_3jd['sin desc'][i[1]-1] - D_ijd[i])
                    elif i[2] == 5: D_3jd['desc 5'].append(D_3jd['sin desc'][i[1]-1] - D_ijd[i])
                    elif i[2] == 6: D_3jd['desc 6'].append(D_3jd['sin desc'][i[1]-1] - D_ijd[i])
                    elif i[2] == 7: D_3jd['desc 7'].append(D_3jd['sin desc'][i[1]-1] - D_ijd[i])
                    elif i[2] == 8: D_3jd['desc 8'].append(D_3jd['sin desc'][i[1]-1] - D_ijd[i])
                    elif i[2] == 9: D_3jd['desc 9'].append(D_3jd['sin desc'][i[1]-1] - D_ijd[i])
                    elif i[2] == 10: D_3jd['desc 10'].append(D_3jd['sin desc'][i[1]-1] - D_ijd[i])

                # Descuentos auditor 4
                elif i[0] == 4:
                    if i[2] == 1: D_4jd['desc 1'].append(D_4jd['sin desc'][i[1]-1] - D_ijd[i])
                    elif i[2] == 2: D_4jd['desc 2'].append(D_4jd['sin desc'][i[1]-1] - D_ijd[i])

                # Descuentos auditor 5
                elif i[0] == 5:
                    if i[2] == 1: D_5jd['desc 1'].append(D_5jd['sin desc'][i[1]-1] - D_ijd[i])
                    elif i[2] == 2: D_5jd['desc 2'].append(D_5jd['sin desc'][i[1]-1] - D_ijd[i])
                    elif i[2] == 3: D_5jd['desc 3'].append(D_5jd['sin desc'][i[1]-1] - D_ijd[i])
                    elif i[2] == 4: D_5jd['desc 4'].append(D_5jd['sin desc'][i[1]-1] - D_ijd[i])

            # Grafico completo
            df1 = pd.DataFrame.from_dict(D_1jd)
            df2 = pd.DataFrame.from_dict(D_2jd)
            df3 = pd.DataFrame.from_dict(D_3jd)
            df4 = pd.DataFrame.from_dict(D_4jd)
            df5 = pd.DataFrame.from_dict(D_5jd)

            df1.index = [i+1 for i in range(len(df_cost_matrix))]
            df2.index = [i+1 for i in range(len(df_cost_matrix))]
            df3.index = [i+1 for i in range(len(df_cost_matrix))]
            df4.index = [i+1 for i in range(len(df_cost_matrix))]
            df5.index = [i+1 for i in range(len(df_cost_matrix))]

            # Filtramos empresas
            df1_n = df1.loc[emp_selct]
            df2_n = df2.loc[emp_selct]
            df3_n = df3.loc[emp_selct]
            df4_n = df4.loc[emp_selct]
            df5_n = df5.loc[emp_selct]

            fig, axes = plt.subplots(nrows=1, ncols=len(aud_selct), sharey=True, figsize=(20,10)) # figsize=(ancho, altura)
            i = 0
            if 'Auditor 1' in aud_selct:
                df1_n.plot(ax=axes[i], kind='bar', title='Auditor 1')
                i += 1
            if 'Auditor 2' in aud_selct:
                df2_n.plot(ax=axes[i], kind='bar', title='Auditor 2')
                i += 1
            if 'Auditor 3' in aud_selct:
                df3_n.plot(ax=axes[i], kind='bar', title='Auditor 3')
                i += 1
            if 'Auditor 4' in aud_selct:
                df4_n.plot(ax=axes[i], kind='bar', title='Auditor 4')
                i += 1
            if 'Auditor 5' in aud_selct: df5_n.plot(ax=axes[i], kind='bar', title='Auditor 5')

            st.pyplot(fig)


    st.write('''
    ### **Modelo**
    ''')
    with st.expander('Explicación del Modelo'):
        st.write(r'''
**Conjuntos**

$i \in I$ : Auditores

$j \in J$ : Empresas

$d \in D_{i}$: Descuentos variables que propone el auditor i

**Parámetros**

$C_{ij}$: Costo base por el cual el auditor i audita empresa j.

$D_{ijd}$: Valor del descuento (opción d) que se aplica a auditoria de empresa j realizada por auditor i.

**Variables**

$X_{ij} \in (0, 1)$: Si auditor i audita empresa j.

$Y_{ijd} \in (0, 1)$: Si el auditor i aplica descuento d en empresa j.

**Función Objetivo**

$$
min \sum_{i} \sum_{j} C_{ij} X_{ij} - \sum_{i} \sum_{j} \sum_{d} D_{ijd} Y_{ijd}
$$

**Restricciones generales**

1) Una empresa no puede ser auditada dos veces.

$$
\sum_{i} X_{ij} \leq 1 \quad j \in J
$$

2) No se presentan los precios del auditor 1 para las empresas 2, 3, 8. Se interpretan como que el auditor 1 se encuentra inhabilitado para auditar aquellas empresas.

$$
X_{12} = 0
$$

$$
X_{13} = 0
$$

$$
X_{18} = 0
$$

3) Satisfacer demanda

$$
\sum_{i} \sum_{j} X_{ij} = 11
$$

4) Que todos los auditores ganen al menos 1 licitación.

$$
\sum_{j} X_{ij} \geq 1 \quad \forall i \in I
$$

5) Que ningún auditor audite más de 5 auditorias.

$$
\sum_{j} X_{ij} \leq 5 \quad \forall i \in I
$$

6) Restricción auditor EEFF

$$
X_{17} = 0
$$

$$
X_{32} = 0
$$

$$
X_{35} = 0
$$

$$
X_{411} = 0
$$

**Restricciones de los descuentos**

7) Solo se podrá aplicar los descuentos donde correspondan. Dicho de otra manera, no puedo aplicar un descuento sobre una auditoria que no he asignado.

$$
Y_{ijd} \leq X_{ij} \quad \forall i \in I, j \in J, d \in D_{i}
$$

8) No puedo acumular descuentos.

$$
Y_{ijd} \leq (1 - Y_{ijd'}) \quad \forall i \in I, j \in J, d \neq d' \in D_{i}
$$

9) Descuentos auditor 1 (presenta una única opción de descuento)

$$
Y_{1j1} \leq \frac{\sum_{j} X{1j}}{7} \quad \forall j \in J
$$

10) Descuentos auditor 2 (presenta 3 opciones de descuentos).

$$
Y_{2j2} \leq \frac{\sum_{j}C_{2j}X_{2j}}{760} \quad \forall j \in J
$$

$$
Y_{2j3} \leq \frac{\sum_{j}C_{2j}X_{2j}}{1400} \quad \forall j \in J
$$

$$
Y_{2j4} \leq \frac{\sum_{j}C_{2j}X_{2j}}{1950} \quad \forall j \in J
$$

11) Descuentos auditor 3 (presenta 10 opciones de descuentos)

$$
Y_{3j1} \leq \frac{\sum_{j} X{3j}}{2} \quad \forall j \in J
$$

$$
Y_{3j2} \leq \frac{\sum_{j} X{3j}}{3} \quad \forall j \in J
$$

$$
Y_{3j3} \leq \frac{\sum_{j} X{3j}}{4} \quad \forall j \in J
$$

$$
Y_{3j4} \leq \frac{\sum_{j} X{3j}}{5} \quad \forall j \in J
$$

$$
Y_{3j5} \leq \frac{\sum_{j} X{3j}}{6} \quad \forall j \in J
$$

$$
Y_{3j6} \leq \frac{\sum_{j} X{3j}}{7} \quad \forall j \in J
$$

$$
Y_{3j7} \leq \frac{\sum_{j} X{3j}}{8} \quad \forall j \in J
$$

$$
Y_{3j8} \leq \frac{\sum_{j} X{3j}}{9} \quad \forall j \in J
$$

$$
Y_{3j9} \leq \frac{\sum_{j} X{3j}}{10} \quad \forall j \in J
$$

$$
Y_{3j10} \leq \frac{\sum_{j} X{3j}}{11} \quad \forall j \in J
$$

12) Descuentos auditor 4 (presenta 4 opciones de descuentos).

$$
Y_{4j1} \leq \frac{\sum_{j}X_{4j}}{3} \quad \forall j \in J
$$

$$
Y_{4j2} \leq \frac{\sum_{j}X_{4j}}{6} \quad \forall j \in J
$$

$$
Y_{4j3} \leq \frac{\sum_{j}X_{4j}}{8} \quad \forall j \in J
$$

13) Descuentos auditor 5 (presenta 4 opciones de descuentos)

$$
Y_{5j1} \leq \frac{\sum_{j} C_{5j}X_{5j}}{1801} \quad \forall j \in J
$$

$$
Y_{5j2} \leq \frac{\sum_{j} C_{5j}X_{5j}}{3601} \quad \forall j \in J
$$

$$
Y_{5j3} \leq \frac{\sum_{j} C_{5j}X_{5j}}{5401} \quad \forall j \in J
$$

$$
Y_{5j4} \leq \frac{\sum_{j} C_{5j}X_{5j}}{7001} \quad \forall j \in J
$$

14) Nota: monto máximo de 450 para el auditor 2.

$$
\sum_{j} \sum_{d \in D_2} D_{2jd}Y_{2jd} \leq 450
$$

        ''')


    with st.expander('Modelo Programado'):
        with open('modelo.txt', mode='r') as body:
            st.code(body.read(), language="python")

    st.write('''
    ### Resultados
    ''')

    # Funcion para mezclar los dataframes (solo para el caso de descuentos)
    def _mix_df(df_x, Y):
        df_mix = df_x.copy()
        df_mix['Opcion'] = 0
        df_mix['Descuento'] = 0
        for i in df_mix.index:
            aud = df_mix['Auditor'][i]
            emp = df_mix['Empresa'][i]
            opc = 0
            des = 0
            # Buscar en Y
            for j in Y:
                if j[0] == aud and j[1] == emp:
                    opc = j[2]
                    des = Y[j]
            df_mix['Opcion'][i] = opc
            df_mix['Descuento'][i] = des
        return df_mix

    with st.expander('Resultados'):
        if file_discounts != None:
            modelo = model(C_ij=df_new)
            Z,X,Y,DT = modelo.solve(D_ijd=D_ijd)
            st.write('''
            **Asignaciones**
            ''')
            col1, col2, col3 = st.columns([10, 1, 10])
            df_x = pd.DataFrame([[i[0], i[1], X[i]] for i in X])
            df_x.columns=['Auditor', 'Empresa', 'Costo']
            col1.dataframe(df_x)

            cost_sin_desc = int(np.sum([i for i in df_x['Costo']]))
            col3.metric(label='Costo sin descuentos', value=cost_sin_desc)

            st.write('''
            **Descuentos**
            ''')
            col1, col2, col3 = st.columns([14, 1, 6])
            df_y = pd.DataFrame([[i[0], i[1], i[2], int(Y[i])] for i in Y])
            df_y.columns=['Auditor', 'Empresa', 'Opcion', 'Descuento']
            col1.dataframe(df_y)
            col3.metric(label='Descuento realizado', value=int(DT))

            st.write('''
            **Total**
            ''')
            c1, c2, c3 = st.columns(3)
            delta = np.round((1 - (float(cost_sin_desc)/float(Z)))*100, 0)
            c2.metric(label='Costo Total', value=int(Z), delta=f'{delta}%')

            # Mezclar ambos dataframes
            df_mix = _mix_df(df_x, Y)

            # Descargar la solucion
            c2.download_button(label='Descargar',
                                       data=df_mix.to_csv(sep=';', index=False),
                                       file_name='Solution.csv',
                                       mime='text/csv')

        else:
            modelo = model(C_ij=df_new)
            Z,X,Y,DT = modelo.solve()
            st.write('''
            **Asignaciones**
            ''')
            col1, col2, col3 = st.columns([10, 1, 10])
            df_x = pd.DataFrame([[i[0], i[1], X[i]] for i in X])
            df_x.columns=['Auditor', 'Empresa', 'Costo']
            col1.dataframe(df_x)

            cost_sin_desc = int(np.sum([i for i in df_x['Costo']]))
            col3.metric(label='Costo sin descuentos', value=cost_sin_desc)

            st.write('''
            **Total**
            ''')
            c1, c2, c3 = st.columns(3)
            delta = np.round((1 - (float(cost_sin_desc)/float(Z)))*100, 0)
            c2.metric(label='Costo Total', value=int(Z), delta=f'{delta}%')

            # Descargamos la solucion
            c2.download_button(label='Descargar',
                                       data=df_x.to_csv(sep=';', index=False),
                                       file_name='Solution.csv',
                                       mime='text/csv')
