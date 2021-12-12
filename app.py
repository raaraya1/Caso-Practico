import streamlit as st
import pandas as pd
import numpy as np
from model import model


st.write('''
    ## **PROBLEMA DE ASIGNACIÓN DE AUDITORES**

    ## **Contexto**

    El problema surge de la necesidad de buscar la asignacion optima
    entre los precios que ofrecen los auditores para auditar un conjunto
    de empresas. Asi, se espera que la solucion al problema contemple una
    asignacion en donde cada empresa tiene asignado un auditor de forma de
    reducir los costos totales del problema.

    ### **Datos**
''')

# Archivos descargables
st.sidebar.write('**Archivos descargables**')
cost_matrix = pd.read_csv('https://raw.githubusercontent.com/raaraya1/Caso-Practico/main/cost_matrix.csv')
st.sidebar.download_button(label='Matriz de costos',
                           data=cost_matrix.to_csv(index=False),
                           file_name='Matrix de Costos.csv',
                           mime='text/csv')

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
    with st.expander('Visualizacion de costos'):
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

    st.write('''
    ### **Modelo**
    ''')
    with st.expander('Explicacion del Modelo'):
        st.write(r'''
**Conjuntos**

$i \in I$ : Auditores

$j \in J$ : Empresas

$d \in D_{i}$: Descuentos variables que propone el auditor i

**Parametros**

$C_{ij}$: Costo base por el cual el auditor i audita empresa j.

$D_{ijd}$: Valor del descuento (opcion d) que se aplica a auditoria de empresa j realizada por auditor i.

**Variables**

$X_{ij} \in (0, 1)$: Si auditor i audita empresa j.

$Y_{ijd} \in (0, 1)$: Si el auditor i aplica descuento d en empresa j.

**Funcion Objetivo**

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

4) Que todos los auditores ganen al menos 1 licitacion.

$$
\sum_{j} X_{ij} \geq 1 \quad \forall i \in I
$$

5) Que ningun auditor audite mas de 5 auditorias.

$$
\sum_{j} X_{ij} \leq 5 \quad \forall i \in I
$$

6) Restriccion auditor EEFF

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

7) Solo se podra aplicar los descuentos donde correspondan. Dicho de otra manera, no puedo aplicar un descuento sobre una auditoria que no he asignado.

$$
Y_{ijd} \leq X_{ij} \quad \forall i \in I, j \in J, d \in D_{i}
$$

8) No puedo acumular descuentos.

$$
Y_{ijd} \leq (1 - Y_{ijd'}) \quad \forall i \in I, j \in J, d \neq d' \in D_{i}
$$

9) Descuentos auditor 1 (presenta una unica opcion de descuento)

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

14) Nota: monto maximo de 450 para el auditor 2.

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
