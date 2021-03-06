from gurobipy import Model, quicksum, GRB
import numpy as np
import streamlit as st

class model:
    def __init__(self, C_ij):
        self.C_ij = C_ij
        self.D_ijd = None
        self.I = [int(i.split()[-1]) for i in self.C_ij.columns]
        self.J = [i for i in self.C_ij.index]
        self.D_i = {1:[1],
                   2:[1, 2, 3, 4],
                   3:[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   4:[1, 2],
                   5:[1, 2, 3, 4]}

    def solve(self, D_ijd=None):
        # Adaptamos los datos
        df_C_ij = self.C_ij.copy()
        df_C_ij.columns = self.I
        C_ij = {}
        for i in self.I:
            for j in self.J:
                C_ij[(i, j)] = df_C_ij[i][j]

        # Construimos el modelo (sin descuentos)
        m = Model()

        # Variables
        X = {}
        for i in self.I:
          for j in self.J:
            X[i, j] = m.addVar(vtype=GRB.BINARY)

        Y = {}
        for i in self.I:
          for j in self.J:
            for d in self.D_i[i]:
              Y[i, j, d] = m.addVar(vtype=GRB.BINARY)

        obj = m.addVar(vtype=GRB.CONTINUOUS)

        # -------------- Restriccioes generales -------------------
        #1) Una empresa no puede ser auditada dos veces
        for j in self.J:
          m.addConstr(quicksum(X[i, j] for i in self.I) <= 1)

        #2) Auditor 1 inhabilitado para las empresas 2, 3, 8.
        if 1 in self.I and 2 in self.J: m.addConstr(X[1, 2] == 0)
        if 1 in self.I and 3 in self.J: m.addConstr(X[1, 3] == 0)
        if 1 in self.I and 8 in self.J: m.addConstr(X[1, 8] == 0)

        #3) Satisfacer demanda
        m.addConstr(quicksum(X[i, j] for i in self.I for j in self.J) == len(self.J))

        #4) Que todos los auditores ganen al menos 1 licitacion.
        for i in self.I:
          m.addConstr(quicksum(X[i, j] for j in self.J) >= 1)

        #5) Que ningun auditor audite mas de 5 auditorias.
        for i in self.I:
          m.addConstr(quicksum(X[i, j] for j in self.J) <= 5)

        #6) Restriccion auditor EEFF
        if 1 in self.I and 7 in self.J: m.addConstr(X[1, 7] == 0)
        if 3 in self.I and 2 in self.J: m.addConstr(X[3, 2] == 0)
        if 3 in self.I and 5 in self.J: m.addConstr(X[3, 5] == 0)
        if 4 in self.I and 11 in self.J: m.addConstr(X[4, 11] == 0)

        # -------- A??adimos las restricciones de los descuentos------------------------
        self.D_ijd = D_ijd
        if self.D_ijd != None:

            # Restricciones para los descuentos
            #7) Solo se podra aplicar los descuentos donde correspondan. Dicho de otra manera, no puedo aplicar un descuento sobre una auditoria que no he asignado.
            for i in self.I:
              for j in self.J:
                for d in self.D_i[i]:
                  m.addConstr(Y[i, j, d] <= X[i, j])

            #8) No puedo acumular descuentos.
            for i in self.I:
              for j in self.J:
                for d1 in self.D_i[i]:
                  for d2 in self.D_i[i]:
                    if d1 != d2 and len(self.D_i[i]) > 1:
                      m.addConstr(Y[i, j, d1] <= (1 - Y[i, j, d2]))

            #9) Descuentos auditor 1 (presenta una unica opcion de descuento)
            if 1 in self.I:
                for j in self.J:
                  m.addConstr(Y[1, j, 1] <= (quicksum(X[1, j] for j in self.J)/7))

            #10) Descuentos auditor 2 (presenta 3 opciones de descuentos).
            if 2 in self.I:
                for j in self.J:
                  m.addConstr(Y[2, j, 2] <= (quicksum(C_ij[2, j]*X[2, j] for j in self.J)/760))

                for j in self.J:
                  m.addConstr(Y[2, j, 3] <= (quicksum(C_ij[2, j]*X[2, j] for j in self.J)/1400))

                for j in self.J:
                  m.addConstr(Y[2, j, 4] <= (quicksum(C_ij[2, j]*X[2, j] for j in self.J)/1950))

            #11) Descuentos auditor 3 (presenta 10 opciones de descuentos)
            if 3 in self.I:
                for d in self.D_i[3]:
                  for j in self.J:
                    m.addConstr(Y[3, j, d] <= (quicksum(X[3, j] for j in self.J)/(d+1)))

            #12) Descuentos auditor 4 (presenta 2 opciones de descuentos).
            if 4 in self.I:
                for j in self.J:
                  m.addConstr(Y[4, j, 1] <= (quicksum(X[4, j] for j in self.J)/2))

                for j in self.J:
                  m.addConstr(Y[4, j, 2] <= (quicksum(X[4, j] for j in self.J)/6))

            #13) Descuentos auditor 5 (presenta 4 opciones de descuentos)
            if 5 in self.I:
                for j in self.J:
                  m.addConstr(Y[5, j, 1] <= (quicksum(C_ij[5, j]*X[5, j] for j in self.J)/1801))

                for j in self.J:
                  m.addConstr(Y[5, j, 2] <= (quicksum(C_ij[5, j]*X[5, j] for j in self.J)/3601))

                for j in self.J:
                  m.addConstr(Y[5, j, 3] <= (quicksum(C_ij[5, j]*X[5, j] for j in self.J)/5401))

                for j in self.J:
                  m.addConstr(Y[5, j, 4] <= (quicksum(C_ij[5, j]*X[5, j] for j in self.J)/7001))

            #14) Nota: monto maximo de 450 para el auditor 2.
            if 2 in self.I:
                m.addConstr(quicksum(self.D_ijd[2, j, d]*Y[2, j, d] for j in self.J for d in self.D_i[2]) <= 450)

            # Funcion Objetivo
            m.addConstr(obj == quicksum(C_ij[i, j]*X[i, j] for i in self.I for j in self.J) -
                       quicksum(self.D_ijd[i, j, d]*Y[i, j, d] for i in self.I for j in self.J for d in self.D_i[i])
                       )
        else:
            # Funcion Objetivo
            m.addConstr(obj == quicksum(C_ij[i, j]*X[i, j] for i in self.I for j in self.J))

        # Resolver el modelo
        m.setObjective(obj, GRB.MINIMIZE)
        m.optimize()

        Z_sol = obj.x
        X_sol = {(i, j):C_ij[i, j] for i in self.I for j in self.J if X[i, j].x > 0}
        Y_sol = {(i, j, d):self.D_ijd[i, j, d] for i in self.I for j in self.J for d in self.D_i[i] if Y[i, j, d].x > 0}
        Desc_total = np.sum([Y_sol[i] for i in Y_sol])

        return Z_sol, X_sol, Y_sol, Desc_total
