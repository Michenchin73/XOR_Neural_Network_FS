import math
import random

entradas = [[0, 0], [0, 1], [1, 0], [1, 1]]
salida_real = [0, 1, 1, 0]


def sigmoid(x): # Funcion sigmoid. Mientras mas grande x mas se acerca a 1. Y mientras mas pequeño x mas se acerca a 0
    return 1/(1+math.exp(-x))

def redondeaLista(lista, decimales):
    for i in range(len(lista)):
        lista[i] = round(lista[i], decimales)
    return lista

def redNeuronal(entrada, w, b):
    for i, entrada in enumerate(entradas):
            # Primera Capa:
                # Neruona 1:
            N1 = entrada[0] * w[0] + entrada[1] * w[1] + b[0]
            v1 = sigmoid(N1)
                # Neruoan 2:
            N2 = entrada[0] * w[2] + entrada[1] * w[3] + b[1]
            v2 = sigmoid(N2)
    
            # Capa Oculta:
                # Neurona 1:
            h1 = v1 * w[4] + v2 * w[5] + b[2]
            z1 = sigmoid(h1)
                # Neurnoa 2:
            h2 = v1 * w[6] + v2 * w[7] + b[3]
            z2 = sigmoid(h2)
    
            # Salida:
            z3 = z1 * w[8] + z2 * w[9] + b[4]

            prediccion = sigmoid(z3)
            predicciones.append(prediccion)
            error = prediccion - salida_real[i]

            delta_salida = 2 * error * prediccion * (1 - prediccion) # ESTO ES PARA NO ESCRIBIR TANTO
            delta_z1 = delta_salida * w[8] * z1 * (1 - z1) # ESTO ES PARA NO ESCRIBIR TANTO
            delta_z2 = delta_salida * w[9] * z2 * (1 - z2) # ESTO ES PARA NO ESCRIBIR TANTO

            delta_v1 = (delta_z1 * w[4] + delta_z2 * w[6]) * v1 * (1 - v1) # ESTO ES PORQEU HAY MAS DE CAMINO DE LLEGAR AQUI (2) Y PARA NO ESCRIBIR TANTO 
            delta_v2 = (delta_z1 * w[5] + delta_z2 * w[7]) * v2 * (1 - v2) # ESTO ES PORQEU HAY MAS DE CAMINO DE LLEGAR AQUI (2) Y PARA NO ESCRIBIR TANTO
            # Vamos a calcular los 10 gradientes de w:                  (CALCULOS HECHOS A MANO EN PAPEL)
            grad_wn_total[9] += delta_salida * z2
            grad_wn_total[8] += delta_salida * z1
            grad_wn_total[7] += delta_z2 * v2
            grad_wn_total[6] += delta_z2 * v1
            grad_wn_total[5] += delta_z1 * v2
            grad_wn_total[4] += delta_z1 * v1
            grad_wn_total[3] += delta_v2 * entrada[1]
            grad_wn_total[2] += delta_v2 * entrada[0]
            grad_wn_total[1] += delta_v1 * entrada[1]
            grad_wn_total[0] += delta_v1 * entrada[0]

            # Calculamos los 5 sesgos:                  (CALCULOS HECHOS A MANO EN PAPEL)
            grad_bn_total[4] += delta_salida
            grad_bn_total[2] += delta_z1
            grad_bn_total[3] += delta_z2
            grad_bn_total[0] += delta_v1
            grad_bn_total[1] += delta_v2

def ProbarNeurona(entrada, w, b):
    predicciones = []
    for i, entrada in enumerate(entrada):
        # Primera Capa:
            # Neruona 1:
        N1 = entrada[0] * w[0] + entrada[1] * w[1] + b[0]
        v1 = sigmoid(N1)
            # Neruoan 2:
        N2 = entrada[0] * w[2] + entrada[1] * w[3] + b[1]
        v2 = sigmoid(N2)

        # Capa Oculta:
            # Neurona 1:
        h1 = v1 * w[4] + v2 * w[5] + b[2]
        z1 = sigmoid(h1)
            # Neurnoa 2:
        h2 = v1 * w[6] + v2 * w[7] + b[3]
        z2 = sigmoid(h2)

        # Salida:
        z3 = z1 * w[8] + z2 * w[9] + b[4]
        predicciones.append(sigmoid(z3))
    return predicciones

# 2 Neuronas Entrada (2 entradas c/u (x1, x2), 2 pesos c/u, 1 sesgos c/u, 1 salida c/u)
# 2 Neuronas Capa Oculta (1 entrada c/u, 1 peso c/u, 1 sesgo c/u, 1 salida c/u)
# 1 Neurona Salida (2 entrada, 2 peso, 1 sesgo, 1 salida)

w = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]
# w = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]
b = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5),]
tasa = 1
epoch = 0

while True:
    predicciones = []
    epoch += 1
    grad_wn_total = [0] * 10
    grad_bn_total = [0] * 5

    redNeuronal(entradas, w, b)

    for i in range(len(grad_wn_total)):
        w[i] -= tasa * grad_wn_total[i] / 4

    for i in range(5):
        b[i] -= tasa * grad_bn_total[i] / 4

    erroresCuadrado = []
    for i in range(len(salida_real)): # Calculamos los errores al cuadrado
        errorTMP = (predicciones[i] - salida_real[i])
        erroresCuadrado.append(errorTMP ** 2)

    MSE = sum(erroresCuadrado) / len(erroresCuadrado)

    if epoch % 1000 == 0:
        print(epoch, MSE ** 0.5, predicciones)
    if MSE ** 0.5 < 0.1:
        print(epoch, MSE ** 0.5, predicciones)
        print(f"\nPesos:\n{w}")
        print("RESULTADOS DEL ENTRENAMIENTO: ")
        print(redondeaLista(predicciones, 0))
        break
    # print("sigo...")

input("\n\nPRESIONE CUALQUIER TECLA PARA HACER LA PRUEBA\n")
print(redondeaLista(ProbarNeurona(entradas, w, b), 0))

