#Finanzas.py
# %%

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import networkx as nx

class Finanzas:
    """
    Clase principal que organiza funcionalidades financieras: Binomial, Deuda y Graficos.
    """

    class Binomial:
        """
        Funcionalidades relacionadas con opciones binomiales y Black-Scholes.
        """

        @staticmethod
        def americana(tipo, S, K, sigma, T, r, n):  
            """
            Calcula el precio de una opción americana utilizando el modelo binomial.

            Args:
                tipo (str): Tipo de opción ("call" o "put").
                S (float): Precio del activo subyacente.
                K (float): Precio de ejercicio.
                sigma (float): Volatilidad del activo.
                T (float): Tiempo hasta el vencimiento (en años).
                r (float): Tasa libre de riesgo.
                n (int): Número de pasos en el árbol binomial.

            Returns:
                float: Precio de la opción americana.
            """
            z = 1 if tipo == "call" else -1
            dt = T / n
            u = np.exp(sigma * np.sqrt(dt))
            d = 1 / u
            p = (np.exp(r * dt) - d) / (u - d)
            Df = np.exp(-r * dt)
            
            prices_up = S * u ** np.arange(n + 1)
            prices_down = d ** np.arange(n, -1, -1)
            opt_val = z * (prices_up * prices_down - K)
            opt_val = np.maximum(opt_val, 0)

            for j in range(n - 1, -1, -1):
                for i in range(j + 1):
                    early_exercise = z * (S * u**i * d**(j - i) - K)
                    continuation = (p * opt_val[i + 1] + (1 - p) * opt_val[i]) * Df
                    opt_val[i] = max(early_exercise, continuation)

            return opt_val[0]

        @staticmethod
        def europea(tipo, S, K, sigma, T, r, n):

            """
            Calcula el precio de una opción europea utilizando el modelo binomial.

            Args:
                tipo (str): Tipo de opción ("call" o "put").
                S (float): Precio del activo subyacente.
                K (float): Precio de ejercicio.
                sigma (float): Volatilidad del activo.
                T (float): Tiempo hasta el vencimiento (en años).
                r (float): Tasa libre de riesgo.
                n (int): Número de pasos en el árbol binomial.

            Returns:
                float: Precio de la opción europea.
            """
            z = 1 if tipo == "call" else -1
            dt = T / n
            u = np.exp(sigma * np.sqrt(dt))
            d = 1 / u
            p = (np.exp(r * dt) - d) / (u - d)
            Df = np.exp(-r * dt)

            prices_up = S * u ** np.arange(n + 1)
            prices_down = d ** np.arange(n, -1, -1)
            opt_val = z * (prices_up * prices_down - K)
            opt_val = np.maximum(opt_val, 0)

            for j in range(n - 1, -1, -1):
                opt_val = (p * opt_val[1:j + 2] + (1 - p) * opt_val[:j + 1]) * Df

            return opt_val[0]

        @staticmethod
        def black_scholes(tipo, S, K, sigma, T, r):
            """
            Calcula el precio de una opción usando el modelo Black-Scholes.

            Args:
                tipo (str): Tipo de opción ("call" o "put").
                S (float): Precio del activo subyacente.
                K (float): Precio de ejercicio.
                sigma (float): Volatilidad.
                T (float): Tiempo hasta el vencimiento (en años).
                r (float): Tasa libre de riesgo.

            Returns:
                float: Precio de la opción.
            """
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if tipo == "call":
                return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        @staticmethod
        def verificar_paridad(S, K, sigma, T, r):
            """
            Verifica la paridad Put-Call para opciones europeas.
            """
            call = Finanzas.Binomial.black_scholes("call", S, K, sigma, T, r)
            put = Finanzas.Binomial.black_scholes("put", S, K, sigma, T, r)
            pv_k = K * np.exp(-r * T)  # Valor presente del strike
            return np.isclose(call - put, S - pv_k)

        @staticmethod
        def calcular_griegos(S, K, sigma, T, r, tipo="call"):
            """
            Calcula los griegos Delta, Gamma, Vega y Theta.
            """
            # Fórmulas completas para Delta, Gamma, Vega y Theta

        @staticmethod
        def comparar_convergencia(S, K, sigma, T, r, max_n=100):
            """
            Compara la convergencia del modelo binomial hacia Black-Scholes.
            """
            # Gráfica de convergencia (similar al diseño anterior)

    class Deuda:
        """
        Funcionalidades relacionadas con la gestión de deuda.
        """

        @staticmethod
        def calcular_amortizacion(monto, tasa_semestral, num_semestral):
            """
            Calcula la tabla de amortización de un préstamo.
            """
            cuota = monto * tasa_semestral / (1 - (1 + tasa_semestral) ** -num_semestral)
            saldo = monto
            tabla = []

            for semestre in range(1, num_semestral + 1):
                interes = saldo * tasa_semestral
                capital = cuota - interes
                saldo -= capital
                tabla.append({"Semestre": semestre, "Cuota": cuota, "Interés": interes, "Capital": capital, "Saldo": max(saldo, 0)})

            return pd.DataFrame(tabla)

        @staticmethod
        def calcular_valor_presente(tabla, tasa_semestral):
            """
            Calcula el valor presente de los flujos de deuda.
            """
            tabla["Valor Presente"] = tabla["Cuota"] / (1 + tasa_semestral) ** tabla["Semestre"]
            return tabla["Valor Presente"].sum()

    class Graficos:
        """
        Funcionalidades relacionadas con la generación de gráficos.
        """

        @staticmethod
        def graficar_flujo(tabla):
            """
            Genera un gráfico del flujo de caja de deuda.
          
            Args:
                tabla (pandas.DataFrame): Tabla de amortización.
            """
            plt.figure(figsize=(10, 6))
            plt.bar(tabla["Semestre"], tabla["Cuota"], label="Cuota Total", color="blue", alpha=0.7)
            plt.plot(tabla["Semestre"], tabla["Interés"], label="Interés", color="orange", linestyle="--")
            plt.plot(tabla["Semestre"], tabla["Capital"], label="Capital", color="green", linestyle="--")
            plt.title("Flujo de Caja de Deuda")
            plt.xlabel("Semestre")
            plt.ylabel("Monto")
            plt.legend()
            plt.grid()
            plt.show()

        @staticmethod
        def graficar_arbol(S, u, d, n):
            """
            Genera un gráfico del árbol binomial de precios.

            Args:
                S (float): Precio inicial del activo subyacente.
                u (float): Factor de aumento del precio.
                d (float): Factor de disminución del precio.
                n (int): Número de pasos en el árbol.
            """
            G = nx.DiGraph()
            pos = {}

            # Crear nodos y sus posiciones
            for i in range(n + 1):
                for j in range(i + 1):
                    nombre_nodo = f"{i}-{j}"
                    precio = S * (u ** j) * (d ** (i - j))
                    G.add_node(nombre_nodo, precio=precio)
                    pos[nombre_nodo] = (i, -j)

            # Agregar aristas solo si los nodos destino existen
            for i in range(n):
                for j in range(i + 1):
                    nodo_actual = f"{i}-{j}"
                    nodo_arriba = f"{i+1}-{j+1}"
                    nodo_abajo = f"{i+1}-{j}"
                    if nodo_arriba in G.nodes:
                        G.add_edge(nodo_actual, nodo_arriba)
                    if nodo_abajo in G.nodes:
                        G.add_edge(nodo_actual, nodo_abajo)

            plt.figure(figsize=(12, 8))
            nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=8, font_weight="bold")
            labels = {node: f"{data['precio']:.2f}" for node, data in G.nodes(data=True)}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight="bold")
            plt.title("Árbol Binomial de Precios")
            plt.show()

        @staticmethod
        def graficar_convergencia(binomial_precio, bs_precio, pasos):
            """
            Genera un gráfico de la convergencia del modelo binomial hacia Black-Scholes.

            Args:
                binomial_precio (list): Lista de precios binomiales para diferentes pasos.
                bs_precio (float): Precio calculado con Black-Scholes.
                pasos (list): Lista del número de pasos correspondientes.
            """
            plt.figure(figsize=(10, 6))
            plt.plot(pasos, binomial_precio, label="Modelo Binomial", marker="o", linestyle="-")
            plt.axhline(y=bs_precio, color="red", linestyle="--", label="Black-Scholes")
            plt.title("Convergencia del Modelo Binomial")
            plt.xlabel("Número de pasos (n)")
            plt.ylabel("Precio de la Opción")
            plt.legend()
            plt.grid()
            plt.show()

        @staticmethod
        def graficar_griegos(S, K, sigma, T, r):
            """
            Genera gráficos de los griegos en función del precio del activo subyacente.

            Args:
                S (float): Precio del activo subyacente inicial.
                K (float): Precio de ejercicio.
                sigma (float): Volatilidad.
                T (float): Tiempo hasta el vencimiento.
                r (float): Tasa libre de riesgo.
            """
            S_range = np.linspace(S * 0.5, S * 1.5, 100)
            deltas, gammas, vegas, thetas = [], [], [], []

            for S_i in S_range:
                d1 = (np.log(S_i / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)
                delta = norm.cdf(d1)
                gamma = norm.pdf(d1) / (S_i * sigma * np.sqrt(T))
                vega = S_i * norm.pdf(d1) * np.sqrt(T) / 100
                theta = (-S_i * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
                         r * K * np.exp(-r * T) * norm.cdf(d2)) / 365

                deltas.append(delta)
                gammas.append(gamma)
                vegas.append(vega)
                thetas.append(theta)

            # Graficar cada griego
            plt.figure(figsize=(10, 6))
            plt.plot(S_range, deltas, label="Delta")
            plt.plot(S_range, gammas, label="Gamma")
            plt.plot(S_range, vegas, label="Vega")
            plt.plot(S_range, thetas, label="Theta")
            plt.title("Griegos en función del Precio del Activo Subyacente")
            plt.xlabel("Precio del Activo Subyacente (S)")
            plt.ylabel("Valor del Griego")
            plt.legend()
            plt.grid()
            plt.show()

    class Valoracion:
        """
        Funcionalidades relacionadas con valoración empresarial utilizando Black-Scholes.
        """
        @staticmethod
        def calcular_d1(S, K, rf, sigma, T):
            d1 = (np.log(S / K) + (rf + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            #print(f"d1: {d1:.4f}")
            globals()['d1'] = round(d1, 4)
            return d1

        @staticmethod
        def calcular_d2(S, K, rf, sigma, T):
            d1 = Finanzas.Valoracion.calcular_d1(S, K, rf, sigma, T)
            d2 = d1 - sigma * np.sqrt(T)
            print(f"d2: {d2:.4f}")
            globals()['d2'] = round(d2, 4)
            return d2

        @staticmethod
        def obtener_d1_d2(S, K, rf, sigma, T):
            d1 = Finanzas.Valoracion.calcular_d1(S, K, rf, sigma, T)
            d2 = d1 - sigma * np.sqrt(T)
            #print(f"d1: {d1:.4f}, d2: {d2:.4f}")
            globals()['d1'] = round(d1, 4)
            globals()['d2'] = round(d2, 4)
            return d1, d2
        
        @staticmethod
        def calcular_phi(S, K, rf, sigma, T, variable="d1"):
            """
            Calcula el valor de N(d1) o N(d2), es decir, la probabilidad acumulada bajo la distribución normal estándar.

            Args:
                variable (str): "d1" o "d2" para especificar qué valor calcular.

            Returns:
                float: N(d1) o N(d2)
            """
            d1 = Finanzas.Valoracion.calcular_d1(S, K, rf, sigma, T)
            if variable == "d2":
                valor = d1 - sigma * np.sqrt(T)
                nombre = "N(d2)"
            else:
                valor = d1
                nombre = "N(d1)"
            phi = norm.cdf(valor)
            #print(f"{nombre} o Φ({variable}): {phi:.4f}")
            globals()[f"phi_{variable}"] = round(phi, 4)
            return phi

        @staticmethod
        def calcular_call_put(S, K, rf, sigma, T):
            """
            Calcula los valores de las opciones call y put.

            Returns:
                dict: Valores de las opciones call y put.
            """
            d1, d2 = Finanzas.Valoracion.calcular_d1_d2(S, K, rf, sigma, T)
            call = S * norm.cdf(d1) - K * np.exp(-rf * T) * norm.cdf(d2)
            put = K * np.exp(-rf * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            return {"Call": call, "Put": put}
       
        @staticmethod
        def calcular_valor_presente_dividendo(div, r, T):
            """
            Calcula el valor presente de un dividendo futuro descontado a una tasa dada.

            Fórmula:
                VP = div / (1 + r)^T

            Args:
                div (float): Monto del dividendo esperado.
                r (float): Tasa de descuento.
                T (float): Tiempo en años hasta recibir el dividendo.

            Returns:
                float: Valor presente del dividendo.
            """
            vp = div / (1 + r) ** T
            print(f"Valor presente del dividendo: {vp:.4f}")
            return vp
        
        @staticmethod
        def convertir_tasa(tasa, from_period, to_period):
            """
            Convierte una tasa de interés entre diferentes periodos.
            """
            periodos = {"anual": 1, "mensual": 12, "semestral": 2, "diario": 365}
            factor = periodos[to_period] / periodos[from_period]
            return (1 + tasa) ** factor - 1

        @staticmethod
        def calcular_valor_empresa(S, K, rf, sigma, T):
            """
            Calcula el valor del patrimonio y la deuda riesgosa.
            """
            valores_opciones = Finanzas.Valoracion.calcular_call_put(S, K, rf, sigma, T)
            patrimonio = valores_opciones["Call"]
            deuda_riesgosa = S - patrimonio
            return {"Patrimonio": patrimonio, "Deuda Riesgosa": deuda_riesgosa}
# %%
S=50
K=50
r=.1
sigma=0.30
T=3/12
n=1
# %% Opción europea put

eu_put=Finanzas.Binomial.europea("put", S, K, sigma, T, r, n)
print(f"Precio de la opción europea: {eu_put:.2f}")

# %% Opción europea call

eu_call=Finanzas.Binomial.europea("call", S, K, sigma, T, r, n)
print(f"Precio de la opción europea: {eu_call:.2f}")

# %% Opción americana put

am_put=Finanzas.Binomial.americana("put", S, K, sigma, T, r, n)    
print(f"Precio de la opción americana: {am_put:.2f}")

# %%  Opción americana call

am_call=Finanzas.Binomial.americana("call", S, K, sigma, T, r, n)
print(f"Precio de la opción americana: {am_call:.2f}")

# %% Verificar paridad Put-Call

paridad = Finanzas.Binomial.verificar_paridad(S, K, sigma, T, r)
print(f"Paridad Put-Call: {'Cumple' if paridad else 'No cumple'}")

# %% Black-Scholes Call

bs_call = Finanzas.Binomial.black_scholes("call", S, K, sigma, T, r)
print(f"Precio Black-Scholes Call: {bs_call:.2f}")

# %% Black-Scholes Put
bs_put = Finanzas.Binomial.black_scholes("put", S, K, sigma, T, r)
print(f"Precio Black-Scholes Put: {bs_put:.2f}")
# %% Comparar convergencia del modelo binomial hacia Black-Scholes
pasos = np.arange(1, 101)
binomial_precio = [Finanzas.Binomial.europea("call", S, K, sigma, T, r, n) for n in pasos]
bs_precio = Finanzas.Binomial.black_scholes("call", S, K, sigma, T, r)
Finanzas.Graficos.graficar_convergencia(binomial_precio, bs_precio, pasos)

# %% Árbol binomial de precios

S = 100 
u = 1.2
d = 0.8
n = 3

Finanzas.Graficos.graficar_arbol(S, u, d, n)

# %% Ejercicio 15 .13 Hull
S=50
K=50
r=0.1
sigma=0.30
T=3/12
n=1
Finanzas.Valoracion.obtener_d1_d2(S, K, r, sigma, T)
print(d1, d2)
eup_1513=Finanzas.Binomial.europea("put", S, K, sigma, T, r, n)
print(f"Precio de la opción europea put 15.13 Hull: {eup_1513:.2f}")
bsp_1513=Finanzas.Binomial.black_scholes("put", S, K, sigma, T, r)
print(f"Precio Black-Scholes put 15.13 Hull: {bsp_1513:.2f}")   
Finanzas.Valoracion.calcular_phi(S, K, r, sigma, T, variable="d1")  # N(d1)
Finanzas.Valoracion.calcular_phi(S, K, r, sigma, T, variable="d2")  # N(d2)
print(f"Phid1: {phi_d1}\nPhid2: {phi_d2}")# %%
phild1 = 1-phi_d1  # N(d1)
phild2 = 1-phi_d2  # N(d1)
print(f"1-Phid1: {phild1}\n1-Phid2: {phild2}")# %%

# %% ejercicio 15.14 Hull
t=2/12
div =1.5
div_vp=Finanzas.Valoracion.calcular_valor_presente_dividendo(div,r,t)
S1=S-div_vp
print(f"Precio del activo subyacente ajustado por dividendo: {S1:.2f}")

# Recalcular d1 y d2 con S ajustado por dividendo
Finanzas.Valoracion.obtener_d1_d2(S1, K, r, sigma, T)
Finanzas.Valoracion.calcular_phi(S1, K, r, sigma, T, variable="d1")
Finanzas.Valoracion.calcular_phi(S1, K, r, sigma, T, variable="d2")
print(f"d1: {d1}, d2: {d2}")
print(f"Phid1: {phi_d1}\nPhid2: {phi_d2}")
# %% Opción europea put ajustada por dividendo
eup_1514 = Finanzas.Binomial.europea("put", S1, K, sigma, T, r, n)
print(f"Precio de la opción europea put ajustada por dividendo: {eup_1514:.2f}")
# %% Black-Scholes put ajustada por dividendo
bsp_1514 = Finanzas.Binomial.black_scholes("put", S1, K, sigma, T, r)
print(f"Precio Black-Scholes put ajustada por dividendo: {bsp_1514:.2f}")