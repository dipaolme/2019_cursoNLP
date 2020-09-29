import os
import re
import functools
import itertools
import nltk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from operator import mul
from string import punctuation
from collections import Counter, defaultdict


PUNTUACION = re.compile(r'[{}¡¿…«»—]'.format(punctuation))


def _obtener_ubicacion_absoluta(ubicacion):
    directorio_actual = os.getcwd()
    if '..' in ubicacion:
        lista_ubicacion = ubicacion.split('/')
        lista_directorio_actual = directorio_actual.split('/')
        ubicacion_real = '{}'.format('/'.join(lista_directorio_actual[:-lista_ubicacion.count('..')] + lista_ubicacion[lista_ubicacion.count('..'):]))
    elif '.' in ubicacion:
        ubicacion_real = '{}/{}'.format(directorio_actual, ubicacion)
    else:
        ubicacion_real = '{}/{}'.format(directorio_actual, ubicacion)
    return ubicacion_real

##### MÉTODOS DE TF-IDF Y DISTANCIA COSENO DEBERÍAN APLICARSE CON SCIPY O SKLEARN
def _similitud_coseno(vector1, vector2):
    similitud = np.dot(vector1, vector2) / (np.linalg.norm(vector1)*np.linalg.norm(vector2))
    return similitud

##### MÉTODOS DE TF-IDF Y DISTANCIA COSENO DEBERÍAN APLICARSE CON SCIPY O SKLEARN
def calcular_similitudes(matriz):
    documentos = matriz.columns
    pares = itertools.combinations(documentos, 2)
    similitudes = {'documento_1':[], 'documento_2':[], 'similitud':[]}
    for doc1, doc2 in pares:
        vector1 = matriz[doc1].values
        vector2 = matriz[doc2].values
        similitudes['documento_1'].append(doc1)
        similitudes['documento_2'].append(doc2)
        valor_similitud = _similitud_coseno(vector1, vector2)
        similitudes['similitud'].append(valor_similitud)
    similitudes = pd.DataFrame(similitudes)
    return similitudes

##### MÉTODOS DE TF-IDF Y DISTANCIA COSENO DEBERÍAN APLICARSE CON SCIPY O SKLEARN
def calcular_similitud(matriz, documento1, documento2):
    vector1 = matriz[documento1].values
    vector2 = matriz[documento2].values
    similitud = _similitud_coseno(vector1, vector2)
    return similitud


class Documento:
    
    def __init__(self, **kwargs):
        self.texto = None
        self.metadata = dict()
        self.conteo = None
        self.vocabulario = None
        for atributo, valor in kwargs.items():
            setattr(self, atributo, valor)
            
    def _contar_palabras_texto(funcion_lectura_texto):
        def contar_palabras_texto(self, *args):
            if hasattr(self, 'tokenizador_palabras'):
                self.conteo = Counter(self.tokenizador_palabras.tokenize(funcion_lectura_texto(self, *args)))
            else:
                self.conteo = Counter(funcion_lectura_texto(self, *args).split())
            self.vocabulario = list(self.conteo.keys())
            if hasattr(self, 'tokenizador_oraciones'):
                self.oraciones = self.tokenizador_oraciones.tokenize(self.texto)
            else:
                self.oraciones = nltk.tokenize.sent_tokenize(self.texto)
            self.cantidad_types = len(self.vocabulario)
            self.cantidad_tokens = sum(self.conteo.values())
        return contar_palabras_texto
    
    def _extraer_n_gramas_oracion(self, oracion, n, remover_puntuacion = True, minusculas = True):
        inicio = (n-1) * ['<oracion>']
        final = (n-1) * ['</oracion>']
        if minusculas:
            oracion = oracion.lower()
        if remover_puntuacion:
            oracion = PUNTUACION.sub('', oracion)
        if hasattr(self, 'tokenizador_palabras'): 
            tokens = self.tokenizador_palabras.tokenize(oracion)
        else:
            tokens = oracion.split()
        cadena = inicio + tokens + final
        for indice in range(len(tokens) + n - 1):
            yield tuple(cadena[indice:indice+n])

    @_contar_palabras_texto
    def leer_archivo_texto(self, ubicacion, codificacion = 'utf8'):
        with open(ubicacion, encoding = codificacion) as archivo:
            texto = '\n'.join(archivo.readlines())
        self.texto = texto
        ubicacion_real = _obtener_ubicacion_absoluta(ubicacion)
        self.metadata['archivo de origen'] = ubicacion_real
        return texto
        
    @_contar_palabras_texto
    def leer_variable_texto(self, variable):
        self.texto = variable
        return texto
    
    def calcular_n_gramas(self, n, remover_puntuacion = True, minusculas = True):
        n_gramas = Counter()
        for oracion in self.oraciones:
            n_gramas += Counter(self._extraer_n_gramas_oracion(oracion, n, remover_puntuacion = remover_puntuacion, minusculas = minusculas))
        return n_gramas
                
    def incorporar_metadata(self, **kwargs):
        for campo, valor in kwargs.items():
            self.metadata[campo] = valor
            
    def calcular_zipf(self):
        recuento = pd.Series(self.conteo)
        ranking = (
            recuento.reset_index(name = 'cantidad')
            .rename({'index':'palabra'}, axis = 1)
            .sort_values('cantidad', ascending = False)
            .reset_index(drop = 'true')
        )
        ranking['orden'] = ranking.index + 1
        return ranking
    
    def graficar_zipf(self, log = True, dimension = (10,10), **kwargs):
        datos = self.calcular_zipf()
        plt.figure(figsize = dimension)
        sns.set_style('whitegrid')
        if log:
            plt.xscale('log')
            plt.yscale('log')
        plt.tight_layout()
        sns.lineplot(x='orden', y='cantidad',data=datos, **kwargs)
        plt.show()

    
class Corpus:
    
    def __init__(self, **kwargs):
        self.documentos = dict()
        self.metadata = pd.DataFrame()
        self.conteo = Counter()
        self.vocabulario = self.conteo.keys()
        self.frecuencia_en_documentos = Counter()
        for atributo, valor in kwargs.items():
            setattr(self, atributo, valor)
            
    def actualizar_metadata(self):
        self.metadata = pd.DataFrame()
        for nombre_documento, documento in self.documentos.items():
            self.metadata = pd.concat([self.metadata, pd.DataFrame(documento.metadata, [0])], sort = True).reset_index(drop = True)
        
    def _actualizar(funcion_lectura_texto):
        def actualizar(self, *args):
            estado_previo = set(self.documentos.items())
            funcion_lectura_texto(self, *args)
            estado_nuevo = set(self.documentos.items())
            diferencia = set(estado_nuevo).difference(set(estado_previo))
            for nombre_documento, valores_documento in diferencia:
                self.conteo += self.documentos[nombre_documento].conteo
                self.metadata = pd.concat([self.metadata, pd.DataFrame(self.documentos[nombre_documento].metadata, [0])], sort = True)
                self.frecuencia_en_documentos += Counter(valores_documento.vocabulario)
            self.vocabulario = self.conteo.keys()
            self.cantidad_tokens = sum(self.conteo.values())
            self.cantidad_types = len(self.vocabulario)
            self.cantidad_documentos = len(self.documentos.items())
            self.metadata = self.metadata.reset_index(drop = True)
        return actualizar
    
    def _graficar(funcion_para_armar_grafico):
        def graficar(self, dimension = (10,10), mostrar = True, ubicacion = False, *args, **kwargs):
            sns.set_style('whitegrid')
            plt.figure(figsize = dimension)
            funcion_para_armar_grafico(self, *args, **kwargs)
            if ubicacion:
                plt.savefig(ubicacion, bbox_inches = 'tight')
            if mostrar:
                plt.show()
            else:
                plt.close()
        return graficar
    
    @_actualizar
    def leer_objeto_documento(self, objeto_documento):
        if 'archivo de origen' in objeto_documento.metadata.keys():
            self.documentos[objeto_documento.metadata['archivo de origen']] = objeto_documento
        else:
            self.documentos['documento_{}'.format(len(self.documentos.keys()))] = objeto_documento
    
    @_actualizar
    def leer_archivo(self, ubicacion, codificacion = 'utf8'):
        doc = Documento()
        if hasattr(self, 'tokenizador'):
            doc.tokenizador_palabras = self.tokenizador_palabras
        doc.leer_archivo_texto(ubicacion, codificacion)
        self.documentos[doc.metadata['archivo de origen']] = doc
                
    @_actualizar
    def leer_lista_de_archivos(self, lista, codificacion = 'utf8'):
        for item in lista:
            doc = Documento()
            if hasattr(self, 'tokenizador_palabras'):
                doc.tokenizador_palabras = self.tokenizador_palabras
            doc.leer_archivo_texto(item, codificacion)
            self.documentos[doc.metadata['archivo de origen']] = doc
    
    def leer_directorio(self, ubicacion, codificacion = 'utf8'):
        contenido = os.listdir(ubicacion)
        lista = ['{}/{}'.format(ubicacion, archivo) for archivo in contenido]
        self.leer_lista_de_archivos(lista)
        
    def consultar_documento(self, documento):
        return self.documentos[documento]
    
    def consultar_datos_documentos(self):
        datos_documentos = pd.DataFrame()
        for documento, datos in self.documentos.items():
            registro = pd.DataFrame(
                {
                    'archivo de origen':self.documentos[documento].metadata['archivo de origen'],
                    'cantidad de types': self.documentos[documento].cantidad_types,
                    'cantidad de tokens': self.documentos[documento].cantidad_tokens
                }, index = [0]
            )
            datos_documentos = pd.concat([datos_documentos, registro], sort = True)
        datos_documentos = datos_documentos.reset_index(drop = True).merge(self.metadata)
        return datos_documentos
    
    # FUNCIÓN PELIGROSA - Sólo sirve con etiquetadores que tengan un método 'tag' y funcionen sobre oraciones tokenizadas
    def etiquetar_corpus(self, etiquetador, nombre_etiqueta = None):
        if not nombre_etiqueta:
            nombre_etiqueta = etiquetador.__repr__
        for documento, datos in self.documentos.items():
            def etiquetar():
                for oracion in datos.oraciones:
                    if hasattr(self, 'tokenizador_palabras'):
                        tokenizada = self.tokenizador_palabras.tokenize(oracion)
                    else:
                        tokenizada = oracion.split()
                    etiquetada = etiquetador.tag(tokenizada)
                    yield etiquetada
            datos_etiquetados = list(etiquetar())
            setattr(datos, str(nombre_etiqueta), datos_etiquetados)
    
    # FUNCIÓN PELIGROSA - Si el tokenizador no es el mismo que el del tagger, los resultados van a dar sí o sí mal
    def parear_etiquetas(self, nombre_etiqueta, prefijo = 'etiqueta_'):
        for documento, datos in self.documentos.items():
            pares = zip(datos.oraciones, getattr(datos, nombre_etiqueta))
            setattr(datos, '{}{}'.format(prefijo, nombre_etiqueta), list(pares))
            
    def contar_etiquetas_por_documento(self, nombre_etiqueta, prefijo = 'conteo_'):
        for documento, datos in self.documentos.items():
            lista_etiquetada = getattr(datos, nombre_etiqueta)
            conteo_documento = Counter()
            for oracion in lista_etiquetada:
                conteo_oracion = Counter([item[1] for item in oracion])
                conteo_documento += conteo_oracion
            setattr(datos, '{}{}'.format(prefijo, nombre_etiqueta), conteo_documento)
            
    def contar_etiquetas(self, nombre_etiqueta, prefijo = 'conteo_'):
        self.contar_etiquetas_por_documento(nombre_etiqueta, prefijo)
        total = Counter()
        for documento, datos in self.documentos.items():
            total += getattr(datos, '{}{}'.format(prefijo, nombre_etiqueta))
        total = pd.Series(total).reset_index(name = 'cantidad').sort_values('cantidad', ascending = False)
        return total
    
    @_graficar
    def graficar_etiquetas_por_frecuencia(self, nombre_etiqueta, prefijo = 'conteo_', **kwargs):
        datos = self.contar_etiquetas(nombre_etiqueta, prefijo)
        sns.barplot(x='cantidad',y='index', data = datos, palette='viridis')
    
    def calcular_n_gramas(self, n, remover_puntuacion = True, minusculas = True):
        total = Counter()
        for documento, datos in self.documentos.items():
            total += datos.calcular_n_gramas(n, remover_puntuacion = True, minusculas = True)
        return total
    
    ##### MÉTODOS DE TF-IDF Y DISTANCIA COSENO DEBERÍAN APLICARSE CON SCIPY O SKLEARN
    def calcular_matriz_termino_documento(self):
        matriz = pd.DataFrame()
        for documento, datos in self.documentos.items():
            cantidades = pd.Series(datos.conteo).T
            cantidades.name = documento
            matriz = matriz.append(cantidades)
        matriz = matriz.T.fillna(0)
        return matriz
    
    ##### MÉTODOS DE TF-IDF Y DISTANCIA COSENO DEBERÍAN APLICARSE CON SCIPY O SKLEARN
    ##### EL CÁLCULO DE TF-IDF ACÁ ES DUDOSO
    def calcular_matriz_tf_idf(self):
        matriz = pd.DataFrame()
        termino_documento = self.calcular_matriz_termino_documento()
        for termino in termino_documento.index:
            idf = np.log(len(self.documentos.items())/self.frecuencia_en_documentos[termino])
            if idf == 0:
                idf = .1
            tfs = termino_documento.loc[termino]
            tfidf = tfs / idf
            tfidf.name = termino
            matriz = matriz.append(tfidf)
        matriz = matriz.fillna(0)
        return matriz
    
    def calcular_zipf(self):
        recuento = pd.Series(self.conteo)
        ranking = (
            recuento.reset_index(name = 'cantidad')
            .rename({'index':'palabra'}, axis = 1)
            .sort_values('cantidad', ascending = False)
            .reset_index(drop = 'true')
        )
        ranking['orden'] = ranking.index + 1
        return ranking
    
    @_graficar
    def graficar_zipf(self, log = True, **kwargs):
        datos = self.calcular_zipf()
        if log:
            plt.xscale('log')
            plt.yscale('log')
        plt.tight_layout()
        sns.lineplot(x='orden', y='cantidad',data=datos, **kwargs)
    
    @_graficar
    def graficar_heaps(self, log = True, **kwargs):
        datos = self.consultar_datos_documentos()
        if log:
            plt.xscale('log')
            plt.yscale('log')
        plt.tight_layout()
        sns.lineplot(x='cantidad de tokens' ,y='cantidad de types', data=datos, **kwargs)
        sns.scatterplot(x='cantidad de tokens' ,y='cantidad de types', data=datos, **kwargs, s = 200)
                
    def seleccionar_subconjunto(self, lista_documentos):
        subconjunto = Corpus()
        for documento in lista_documentos:
            subconjunto.leer_objeto_documento(documento)
        return subconjunto
    
    
class NGramas: #### TODA ESTA CLASE ESTÁ MEDIO MOCHA
    
    def __init__(self, corpus, n):
        self.n = n
        setattr(self, '_{}_gramas'.format(n), dict(corpus.calcular_n_gramas(n)))
        setattr(self, '_{}_gramas'.format(n-1), dict(corpus.calcular_n_gramas(n-1)))
        setattr(self, 'probabilidad_{}_gramas'.format(n), sum(getattr(self, '_{}_gramas'.format(n)).values()))
        setattr(self, 'probabilidad_{}_gramas'.format(n-1), sum(getattr(self, '_{}_gramas'.format(n-1)).values()))
        
        
    #### ESTA FUNCIÓN ANDA LISA Y LLANAMENTE MAL. REVISAR.    
    def calcular_probabilidad_oracion(self, oracion, tokenizador = None):
        inicio = (self.n-1) * ['<oracion>']
        final = (self.n-1) * ['</oracion>']
        oracion = oracion.lower()
        if tokenizador: 
            tokens = tokenizador.tokenize(oracion)
        else:
            tokens = oracion.split()
        cadena = inicio + tokens + final
        def probabilidad_ngramas_oracion():
            for indice in range(len(tokens) + self.n):
                if not tuple(cadena[indice:indice+self.n]) in getattr(self, '_{}_gramas'.format(self.n)):
                    frecuencia_cadena = 0.01
                else:
                    frecuencia_cadena = getattr(self, '_{}_gramas'.format(self.n))[tuple(cadena[indice:indice+self.n])]
                if not tuple(cadena[indice:indice+self.n-1]) in getattr(self, '_{}_gramas'.format(self.n-1)):
                    frecuencia_previos = 0.01
                else:
                    frecuencia_previos = getattr(self, '_{}_gramas'.format(self.n-1))[tuple(cadena[indice:indice+self.n-1])]
                probabilidad_cadena = frecuencia_cadena/getattr(self, 'probabilidad_{}_gramas'.format(self.n))
                probabilidad_previos = frecuencia_previos/getattr(self, 'probabilidad_{}_gramas'.format(self.n-1))
                yield probabilidad_cadena/probabilidad_previos
        probabilidad_oracion = functools.reduce(mul, probabilidad_ngramas_oracion())
        return probabilidad_oracion
    
    def generar_tupla(self, inicio):
        posibilidades = [(tupla,getattr(self, '_{}_gramas'.format(self.n))[tupla]) for tupla in getattr(self, '_{}_gramas'.format(self.n)) if tuple(tupla[:self.n-1]) == inicio]
        valores_muestreo = np.cumsum([valor[1] for valor in posibilidades])
        pares_muestreo = zip(valores_muestreo, [valor[0] for valor in posibilidades])
        tabla_muestreo = {valor:tupla for valor,tupla in pares_muestreo}
        muestra = np.random.randint(max(tabla_muestreo.keys()))
        if muestra in tabla_muestreo.keys():
            tupla = tabla_muestreo[muestra]
        else:
            resto = [valor for valor in tabla_muestreo.keys() if valor > muestra]
            tupla = tabla_muestreo[min(resto)]
        return tupla[-1]
    
    def generar_texto(self, oracion = None):
        if not oracion:
            oracion = tuple((self.n-1) * ['<oracion>'])
            nueva_palabra = self.generar_tupla(tuple(oracion[-(self.n-1):]))
        while nueva_palabra != '</oracion>':
            nueva_palabra = self.generar_tupla(tuple(oracion[-(self.n-1):]))
            oracion = list(oracion) + [nueva_palabra]
        oracion_final = ' '.join(oracion[self.n:-1]).capitalize() +'.'
        return oracion_final