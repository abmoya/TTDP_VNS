# Metaheurísticas de trayectoria para la planificación de rutas turísticas en destino

## Descripción

Los turistas que visitan un determinado destino durante uno o más días, deben decidir qué puntos de interés (POIs, Points of Interest) son más atractivos para visitar, y establecer un orden de visita entre ellos. Habitualmente, dichos turistas disponen de un tiempo y un presupuesto limitados, y deben considerar el horario de apertura de cada POI, el tiempo que se empleará en visitarlo, el coste que se deriva de la visita, y el grado de satisfacción que puede proporcionar el POI, basado en el valor histórico, paisajístico o cultural, y en sus propias preferencias. 

Este problema, conocido como Tourist Trip Design Problem (TTDP), constituye un problema de optimización y presenta una elevada complejidad computacional, por lo que las aproximaciones metaheurísticas resultan muy adecuadas para su resolución. Se propone para su resolución la utilización de metaheurísticas de trayectoria, en concreto de Búsqueda en Entornos Variables (VNS, Variable Neighbourhood Search).

Este repositorio recoge la implementación en Python del algoritmo VNS en su versión Básica, y su aplicación a la planificación de rutas turísticas en destino (TTDP), junto con el conjunto de instancias de test utilizadas y los resultados obtenidos, para dos versiones del problema: TOPTW y TDTOPTW.

## Estructura del repositorio

El contenido se estructura como sigue:

* ttdp: Código en Python
* data: Instancias de prueba. Todas ellas pueden obtenerse en https://www.mech.kuleuven.be/en/cib/op/instances. Se agrupan en dos carpetas, *toptw* y *tdtoptw*.
* results: Se subdivide a su vez en:
   * csv: El resultado de la ejecución genera un fichero csv con la información de los parámetros de ejecución y el resultado obtenido, que se almacena por defecto en este directorio
   * json: En instancias con información de posición, es posible generar ficheros json con la información de los POIs y de las soluciones generadas en el proceso, que posteriormente pueden visualizarse gráficamente.
   * xls: Resultados brutos obtenidos para las instancias contenidas en data/tdtoptw, formateados en Excel
   * summary: Tablas resumen de los resultados recogidos en *xls*


## Instrucciones de ejecución

El desarrollo se ha realizado con Python 3.7. El módulo a ejecutar es *test.py*, al que debe pasarse, vía fichero de configuración json, los parámetros de ejecución, a saber,   problema a ejecutar (TOPTW, TDTOPTW), el/los fichero/s de instancias, directorio de resultados, etc. La ejecución *test.py -h* devuelve una lista completa de opciones.

Existen dos modos de ejecución: masiva e individual. En el primer caso, para el/los fichero/s que cumplan con las características indicadas se ejecuta una batería de test modificando diversos parámetros. En el segundo, se especifican individualmente el fichero de instancias y los parámetros del algoritmo a aplicar, y se realiza un test simple. En la carpeta *ttdp* se incluyen ejemplos de ficheros json de configuración para cada caso.

