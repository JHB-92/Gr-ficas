from graficar import grafica
import json

f = open('points.json', encoding = "utf8")
graficas = json.load(f)
array_graphics = ['line','scatter','fill_line']
control = False
## FUNCIONES
# EVITA QUE DE FALLO EL SCRIPT POR METER MAL UN NOMBRE DE VARIABLE
def review_input(input_str):
    if input_str in array_graphics:
        return True
    else:
        print("No existe la gráfica solicitada. Introduzca otra de nuevo.\n")
# PLOTEA EN FUNCIÓN DE LAS GRÁFICAS QUE SE DESEEN. DEBE MODIFICARSE CADA VEZ QUE
# SE INCLUYA UNA NUEVA GRÁFICA
def plotting(input_graf):
    if input_graf == 'line':
        gr.line()
    elif input_graf == 'fill_line':
        gr.fill_line()
    elif input_graf == 'scatter':
        gr.scatter()
## SCRIPT PRINCIPAL
# PETICIÓN A USUARIO
while control == False:
    query_1 = input("Introduzca la gráfica que desea obtener: ")
    control = review_input(query_1)

query_2 = input("¿Desea guardar la gráfica? (y/n) ")

for graf in graficas['C_N6_97_202']:
    # SET UP
    title = graf['title']
    var_x = graf['var_x']
    label_x = graf['label_x']
    label_y = graf['label_y']
    x_limit = (graf['x_limit_inf'], graf['x_limit_sup'])
    y_limit = (graf['y_limit_inf'], graf['y_limit_sup'])
    save = False

    # HACE LA GRÁFICA
    for func in graf['funciones']:

        label = func['name']
        var_y = func['valores']
        color = func['color']

        if func == graf['funciones'][-1]:
            finish = True
            if query_2 == 'y':
                save = True
        else:
            finish = False

        gr = grafica(title = title, var_x = var_x, var_y = var_y, label_x = label_x, label_y = label_y, x_limit = x_limit, y_limit = y_limit, finish = finish, label = label, save = save, color = color)
        plotting(query_1)
