from graficar import grafica
import json

f = open('points.json', encoding = "utf8")
graficas = json.load(f)
graf = graficas['C_N6_97_202'][0]
array_graphics = ['line','scatter','fill_line']
control = False

def review_input(input_str):
    if input_str in array_graphics:
        return True
    else:
        print("No existe la gráfica solicitada. Introduzca otra de nuevo.\n")

def plotting(input_graf):
    if input_graf == 'line':
        gr.line()
    elif input_graf == 'fill_line':
        gr.fill_line()
    elif input_graf == 'scatter':
        gr.scatter()

while control == False:
    query_1 = input("Introduzca la gráfica que desea obtener: ")
    control = review_input(query_1)

query_2 = input("¿Desea guardar la gráfica? (y/n) ")

title = graf['title']
var_x = graf['var_x']
label_x = graf['label_x']
label_y = graf['label_y']
x_limit = (graf['x_limit_inf'], graf['x_limit_sup'])
y_limit = (graf['y_limit_inf'], graf['y_limit_sup'])
save = False
if query_2 == 'y':
    save = True

for func in graf['funciones']:
    label = func['name']
    var_y = func['valores']
    if func == graf['funciones'][-1]:
        finish = True
    else:
        finish = False
    gr = grafica(title = title, var_x = var_x, var_y = var_y, label_x = label_x, label_y = label_y, x_limit = x_limit, y_limit = y_limit, finish = finish, label = label, save = save)
    plotting(query_1)
print("hola mundo")