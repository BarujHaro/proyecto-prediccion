from flask import Blueprint, render_template, request, jsonify
from RF import obtener_metricas, model, prediccion, obtener_parametros_modelo, explicar_fallo

views = Blueprint(__name__, "views")

@views.route("/", methods=["GET", "POST"])
def show_index():
    pred = None
    if request.method == 'POST':
        try:
            # Manejar tanto JSON como form-data
            if request.is_json:
                data = request.get_json()
            else:
                data = request.form
            
            datos_usuario = {
                'activos': float(data.get('activos')),
                'depreciacion': float(data.get('depreciacion')),
                'inventario': float(data.get('inventario')),
                'ingresos netos': float(data.get('ingresos_netos') or data.get('ingresos netos')),
                'total de cuentas por cobrar': float(data.get('total_cuentas_por_cobrar') or data.get('total de cuentas por cobrar')),
                'valor del mercado': float(data.get('valor_del_mercado') or data.get('valor del mercado')),
                'deuda a largo plazo': float(data.get('deuda_a_largo_plazo') or data.get('deuda a largo plazo')),
                'beneficio bruto': float(data.get('beneficio_bruto') or data.get('beneficio bruto')),
                'pasivos corrientes totales': float(data.get('pasivos_corrientes_totales') or data.get('pasivos corrientes totales')),
                'ganancias retenidas': float(data.get('ganancias_retenidas') or data.get('ganancias retenidas'))
            }
            
            pred = prediccion(datos_usuario)
            
            mensaje = explicar_fallo(datos_usuario)
            
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({
                    'prediccion': int(pred[0]),
                    'status': 'success',
                    'mensaje': mensaje
                })
                
        except Exception as e:
            print(str(e))
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({
                    'error': str(e),
                    'status': 'error'
                }), 400
            raise


    parametros = obtener_parametros_modelo(model)
    metricas = obtener_metricas()
    return render_template("index.html", 
                         parametros=parametros,
                         metricas=metricas,
                         ruta_ma='static/cm.png',
                         pred=pred,
                         arbol='static/arbol.png')