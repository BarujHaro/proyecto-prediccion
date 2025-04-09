document.addEventListener('DOMContentLoaded', function () {
    const form = document.querySelector('form');
    const modal = document.getElementById('resultModal');
    const resultText = document.getElementById('resultText');
    const closeButton = document.querySelector('.close-button');

    function showModal(message) {
        resultText.innerHTML = message;
        modal.style.display = 'flex';
    }

    function hideModal() {
        modal.style.display = 'none';
    }

    closeButton.addEventListener('click', hideModal);
    modal.addEventListener('click', function(event) {
        if (event.target === modal) hideModal();
    });

    form.addEventListener('submit', async function (event) {
        event.preventDefault();
        
        // Mostrar estado de carga
        const submitBtn = form.querySelector('button[type="submit"]');
        submitBtn.disabled = true;
        submitBtn.textContent = 'Procesando...';

        try {
            // Crear objeto con los datos del formulario
            const formData = {
                anno: parseInt(form.querySelector('#anno').value),
                activos: parseFloat(form.querySelector('#activos').value),
                depreciacion: parseFloat(form.querySelector('#depreciacion').value),
                inventario: parseFloat(form.querySelector('#inventario').value),
                ingresos_netos: parseFloat(form.querySelector('#ingresos_netos').value),
                total_cuentas_por_cobrar: parseFloat(form.querySelector('#total_de_cuentas_por_cobrar').value),
                valor_del_mercado: parseFloat(form.querySelector('#valor_del_mercado').value),
                deuda_a_largo_plazo: parseFloat(form.querySelector('#deuda_a_largo_plazo').value),
                beneficio_bruto: parseFloat(form.querySelector('#beneficio_bruto').value),
                pasivos_corrientes_totales: parseFloat(form.querySelector('#pasivos_corrientes_totales').value),
                ganancias_retenidas: parseFloat(form.querySelector('#ganancias_retenidas').value)
            };

            const response = await fetch('/form', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                throw new Error(`Error HTTP: ${response.status}`);
            }

            const result = await response.json();
            
            if (result.prediccion === undefined) {
                throw new Error('La respuesta no contiene predicción');
            }

            const prediction = result.prediccion === 1 ? 
                'Empresa fuera de riesgo' : 'Empresa dentro de riesgo';
            
            showModal(`<strong>Resultado:</strong> ${prediction}`);

        } catch (error) {
            console.error('Error completo:', error);
            showModal(`<strong>Error:</strong> ${error.message}`);
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Predecir';
        }
    });
});