document.addEventListener('DOMContentLoaded', function () {
    const form = document.querySelector('form');
    const modal = document.getElementById('resultModal');
    const resultText = document.getElementById('resultText');
    const closeButton = document.querySelector('.close-button');

    function showModal(message, isSuccess) {
        resultText.innerHTML = message;
        resultText.style.color = isSuccess ? 'green' : 'red';
        modal.style.display = 'flex';
    }

    function hideModal() {
        modal.style.display = 'none';
    }

    closeButton.addEventListener('click', hideModal);
    modal.addEventListener('click', function (event) {
        if (event.target === modal) hideModal();
    });

    form.addEventListener('submit', async function (event) {

        event.preventDefault();


                // Obtener todos los valores numéricos del formulario
        const inputs = Array.from(form.querySelectorAll('input[type="number"]'));
        const values = inputs.map(input => parseFloat(input.value));
                
                // Validación 1: Todos los valores son iguales
        const allEqual = values.every(val => val === values[0]);

        if (allEqual) {

            showModal('<strong>Error:</strong> Todos los valores no pueden ser idénticos. Por favor ingrese datos realistas.', false);
            return;
            }

        // Mostrar estado de carga
        const submitBtn = form.querySelector('button[type="submit"]');
        submitBtn.disabled = true;
        submitBtn.textContent = 'Procesando...';

        try {
            // Crear objeto con los datos del formulario
            const formData = {
                
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

            const response = await fetch('/', {
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
            const isSuccess = result.prediccion === 1;
            showModal(`<strong>Resultado:</strong> ${prediction}`, isSuccess);

        } catch (error) {
            console.error('Error completo:', error);
            showModal(` < strong > Error: < /strong> ${error.message}`);
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Predecir';
        }
    });
});