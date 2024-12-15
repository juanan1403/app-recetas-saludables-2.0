import cv2
import numpy as np
import tensorflow as tf
import cohere
import streamlit as st

# Inicializar Cohere
cohere_api_key = 'JT3Oiog1cwFuKZotUAh3Qwi4UUrKAmoKT8uepSCp'  # Reemplaza con tu API key
co = cohere.Client(cohere_api_key)

# Cargar el modelo TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path='model_unquant.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Cargar labels
clases_teachable = ['Fresa', 'Naranja', 'Piña', 'Otros', 'Cebolla', 'Tomate', 'Zanahoria', 'Pepino', 'Puerro', 'Apio', 'Pan', 'Salmón', 'Pollo']
colores = {'Fresa': (255, 165, 0), 'Naranja': (255, 0, 255), 'Piña': (255, 255, 0), 'Cebolla': (0, 165, 255), 'Tomate': (255, 165, 255),
           'Zanahoria': (165, 165, 0), 'Pepino': (0, 255, 0), 'Puerro': (100, 180, 0), 'Apio': (255, 255, 255), 'Pan': (0, 165, 165),
           'Salmón': (0, 165, 200), 'Pollo': (165, 255, 255)}

# Lista para almacenar los alimentos detectados
alimentos_detectados = []

def preprocess_input_teachablemachine(img):
    img = img / 255.0  # Normalizar los valores de los píxeles entre 0 y 1
    return img

# Función para detectar alimentos en un solo frame o en divisiones de la imagen
def detectar_alimentos(frame):
    global alimentos_detectados

    # Redimensionar la imagen completa a 224x224 píxeles para la primera detección
    img_resized = cv2.resize(frame, (224, 224))
    x_img = np.expand_dims(img_resized, axis=0)
    x_img = preprocess_input_teachablemachine(x_img)

    # Hacer la predicción con la imagen completa redimensionada
    interpreter.set_tensor(input_details[0]['index'], x_img.astype(np.float32))
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    max_prob = np.max(preds)
    prediccion_clase = np.argmax(preds)
    nombre_clase = clases_teachable[prediccion_clase]

    # Verificar si la probabilidad supera el umbral del 80%
    if max_prob > 0.8 and nombre_clase in colores:
        if nombre_clase not in alimentos_detectados:
            alimentos_detectados.append(nombre_clase)
        return nombre_clase, max_prob

    # Si no detecta nada en la imagen general, realizar la detección en subdivisiones de 224x224 píxeles
    step_size = 224
    for y in range(0, frame.shape[0] - step_size + 1, step_size):
        for x in range(0, frame.shape[1] - step_size + 1, step_size):
            sub_img = frame[y:y + step_size, x:x + step_size]
            x_sub_img = np.expand_dims(sub_img, axis=0)
            x_sub_img = preprocess_input_teachablemachine(x_sub_img)

            interpreter.set_tensor(input_details[0]['index'], x_sub_img.astype(np.float32))
            interpreter.invoke()
            sub_preds = interpreter.get_tensor(output_details[0]['index'])[0]
            sub_max_prob = np.max(sub_preds)
            sub_prediccion_clase = np.argmax(sub_preds)
            sub_nombre_clase = clases_teachable[sub_prediccion_clase]

            if sub_max_prob > 0.8 and sub_nombre_clase in colores:
                if sub_nombre_clase not in alimentos_detectados:
                    alimentos_detectados.append(sub_nombre_clase)
                return sub_nombre_clase, sub_max_prob  # Retornar si se encuentra en alguna subdivisión

    return None, 0  # Retornar nada si no se detecta ningún alimento en todo el frame


def obtener_recetas_con_cohere(alimentos):
    prompt = f"Tengo los siguientes ingredientes: {', '.join(alimentos)}. ¿Puedes sugerirme algunas recetas saludables que pueda hacer con ellos?"
    response = co.generate(model='command-xlarge-nightly', prompt=prompt, max_tokens=1000, temperature=0.7)
    return response.generations[0].text.strip()

# Configurar la interfaz de Streamlit
st.title("Detección de Alimentos en Tiempo Real")
st.write("Apunta la cámara a los alimentos para detectarlos y obtener recetas.")

# Inicializar la lista de alimentos detectados en el estado de sesión
if "alimentos_detectados" not in st.session_state:
    st.session_state["alimentos_detectados"] = []

# Crear un marcador para detener o iniciar el video
run = st.checkbox("Iniciar Cámara")

frame_placeholder = st.empty()

if run:
    cap = cv2.VideoCapture(0)  # Capturar desde la cámara
    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("No se pudo acceder a la cámara.")
            break

        # Detección de alimentos en el frame
        nombre_clase, prob = detectar_alimentos(frame)
        if nombre_clase and prob > 0.8:
            # Agregar a la lista si no está ya presente
            if nombre_clase not in st.session_state["alimentos_detectados"]:
                st.session_state["alimentos_detectados"].append(nombre_clase)

            # Dibujar el rectángulo y texto en el frame
            color = (0, 255, 0)  # Verde para el rectángulo
            cv2.putText(frame, f'{nombre_clase} ({prob:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, frame.shape[0] - 10), color, 2)

        # Mostrar el video en la interfaz
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")

    cap.release()

# Mostrar alimentos detectados
st.write("### Alimentos Detectados:")
st.write(", ".join(st.session_state["alimentos_detectados"]) if st.session_state["alimentos_detectados"] else "Ninguno")

# Botón para obtener recetas
if st.button("Obtener Recetas"):
    if st.session_state["alimentos_detectados"]:
        with st.spinner("Generando recetas..."):
            recetas = obtener_recetas_con_cohere(st.session_state["alimentos_detectados"])
            st.write("### Recetas Saludables:")
            st.text(recetas)
    else:
        st.warning("No se han detectado alimentos aún.")