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

clases_teachable = ['Fresa', 'Naranja', 'Piña', 'Otros', 'Cebolla', 'Tomate', 'Zanahoria', 'Pepino', 'Puerro', 'Apio', 'Pan', 'Salmón', 'Pollo']

# Lista para almacenar los alimentos detectados
if "alimentos_detectados" not in st.session_state:
    st.session_state["alimentos_detectados"] = []

def preprocess_input_teachablemachine(img):
    img = img / 255.0
    return img

def detectar_alimentos(frame):
    img_resized = cv2.resize(frame, (224, 224))
    x_img = np.expand_dims(img_resized, axis=0)
    x_img = preprocess_input_teachablemachine(x_img)

    interpreter.set_tensor(input_details[0]['index'], x_img.astype(np.float32))
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    max_prob = np.max(preds)
    prediccion_clase = np.argmax(preds)
    nombre_clase = clases_teachable[prediccion_clase]

    if max_prob > 0.8 and nombre_clase not in st.session_state["alimentos_detectados"]:
        st.session_state["alimentos_detectados"].append(nombre_clase)

    return nombre_clase, max_prob

def obtener_recetas_con_cohere(alimentos):
    prompt = f"Tengo los siguientes ingredientes: {', '.join(alimentos)}. ¿Puedes sugerirme algunas recetas saludables que pueda hacer con ellos?"
    response = co.generate(model='command-xlarge-nightly', prompt=prompt, max_tokens=1000, temperature=0.7)
    return response.generations[0].text.strip()

# Configurar la interfaz de Streamlit
st.title("Detección de Alimentos en Tiempo Real")
st.write("Apunta la cámara a los alimentos para detectarlos y obtener recetas.")

# Iniciar cámara con OpenCV
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
