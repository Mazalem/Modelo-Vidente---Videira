import streamlit as st
import gdown
import tensorflow as tf

def main():
    st.set_page_config(
        page_title="Classificador de Folhas de Videiras"
    )
    st.write("# Classifica Folhas de Videiras!")

    #carregar modelo
    def carrega_modelo():
        # https://drive.google.com/file/d/1GpWix8dp6FeFAs6g0etbnw_avu9Aflfp/view
        url = "https://drive.google.com/uc?id=1GpWix8dp6FeFAs6g0etbnw_avu9Aflfp/"

        # baixa o modelo
        gdown.down(url, 'modelo_quantizado16bits.tflite')

        #carrega o modelo
        interpreter = tf.lite.Interpreter(model_path='modelo_quantizado16bits.tflite')

        #disponibiliza pra uso
        interpreter.allocate_tensors()
        return interpreter

    #carregar imagem


    #classificar imagem


if __name__ == "__main__":
    main()