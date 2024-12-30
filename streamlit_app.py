import streamlit as st

st.title("Hello World")
st.write("This is a test app")

name = st.text_input("What's your name?")
if name:
    st.write(f"Hello, {name}!") 