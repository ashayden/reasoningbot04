import streamlit as st

st.title("Test App")
st.write("Hello World!")

name = st.text_input("Enter your name")
if name:
    st.write(f"Hello {name}!") 