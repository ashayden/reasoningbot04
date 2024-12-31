from setuptools import setup, find_packages

setup(
    name="streamlit-depth-slider",
    version="0.1.0",
    author="MARA Team",
    author_email="",
    description="A custom depth slider component for Streamlit",
    long_description="",
    long_description_content_type="text/plain",
    url="",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[],
    python_requires=">=3.7",
    install_requires=[
        "streamlit >= 0.63",
    ]
) 