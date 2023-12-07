from setuptools import setup, find_packages

setup(
    name='skin_classification_cv', 
    version='0.1', 
    packages=find_packages(), 
    install_requires=[
        'numpy', 
        'pandas',
        'scipy',
        'matplotlib'
    ]
)