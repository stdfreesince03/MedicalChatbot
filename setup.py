from setuptools import setup,find_packages

setup(
    name='medical-chatbot-rag',
    version='0.1.0',
    author='Winata Tristan',
    author_email='winatatristan04@gmail.com',
    packages=find_packages(where="src"),
    package_dir={"":"src"},
    install_requires=[]
)