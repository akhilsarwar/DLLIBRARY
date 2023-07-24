from setuptools import find_packages, setup
setup(
    name='dllibrary',
    packages=find_packages(include=['dllibrary']),
    version='0.1.0',
    description='Deep Learning Library',
    author='akhil sarwar t h',
    license='MIT',
    install_requires=['numpy', ]
)