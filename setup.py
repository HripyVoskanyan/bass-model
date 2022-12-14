from setuptools import setup, find_packages

setup(
    author="Hripsime Voskanyan",
    description="A bass diffusion model for Marketing Analytics course",
    name="bassmodel",
    version="0.1.0",
    packages=find_packages(include=['bassmodel', 'bassmodel.*']),
    install_requires=[
        'pandas>=1.5.1',
        'numpy>=1.23.4',
        'matplotlib>=3.6.2',
        'statsmodels>=0.13.5',
        'sklearn>=0.0.post1',
        'scikit-learn>=1.2.0',
        'scipy>=1.9.3'
    ],
    python_requires='>=2.7'
)
