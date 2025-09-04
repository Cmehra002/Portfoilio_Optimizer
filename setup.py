from setuptools import setup, find_packages

setup(
    name='portfolio_optimizer',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'yfinance',
        'cvxpy',
        # Add any other dependencies here
    ],
    author='Faiz Nazeer',
    description='SDK for financial data handling and portfolio optimization',
    license='MIT',
)
