from setuptools import setup, find_packages

setup(
    name="quant-fund-sim",
    version="0.1.0",
    description="End-to-End Quant Fund Simulation",
    author="Quant Developer",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.0",
        "statsmodels>=0.14.0",
        "yfinance>=0.2.18",
        "fredapi>=0.5.0",
        "sqlalchemy>=2.0.0",
        "psycopg2-binary>=2.9.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "black>=23.0.0", "flake8>=6.0.0"],
        "dashboard": ["streamlit>=1.25.0"],
    },
)
