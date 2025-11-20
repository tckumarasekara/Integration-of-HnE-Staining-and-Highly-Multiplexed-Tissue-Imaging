from setuptools import setup, find_packages

setup(
    name="he2multi_reg",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy==2.2.6",
        "tifffile==2025.5.10",
        "histomicstk==1.4.0",
        "scikit-image==0.25.2",
        "scipy==1.16.3",
        "itk==5.4.4.post1",
        "itk-elastix==0.23.0",
        "typer[all]==0.20.0",
    ],
    python_requires=">=3.11",
    entry_points={
        "console_scripts": [
            "he2multi-reg=registration.reg_cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    author="Thusheera Kumarasekara",
    author_email="tckumarasekara@gmail.com",
    description="Tools for image registration between multiplexed and HnE stained tissue images",
    #url="https://github.com/tckumarasekara/", 
)