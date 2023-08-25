from setuptools import setup

setup(
    name='NeuroSurgeon',
    version='0.1.0',    
    description='A toolkit for subnetwork analysis',
    url='https://github.com/mlepori1/NeuroSurgeon',
    author='Michael Lepori',
    author_email='michael_lepori@brown.edu',
    license='MIT',
    packages=['NeuroSurgeon'],
    install_requires=['torch>=2.0.1',
                      'torchvision>=0.15.2',
                      'torchaudio>=2.0.2',
                      'transformers',
                      'datasets',
                      'matplotlib',
                      ],

    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.9',
    ],
)