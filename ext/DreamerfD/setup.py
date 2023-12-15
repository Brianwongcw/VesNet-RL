import setuptools
import pathlib


setuptools.setup(
    name='dreamer_fd',
    version='2.2.0',
    description='Dreamer from Demonstration',
    long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    packages=['dreamer_fd', 'dreamer_fd.common'],
    install_requires=[
        'ruamel.yaml',
        'tensorflow', 
        'tensorflow_probability',
        'pandas',
        ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
