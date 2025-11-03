from setuptools import setup, find_packages

setup(
    name='dataset',
    version='0.1.0',
    packages=find_packages(),
    package_data={
        'dataset': ['py.typed'],
    },
    install_requires=open('requirements.txt').read().splitlines(),
    author='momo',
    author_email='mo.najeeb.akbar@gmail.com',
    description='A lightweight TFRecord dataset writing and loading library',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mo-najeeb-akbar/dataset.git',
    license='None',
    python_requires='>=3.10',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Typing :: Typed',
    ],
    zip_safe=False,
)