from setuptools import setup, find_packages

setup(
    name='dataset',
    version='0.0.1',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    author='momo',
    author_email='mo.najeeb.akbar@gmail.com',
    description='Create, write, read any dataset.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mo-najeeb-akbar/dataset.git',
    license='None',
    classifiers=[

    ]
)