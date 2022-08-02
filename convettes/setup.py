from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Education',
    'Operating System :: Microsoft :: Windows :: Windows 10',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3'
]

setup(
    name='convettes',
    version='0.0.3',
    description="a library for building your own convolutional neural networks",
    long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
    url='',
    author='sairbarbaros(Barbaros Şair)',
    requires=('numpy', 'tensorflow'),
    author_email='sairbarbaros@gmail.com',
    license='MIT',
    classifiers=classifiers,
    keywords='convolutional neural networks',
    packages=find_packages(),
    install_requires=['']

)