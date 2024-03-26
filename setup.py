from setuptools import setup, find_packages

setup(
    name='image-annotation-database',
    version='0.0.0',
    author='Peter OConnor',
    author_email='peter.ed.oconnor@gmail.com',
    description='Some methods for saving/loading annotated images in a standardized format - as image files with metadata.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/petered/image-annotation-database',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tinydb',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
