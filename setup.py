from setuptools import setup, find_packages

setup(
  name='scdibs',
  version='0.0.1',
  description='Splicing dynamics based integration of scRNAseq batches.',
  license='BSD 3-Clause License',
  packages=find_packages(),
  author = 'Revant Gupta',
  author_email = 'revant.gupta.93@gmail.com',
  url = 'https://github.com/aron0093/DIBS',
  download_url = 'https://github.com/aron0093/DIBS/archive/v_001.tar.gz',
  keywords = ['Batch integration', 'single-cell RNA sequencing', 'RNA velocity'],
  install_requires=[
      'scvelo'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ])