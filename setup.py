from setuptools import setup, find_packages
#from distutils.core import setup

setup(
  name = 'prl',
  packages = find_packages(exclude=['build', '_docs', 'templates']),
  version = '0.94b',
  install_requires=[
        "numpy",
        "scipy",
        "cvxopt",
        "sklearn"
  ],
  license = "MIT",
  description = '[P]reference and [R]ule [L]earning algorithm implementation',
  author = 'Mirko Polato',
  author_email = 'mak1788@gmail.com',
  url = 'https://github.com/makgyver/PRL',
  download_url = 'https://github.com/makgyver/PRL',
  keywords = ['preference-learning', 'game-theory', 'machine-learning', 'algorithm'],
  classifiers = [
                 'Development Status :: 4 - Beta',
                 'Programming Language :: Python :: 3.6',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'License :: OSI Approved :: MIT License',
                ]
)
