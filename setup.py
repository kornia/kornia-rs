from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension

setup(name='kornia_io',
      version='0.1.0',
      description='Kornia input and output wrapper.',
      author='Kornia.org',
      url='https://github.com/kornia/kornia_rs',
      extras_require={
          'dev': [
              'kornia',
              'pytest',
          ]
      },
      packages=find_packages(),
      # configure rust
      rust_extensions=[RustExtension('kornia_rs', binding=Binding.PyO3)],
      # rust extensions are not zip safe, just like C-extensions.
      zip_safe=False,
      )
