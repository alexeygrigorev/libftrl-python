from setuptools import setup

setup(name='ftrl',
      version='0.0.1',
      description="LibFTRL Python Package",
      long_description="LibFTRL Python Package",
      install_requires=[
          'numpy',
          'scipy',
      ],
      maintainer='Alexey Grigorev',
      maintainer_email='',
      zip_safe=False,
      packages=['ftrl'],
      package_data={'ftrl': ['libftrl.so']},
      include_package_data=True,
      license='WTFPL',
      url='https://github.com/alexeygrigorev/libftrl-python'
)