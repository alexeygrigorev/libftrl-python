from setuptools import setup, Distribution

# Tested with wheel v0.29.0
class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""
    def has_ext_modules(foo):
        return True


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
      url='https://github.com/alexeygrigorev/libftrl-python',
      distclass=BinaryDistribution
)
