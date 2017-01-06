from setuptools import setup

setup(name='rf_perm_feat_import',
      version='0.1',
      description='Random Forest Permutate Feature Importance',
      url='https://github.com/pjh2011/sklearn_perm_feat_import',
      author='Peter Hughes',
      author_email='pethug210@gmail.com',
      license='MIT',
      packages=['rf_perm_feat_import'],
      install_requires=[
          'numpy',
          'sklearn'
      ],
      zip_safe=False)
