import sys
import os
from setuptools import setup, find_packages  
 
setup(  name = "kratosbat",  
        version = "1.0",  
        keywords = ("capacity", "volume change","prediction"),  
        description = "kratos battery predictiion",  
        long_description = "eds sdk for python",  
        license = "MIT Licence",  
   
        url = "https://github.com/kratos-batteries/kratos-batteries",  
        author = "Jizhou Liu, Mitchell Kitt, Yousef Baioumy, Andrew Gonsalve, Lester Jiang",
        author_email = "kittm@uw.edu, jizhol@uw.edu",  
        maintainer = "Yousef Baioumy",
        maintainer_email = "baioumyy@uw.edu",


        classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # These classifiers are *not* checked by 'pip install'. See instead
        # 'python_requires' below.

        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        ],
        packages = find_packages('kratosbat',exclude=['tests']),  
        include_package_data = True,  
        platforms = "any",         
        install_requires = [ 'pandas',
                             'sklearn',
                             'numpy',
                             'matplotlib',
                             'pytorch'
                            ],
        entry_points = {  
             'console_scripts': [  
                  'kratos_run = kratosbat.kratos_run:main',
        ]  
     }  
 )