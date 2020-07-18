import setuptools

#with open("README.md", "r") as fh:

#    long_description = fh.read()

setuptools.setup(

     name='compressionNN',

     version='0.1',

     packages=['compressionNN',
                'compressionNN.test'],

     scripts=['compressionNN/pruning.py', 'compressionNN/weightsharing.py', 'compressionNN/pruning_weightsharing.py', 'compressionNN/huffman.py', 'compressionNN/sparse_huffman.py', 'compressionNN/sparse_huffman_only_data.py', 'compressionNN/stochastic.py'] ,

     author="Giosuè Marinò, Gregorio Ghidoli",

     author_email="giosumarin@gmail.com, gregorio.ghidoli@studenti.unimi.it",

     description="Compress neural networks",

     #long_description=long_description,

   #long_description_content_type="text/markdown",

     url="https://github.com/giosumarin/NeuralNetworkCompression",

     #packages=setuptools.find_packages(),

     classifiers=[

         "Programming Language :: Python :: 3",

         "License :: GPL-3.0",

         "Operating System :: Linux base",

     ],

 )
