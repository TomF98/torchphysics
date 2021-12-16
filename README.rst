==============
TorchPhysics
==============

TorchPhysics is a Python library of deep learning methods for solving differential equations.
You can use TorchPhysics to: 

- solve ordinary and partial differential equations via physics-informed neural networks [#]_ (PINN) 
  or the Deep Ritz method [#]_
- train a neural network to approximate solutions for different parameters
- solve inverse problems and interpolate external data via the above methods

TorchPhysics is build upon the machine learning library PyTorch_. 

.. _PyTorch: https://pytorch.org/

Features of TorchPhysics
========================
The Goal of this library is to create a basic framework that can be used in many
different applications and with different deep learning methods.
To this end, TorchPhysics aims at a:

- modular and expandable structure
- easy to understand code and clean documentation
- intuitive and compact way to transfer the mathematical problem into code
- reliable and well tested code basis 

Some build in features are:

- mesh free domain generation. With pre implemented domain types: 
  *Point, Interval, Parallelogram, Circle, Triangle and Sphere*
- loading external created objects, thanks to a soft dependency on trimesh_  
  and Shapely_
- creating complex domains with the boolean operators *Union*, *Cut* and *Intersection* 
  and higher dimensional objects over the Cartesian product
- allowing interdependence of different domains, e.g. creating moving domains
- different point sampling methods for every domain:
  *RandomUniform, Grid, Gaussian, Latin hypercube, Adaptive* and some more for specific domains
- different operators to easily define a differential equation
- pre implemented fully connected neural network and easy implementation
  of additional model structures 
- sequentially or parallel evaluation/training of different neural networks
- normalization layers and adaptive weights [#]_ to speed up the trainings process
- powerful and versatile training thanks to `PyTorch Lightning`_
  - many options for optimizers and learning rate control
  - monitoring loss of individual conditions while training 


.. _trimesh: https://github.com/mikedh/trimesh
.. _Shapely: https://github.com/shapely/shapely
.. _`PyTorch Lightning`: https://www.pytorchlightning.ai/


Getting Started
===============
To learn the functionality and usage of TorchPhysics we recommend
to have a look at the following sections:

- `Tutorial: Understanding the structure of TorchPhysics`_
- `Examples: Different applications with detailed explanations`_
- Documentation_

.. _`Tutorials: Understanding the structure of TorchPhysics`: does_not_exist_yet
.. _`Examples: Different application problems with detailed explanations`: examples
.. _Documentation: does_not_exist_yet

Note
====

This project has been set up using PyScaffold 4.0.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.


License
=======
TorchPhysics uses a Apache License, see the LICENSE_ file.

.. _LICENSE: LICENSE.txt


Bibliography
============
.. [#] A numerical footnote
.. [#] A numerical footnote ritz
.. [#] A numerical footnote weights