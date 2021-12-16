==============
TorchPhysics
==============


TorchPhysics is a Python library of deep learning methods for solving differential equations.
You can use TorchPhysics to: 

- solve ordinary and partial differential equations via physics-informed neural networks (PINN) 
  or the Deep Ritz method
- train a neural network to approximate solutions for different parameters
- solve inverse problems and interpolate external data via the above methods

TorchPhysics is build upon the machine learning backend PyTorch_. 
.. _PyTorch: https://pytorch.org/

Features of TorchPhysics
========================

A longer description of your project goes here...

In build features are:

- mesh free domain generation. With pre implemented domain types: 
  *Point, Interval, Parallelogram, Circle, Triangle and Sphere*
- loadning external created objects, thanks to a soft dependency on trimesh_  
  and Shapely_
- creating complexer domains with the boolean operators *Union*, *Cut* and *Intersection* 
  and higher dimensional objects over the cartesian product
- allowing interdependence of different domains, for example to create moving domains
  in time 
- different point sampling methods for every domain:
  *RandomUnifom, Grid, Gaussian, Latin hypercube, Adaptive* and more


.. _trimesh: https://github.com/mikedh/trimesh
.. _Shapely: https://github.com/shapely/shapely

Getting Started
===============
To learn the functionality and usage of TorchPhysics we recommend
to have a look at the following sections:

- Tutorials: Understanding the structure of TorchPhysics
- `Examples: Different application problems with detailed explanations`_
- Documentation
 
.. _`Examples: Different application problems with detailed explanations`: examples


Note
====

This project has been set up using PyScaffold 4.0.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.


License
=======
TorchPhysics uses a Apache License, see the LICENSE_ file.

.. _LICENSE: LICENSE.txt