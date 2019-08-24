# Scanning White Light Interferometry

PURPOSE

This project makes use of white light interference patterns (fringes) to deduce the surface elevation profile of a specimen. These fringes are created using a Michelson interferometer which consists of two optical paths: beamsplitter-mirror-beamsplitter (reference arm) and beamsplitter-specimen-beamsplitter (sample arm). Interference patterns are photographed by a camera, while a piezo actuator attached to the mirror holder changes the reference optical path sequentially.

IMPLEMENTATION

The above sequential image acquisition scheme requires a synchronization between the camera and the piezo actuator. This is managed by a Arduino microcontroller indirectly driving the piezo actuator with code written in C/C++ and a computer directly controlling the camera and the Arduino with code written in Python via USB cables.

PACKAGE CODES

Python

This package offers two ways of carrying out measurement and analysis of a specimen:
  1) Via an integrated GUI that sets up the camera configurations, calibrates the piezo actuator, displaying specimen images live, acquires image sets and produces the surface elevation profile of the specimen. It also contains some auxiliary functionalities for the setting of the interferometer arms.
  2) Each aforementioned step can be executed separately via the corresponding python scripts. (See file descriptions)
  
  C/C++
  
  There are two different Arduino codes: one used for oscillating the piezo actuator back and forth for fringe viewing purposes only and one used for the measurement process.
