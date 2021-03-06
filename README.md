# Excess Economics

 [![Generic badge](https://img.shields.io/badge/Python-3.8-blue.svg)](https://shields.io/)  [![Generic badge](https://img.shields.io/badge/License-MIT-green.svg)](https://shields.io/)  [![Generic badge](https://img.shields.io/badge/Maintained-Yes-red.svg)](https://shields.io/)

### Introduction

This is a repository to consider excess economic loss in the UK in response to the Covid-19 pandemic. It might be thought of as a compliament to [Excess-Deaths](https://github.com/OxfordDemSci/Excess-Deaths). It is written in Python 3, and utilizes [X01](https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/datasets/regionalemploymentbyagex01/current) data from the Office for National Statistics. It calculates Excess Economic Loss in Great Britain following the Covid-19 pandemic. It uses optimal ARIMA models based on [pmdarima](https://github.com/alkaline-ml/pmdarima)

### Prerequisites

As a pre-requisite to running this locally, you will need a working Python 3 installation with all of the necessary dependencies detailed in [requirements.txt](https://github.com/crahal/Excess-Economics/blob/master/requirements.txt) (generated by pipreqs). We strongly recommend virtual environments and [Anaconda](https://www.anaconda.com/distribution/).

### Running the Code

This code is operating system independent (through the ``os`` module) and should work on Windows, Linux and macOS all the same. To run: clone this directory, ``cd`` into the directory, and serve the project by simply running the ``main.py`` file. If updating to future variants of the X01 data, don't forget to change the string in ``main.py``!

### Structure

_data_: holds the X01 data.

_figures_: holds the figures for the main letter and for previous months too.

_letter_: A letter describing the work.

_src_: contains the code for parsing and wrangling the data, seasonally adjusting it, conducting optimal arima forecasts, and visualisation

### License

This work is free. You can redistribute it and/or modify it under the terms of the MIT license, subject to the conditions regarding the data imposed by the [EMBL-EBI license](https://www.ebi.ac.uk/about/terms-of-use). The dashboard comes without any warranty, to the extent permitted by applicable law.
