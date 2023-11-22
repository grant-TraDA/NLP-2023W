# NLP-BAMK-project

## What is it?

This repository contains the source code, data sets and documentation for project 1 for the NLP course, MiNI PW 2023.

Authors:

- Anish Gupta
- Martyna Majchrzak
- Bartosz Rożek
- Konrak Welkier

## Structure of the project

### Codes

This folder contains the source code for the project. There are three main python files which holds the structure of the project:

- data_structures.py
- overall_anotator.py
- aspect_anotator.py

There are also additional .ipynb files which contain example of usage and EDA of the data sets.

#### data_structures.py

This files creates datatypes for the both annotations - "ordinary" and aspect based. It also contains classes used for storing results of the annotators.

#### overall_anotator.py

This file contains class OverallAnnotator which implements many different SOTA tools used for sentiment analysis. The main method is _annotate_ which assign labels based on the selected tool. Methods calculate_results and test_annotator are common for all tools and are based on the unified output of all tools. The first method calculates the results when predicted and true labels are provided, the second method uses the previously specified tool to generate the predicted labels and then calculate results.

#### aspect_anotator.py

This file contains class AspectAnnotator which implements many different SOTA tools used for aspect-based sentiment analysis. It makes two aproaches possible - one step (extracting aspect and calculate sentiment with one tool) and two step (extract aspect with one tool and calculating sentiment with another). The main method is _annotate_ which assign labels based on the selected tools. Methods calculate_results and test_annotator are common for all tools and are based on the unified output of all tools. The first method calculates the results when predicted and true labels are provided, the second method uses the previously specified tool to generate the predicted aspects and labels and then calculate results.

### Data

This folder contains data sets which can be used to test annotators.

### External

This folder contains external files which are needed to use some external tools. Currently it contains files needed to use SentiStrength.

### Requirements.txt

This file contains the required package dependencies for this project. Python 3.11.3 was used.
