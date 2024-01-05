# NLP Project Group 7 -- Analysis of questions
*Authors: Mikołaj Malec, Marceli Korbin, Kacper Grzymkowski, Jakub Fołtyn*

This project tackled the problem of analysing questions based on their topic and complexity level. 
It is also related to the topic of creativity: our goal was to determine which questions are more creative than others.

## Folder structure
* `eda/` -- folder containing files related to the Exploratory Data Analysis computed as part of the PoC. Files include:
  * `qg/train.txt.target.txt` -- Standford Questions and Answers Dataset.
  * `embedding.ipynb` -- EDA computed for PoC.
  * `labels.json` -- labels provided from LLM model (`NLP-GPU-models.ipynb`).
* `Analysis of questions1.pdf` -- presentation for project proposal.
* `NLP-GPU-models.ipynb` -- notebook containing PoC LLM prompt engineering tests.
* `NLP_-example_clastering.ipynb` -- notebook containg preliminary clustering for PoC.
* `NLP_Project_Report___Questions-2.pdf` -- literature review report.
* `project_final/` -- folder containg files for the project final. Files include:
  * `squad.txt` -- Standford Questions and Answers Dataset.
  * `computing_complexity_metrics.R` -- file for computing various text complexity metrics.
  * `NLP_Group_7_final_report.pdf` -- final report.
  * `analysis.ipynb` -- notebook containing the entirety of our project.
