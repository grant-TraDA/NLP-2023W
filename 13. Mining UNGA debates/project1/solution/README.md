# Solution - Team 13. UNGA debates
Mateusz Grzyb, Mateusz Krzyziński, Bartłomiej Sobieski, Mikołaj Spytek


## Where is our solution available?
- The application created during this project is available in the *Docker* image at [https://hub.docker.com/krzyzinskim/nlp-unga-debates](https://nlp-unga-debates-g2o6gnzttq-lm.a.run.app/). To run it, you simply need to run the docker container with appropriate port forwarding.
- The application is also hosted at [https://nlp-unga-debates-g2o6gnzttq-lm.a.run.app/](https://nlp-unga-debates-g2o6gnzttq-lm.a.run.app/), at least until we run out of free credits at *Google Cloud Platform*.
- The pickle binary file with processed dataset (used in the app) is available at [https://drive.google.com/drive/folders/16a-Woz3FoPsXd77MsIyIMZkKlwSEIkox?usp=share_link](https://drive.google.com/drive/folders/16a-Woz3FoPsXd77MsIyIMZkKlwSEIkox?usp=share_link). To run app properly, it should be placed in the main project directory.

## Folder structure:

```
solution/
├─ analyses - this folder contains notebooks with the performed analyses using the BERTopic model
├─ app - this folder contains the source code for the interactive application which allows for exploring the final results of this project
├─ dataset - this folder contains the **datasets** used in the project - the previously collected United Nations General Asembly statements, enriched with the this year's session which was scraped by us, and the dataset after cleaning 
├─ metadata - this folder contains the collected **additional metadata** and scripts necessary for its processing and joining to the original dataset
├─ metrics - this folder contains the code and results of ablation studies - comparing the BERTopic model with LDA baseline
├─ modeling - this folder contains the source code used for creating the BERTopic models
├─ models - this folder contains the **trained model weights** for all models considered in the project
├─ preprocessing - this folder contains scripts used for preprocessing the texts before feeding them to the models
├─ scrapping - this folder contains scripts used for downloading this year's statements and extracting them from pdfs
```


Additionally, this folder contains the `requirements.txt` file, which contains all the package versions necessary for **reproducing** the results of this project.