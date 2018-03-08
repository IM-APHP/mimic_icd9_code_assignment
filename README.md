# RAMP starting kit on Automated text classification for ICD 9 diagnosis code assignment from MIMIC Database

Increasing large volumes of healthcare data are regularly generated in the form of Electronic Medical Records (EMR). One major issue that can be approached by capitalizing on the routinely generated textual data is the automation of diagnosis code assignment to medical notes. The task involves characterizing patient’s hospitalstay (symptoms, diagnoses, treatments, etc.) by a small number of codes, usually derived from the International Classification of Diseases (ICD). Comprehensive MIMIC database that spans more than a decade with detailed information about individual patient care offer a great opportunity for development and evaluation of automatic ICD code assignment pipelines that will ensure reproducibility of training and test methods.

Authors:

Challenge :  [Datacamp M2 Datascience](https://datascience-x-master-paris-saclay.fr/)

MIMIC dataset : [MIMIC III](https://mimic.physionet.org/)

RAMP ecosystem : [RAMP](http:www.ramp.studio)



[`ramp-worflow`](https://github.com/paris-saclay-cds/ramp-workflow) for more help on the [RAMP](http:www.ramp.studio) ecosystem.

Install ramp-workflow (rampwf), then execute

```
ramp_test_submission
```

to test the starting kit submission (`submissions/starting_kit`) and

```
ramp_test_submission --submission=starting_kit
```

to test `starting_kit` or any other submission in `submissions`.
