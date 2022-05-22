<!-- dataset description (variables and how they are measured)-->
<!-- how this prediction model is put to use i.e. what it's used for -->
<!-- why it's been talked about so much -->
<!-- what this analysis is meant to achieve -->

## EDA

 Recidivism split for each race:
 

![](/Analysis/COMPAS_recidivism_race/EDA/racebyrecid.png)

Distribution of prior arrests in the dataset:

| ![](/Analysis/COMPAS_recidivism_race/EDA/priors.png) |Out of a total of 7214 individuals in COMPAS criminal risk assessment dataset published by Propublica, 3696 (51%) were African-American, 2454 (34%) were Caucasian , and 1064 (15%) belong to other races including Hispanic, Asian and Native Americans.
51% of the African-Americans were recorded for two year recidivism. For caucasians, this is 39% and for other races, 36%.  |


Distribution of prior arrests for each race:

![](/Analysis/COMPAS_recidivism_race/EDA/priorsbyrace.png)

Prior arrests (categorical) distribution:

![](/Analysis/COMPAS_recidivism_race/EDA/priorscatbyrace.png)

Prior arrests (categorical) distribution for only the recidivating individuals within each race:

![](/Analysis/COMPAS_recidivism_race/EDA/priorscatbyracerecid.png)


Prior arrests (categorical) distribution for only the non-recidivating individuals within each race:

![](/Analysis/COMPAS_recidivism_race/EDA/priorscatbyracenonrecid.png)

 Recidivism status breakdown by prior arrests (categorical):

![](/Analysis/COMPAS_recidivism_race/EDA/priorscatbyrecid.png)


## Classification and fairness metrics


 | COMPAS scores  | Random Forest predictions |
| ------------- | ------------- |
| ![](/Analysis/COMPAS_recidivism_race/METRICS/COMPASclassmetrics.png) |![](/Analysis/COMPAS_recidivism_race/METRICS/RFclassmetrics.png)  |
| ![](/Analysis/COMPAS_recidivism_race/METRICS/COMPASconfusionmatrix.png) |![](/Analysis/COMPAS_recidivism_race/METRICS/RFconfusionmatrix.png)  |
| ![](/Analysis/COMPAS_recidivism_race/METRICS/COMPASfairnessmetrics.png) |![](/Analysis/COMPAS_recidivism_race/METRICS/RFfairnessmetrics.png)  |
