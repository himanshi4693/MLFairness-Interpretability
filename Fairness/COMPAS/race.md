<!-- dataset description (variables and how they are measured)-->
<!-- how this prediction model is put to use i.e. what it's used for -->
<!-- why it's been talked about so much -->
<!-- what this analysis is meant to achieve -->

## EDA

 Recidivism split for each race:
 
 | First Header  | Second Header |
| ------------- | ------------- |
| ![](/Analysis/COMPAS_recidivism_race/EDA/racebyrecid.png)  | Description...  |

![](/Analysis/COMPAS_recidivism_race/EDA/racebyrecid.png)

Distribution of prior arrests in the dataset:

![](/Analysis/COMPAS_recidivism_race/EDA/priors.png)

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

Classification metrics for COMPAS scores:

![](/Analysis/COMPAS_recidivism_race/METRICS/COMPASclassmetrics.png)

Confusion matrix for COMPAS scores:

![](/Analysis/COMPAS_recidivism_race/METRICS/COMPASconfusionmatrix.png)

Fainess metrics for COMPAS scores:

![](/Analysis/COMPAS_recidivism_race/METRICS/COMPASfairnessmetrics.png)
