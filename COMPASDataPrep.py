import pandas as pd

compas = pd.read_csv("RawDatasets/compas-scores-two-years.csv")

# filtering for useful parameters
compas_filtered = compas[
        ["sex", "age_cat", "race", "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count", "decile_score",
         "score_text", "c_charge_degree", "two_year_recid"]]
#sort alphabetically
compas_filtered = compas_filtered.reindex(sorted(compas_filtered.columns), axis=1)

#merge the races Asian, Native American, Hispanic with Other
compas_filtered['race']= compas_filtered['race'].replace(['Hispanic', 'Native American', 'Asian'], 'Other')

compas_filtered.to_csv('PreppedData/compas-scores.csv', index= False)