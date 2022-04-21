import utils

# dataset = 'recidivism'
# attributes_to_protect = ['race']
# attributes_to_protect = ['sex']
# *attributes_to_protect, = 'sex', 'race'

# dataset = 'compas'
# attributes_to_protect = ['race']
# attributes_to_protect = ['sex']
# *attributes_to_protect, = 'sex', 'race'

# dataset = 'german'
# attributes_to_protect = ['sex']
# attributes_to_protect = ['age']
# *attributes_to_protect, = 'sex', 'age'

dataset = 'income'
# attributes_to_protect = ['race']
# attributes_to_protect = ['sex']
*attributes_to_protect, = 'sex', 'race'

metrics= utils.getMetrics(dataset, attributes_to_protect)
fairness_metrics = utils.getFairnessMetrics(metrics, attributes_to_protect)

print(fairness_metrics)

output_name = dataset +'-' + '-'.join(attributes_to_protect)
fairness_metrics.to_csv('FairnessMetrics/'+output_name+'.csv', index= False)




