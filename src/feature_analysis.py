from numpy.core.fromnumeric import argsort
from pandas import read_csv
from numpy import set_printoptions
from sklearn.base import ClassifierMixin
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import ExtraTreesClassifier

def invariantFeatureAnalysis(feature_columns, result_columns) :
  '''
  Uses the provided DataFrame to perform 
  '''
  # All Columns except result and meta data columns
  X = feature_columns

  # Result Column for the Win (H/A/T)
  Y = result_columns

  # feature selection
  test = SelectKBest(score_func=f_classif, k=8)
  fit = test.fit(X,Y)

  
  set_printoptions(precision=5)
  result_array = fit.scores_
  #print(result_array)
  #features = fit.transform(X)
  
  #summarize selected features
  #print(features[0:9,:])
  #print('Univariant Feature Ranking')
  #print(argsort(result_array))
  return result_array
  

def featureImportance(dataframe) :
  '''
  Performs Feature importance based on Extra Tree Classifer
  '''

  array = dataframe.values

  X = array[:,9:48]
  Y = array[:,8]

  model = ExtraTreesClassifier(n_estimators=100)
  model.fit(X,Y)
  result_array = model.feature_importances_
  return result_array
  
def printColumns(array, columns) :
  '''
  Prints the Columns from the provided indexes
  '''
  filtered=[]
  for index in argsort(array):
    filtered.append(columns[index])
  print('Column Headers')
  print(filtered)

def buildColumnList(columns, start, end):
  return columns[start:end]


#load data
filename = 'input/MLData.txt'
delimiter = ','
dataframe = read_csv(filename, delimiter=delimiter, header=0)
dataframe = dataframe.convert_dtypes()

home_wins = dataframe.query('Result == "H"')

print('Invariant Feature Selection based on Home Win Result')
home_array = home_wins.values
home_result = invariantFeatureAnalysis(home_array[:,9:48], home_array[:,8])
printColumns(home_result, dataframe.columns.values[9:48])
print('')

print('Invariant Feature Selection for Home Point Differential')
point_diff = dataframe.values
home_point = invariantFeatureAnalysis(point_diff[:,9:48], point_diff[:,6])
printColumns(home_point, dataframe.columns.values[9:48])
print('')

print('Invariant Feature Selection for Home Point Scored')
filter_col = [col for col in dataframe if col.startswith('H') or col == 'HomePoints']
df = dataframe[filter_col]
point_score = df.values

home_score = invariantFeatureAnalysis(point_score[:,2:21], point_score[:,1])
printColumns(home_score, filter_col)
print('')

print('Feature Importance')
feature = featureImportance(dataframe)
printColumns(feature, dataframe.columns.values[9:48])
