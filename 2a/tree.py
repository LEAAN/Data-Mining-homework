from openml.apiconnector import APIConnector
import pandas as pd
from pandas.tools.plotting import parallel_coordinates
import matplotlib.pyplot as plt

apikey = "1cfa29e70221198faaa83b0aaa97c60c"
connector = APIConnector(apikey=apikey)
dataset_id = 10
dataset = connector.download_dataset(dataset_id)
#X 148 x 18 data
#y 148 lable
X, y, attribute_names = dataset.get_dataset(target=dataset.default_target_attribute, return_attribute_names=True)
iris = pd.DataFrame(X, columns=attribute_names)
iris['class'] = y
plt.figure()
parallel_coordinates(iris, 'class', colormap='rainbow',linewidth=3)
plt.xticks(rotation=30, fontsize = 10)
plt.grid(False)
plt.show()


filtered = iris[(iris['class']==1) | ( iris['class']==2)]
filtered_1 = iris[(iris['class']==1)]
plt.figure()
filter_1 = parallel_coordinates(filtered_1, 'class', colormap='gist_rainbow',linewidth=3)
plt.legend('1')
plt.show
plt.axis([0, 17, -1, 8])
plt.xticks(rotation=30, fontsize = 8)
plt.grid(False)
plt.title('class 1')
plt.show()

filtered_2 = iris[(iris['class']==2)]
plt.figure()
filter_2 = parallel_coordinates(filtered_2, 'class', colormap='rainbow',linewidth=3)
plt.legend('2')
plt.axis([0, 17, -1, 8])
plt.xticks(rotation=30, fontsize = 8)
plt.grid(False)
plt.title('class 2')
plt.show()

#plt filtered
plt.figure()
parallel_coordinates(filtered, 'class', colormap='rainbow',linewidth=3)
plt.xticks(rotation=30, fontsize = 10)
plt.grid(False)
plt.show()
#plt iris
plt.figure()
parallel_coordinates(iris, 'class', colormap='rainbow',linewidth=3)
plt.xticks(rotation=30, fontsize = 8)
plt.grid(False)
plt.show()

print 'hello Wangzong'
plt.show()



