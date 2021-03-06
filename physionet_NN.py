import os, pandas as pd, numpy as np
homedir = os.getenv('HOME')
workpath = os.path.join(homedir, 'Dokumente','python-apps','tensorflow', 'Physionet')
'''
datapath = os.path.join(workpath, 'temp_set_A')
csv_path = os.path.join('.', 'set-A.csv')
# below commented code runs only once, creating csv files on disk made up of files of set-A and padded column for RecordID:
# For an intermediate I/O error I have split up files of set-A into 2 parts (50:50) and processed them separately, then manually renamed them
# as set-A_part1.csv resp. "part2" before concatenating the two creating final "set-A.csv"

retained_row = pd.Series([], dtype=np.str)

def retainer(row):
    global retained_row
    return_value = np.NaN
    if row.sorter == 0:
        return_value = row.value
        retained_row = row.copy()
    else:
        return_value = retained_row['value']
    return return_value

def process_rows(tup):
   if tup[1] == 'RecordID':
      s = 0
   else: s = 1
   df.loc[tup[0], 'sorter'] = s
   df.sort_values(by=['sorter', 'parameter', 'time', 'value'], ascending=True, inplace=True, axis=0)
   #print(df)
df2 = pd.DataFrame(columns=['time','parameter','value','recid'])
for j, f in enumerate(os.listdir(datapath)):
   df = pd.read_csv(datapath+'/'+f, header=0, sep=',',usecols=[0,1,2], names=['time', 'parameter', 'value'], dtype={'time': "string", 'parameter':'string', 'value': np.float64}, error_bad_lines=False)
   print(j, f)
   [process_rows([i,x,y,z]) for i,x,y,z in zip(df.index, df.parameter, df.value, df.time)]
   df = df.assign(recid=df.apply(retainer,  axis=1)) 
   df = df.loc[df.parameter != 'RecordID'] 
   df.drop(columns='sorter', inplace=True) 
   df2 = pd.merge(df, df2, on=['recid','time', 'parameter', 'value'], how='outer', sort=False)
df2.sort_values(by=['recid', 'time', 'parameter'], axis=0, ascending=True, inplace=True, kind='quicksort', na_position='first', ignore_index=True)
print(df2)
#print(df2.shape)
df2.to_csv(csv_path, index=False)
'''
'''
import csv
with open(workpath+'/set-A.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile, delimiter=',')
    with open(workpath+'/set-A_part1.csv', newline='') as csvfile:
       reader = csv.reader(csvfile, delimiter=',')
       for row in reader:
          writer.writerow(row)
    with open(workpath+'/set-A_part2.csv', newline='') as csvfile:
       reader = csv.reader(csvfile, delimiter=',')
       for row in reader:
          writer.writerow(row)
'''
# (1) Read set-A from a single derived CSV:
df = pd.read_csv(workpath+'/set-A.csv', header=0, sep=',',usecols=[0,1,2,3], names=['time', 'parameter', 'value', 'recid'], dtype={'time': "string", 'parameter':'string', 'value': 'string', 'recid':'string'})
paramL = ['DiasABP','MAP','SysABP']
df2 = df.loc[df.parameter.isin(paramL)].copy()
df2.dropna(inplace=True)
df2.sort_values(by=['recid', 'time', 'parameter'], axis=0, ascending=True, inplace=True, kind='quicksort', na_position='first', ignore_index=False)
df3 = df2.groupby(['recid', 'time'], sort=True,  as_index=False).size()
df3 = df3[df3['size'] == 3].copy()
df3.drop(columns='size', inplace=True)
df4 = df2.merge(df3, how='right', on=['recid','time'])
df5 = df4.groupby('recid', sort=False,  as_index=False).size()
df5 = df5[df5['size'] > 59]
df5.drop(columns='size', inplace=True)
df6 = df4.merge(df5, how='right', on=['recid'])
df6['recid_count'] = df6.groupby('recid', sort=False,  as_index=False).cumcount()
df6 = df6.loc[df6.recid_count < 60] # shrink time series to the first 20 time points at 3 measures each
df6['value'] = df6.value.apply(pd.to_numeric)
df6['value'] = df6['value'] / 1000
df6 = df6.loc[:, ['value', 'recid']]
df6.sort_values(by=['recid'], axis=0, ascending=True, inplace=True, kind='quicksort', na_position='first', ignore_index=False)
# Now create the label frame from Outcomes-a.txt assuring the same sort order as df6:
df_l = pd.read_csv(workpath+'/Outcomes-a.txt', header=0, sep=',',usecols=[0,5], names=['recid', 'in_hospital_death'], dtype={'recid': "string", 'in_hospital_death':'int32'})
df_l.sort_values(by='recid', axis=0, ascending=True, inplace=True, kind='quicksort', na_position='first', ignore_index=False)
df_l['recid'] = df_l['recid']+'.0'
df_l = df_l.merge(df6, how='right', on='recid')
df_l.drop(columns='value', inplace=True)
df_l = df_l.groupby('recid', sort=False,  as_index=False).first()
df_l = df_l.loc[:, 'in_hospital_death']
labels_arr = df_l.to_numpy()
#print(labels_arr)

df6.drop(columns='recid', inplace=True)
arr = df6.to_numpy()
arr2 = arr.reshape(2571, 20, 3)#2571,20,3

from tensorflow import keras
model =keras.models.Sequential()
model.add(keras.layers.GRU(50, input_shape=(arr2.shape[1], arr2.shape[2])))
model.add(keras.layers.Dense(25, activation='elu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
History = model.fit(arr2, labels_arr, epochs=1, batch_size=4) # Training phase
train_loss = History.history['loss']
#print(History.history)
acc = History.history['acc']
np.savetxt("train_accuracy.txt", acc, delimiter=",")


