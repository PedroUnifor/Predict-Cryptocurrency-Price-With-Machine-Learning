
# coding: utf-8

# In[2]:


import numpy as np # working with data
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import utilities


# # Predict Cryptocurrency Prices With Machine Learning #

# ### Step 1 Load & Process Data

# In[3]:


currency = "BTC" # moeda a ser operad

currency_data = utilities.get_dataset(currency=currency) #colocar todos os dados da "currency"(abertura, fechamento, data, valor) em currency_data
#<class 'pandas.core.frame.DataFrame'>
currency_close_price = currency_data.close.values.astype('float32') #convert os dados para dtype, podendo ser interpreto pelo numpy
#<class 'numpy.ndarray'>
currency_close_price = currency_close_price.reshape(len(currency_close_price), 1) #convert para array


# In[24]:


def create_datasets(dataset, sequence_length):
    sequence_length += 1
    seq_dataset = [] #lista
    for i in range(len(dataset) - sequence_length):
        seq_dataset.append(dataset[i: i + sequence_length])
    seq_dataset = np.array(seq_dataset)
    
    data_x = seq_dataset[:, :-1]
    data_y = seq_dataset[:, -1]
    return data_x, data_y
scaler = MinMaxScaler(feature_range=(0, 1))
#scaler = <class 'sklearn.preprocessing.data.MinMaxScaler'>
currency_close_price_scaled = scaler.fit_transform(currency_close_price)
print(currency_close_price_scaled)
train_size = int(len(currency_close_price_scaled) * 0.85)
test_size = len(currency_close_price_scaled) - train_size
train, test = currency_close_price_scaled[0:train_size,:], currency_close_price_scaled[train_size:len(currency_close_price_scaled),:]

look_back = 10

x_train, y_train = create_datasets(train, look_back)
x_test, y_test = create_datasets(test, look_back)


# ### Step 2 Build Model

# In[16]:


model = Sequential() #simples modelo de inicialização do Sequencial
#Sequential: é um modelo é uma pilha linear de camadas.

model.add(LSTM( #Aplicação da Rede Neural LSTM (rede na qual todas as informações são recebidas 
    #em sequencia e reavaliadas pelo neuronio)
    input_dim=1,
    #dimensionalidade da entrada (integer). Esse argumento (ou, alternativamente, o argumento da palavra-chave input_shape)
    #é necessário ao usar essa camada como a primeira camada em um modelo.
    output_dim=50,
    #dimensão das projeções internas e do resultado final
    return_sequences=True))
    #Boleano. Se deve retornar a última saída na sequência de saída ou a sequência completa.
model.add(Dropout(0.35))
    #consiste em configurar aleatoriamente uma taxa de fração de unidades de 
    #entrada para 0 a cada atualização durante o tempo de treinamento, o que ajuda a evitar o overfitting.
model.add(LSTM(
    100,
    return_sequences=False))
model.add(Dropout(0.30))

model.add(Dense(
    #Dense: a camada(ou layer) Dense é uma camada onde cada unidade ou neurônio é conectado a cada neurônio na próxima camada.
    output_dim=1))
model.add(Activation('linear'))

# Para um problema de regressão de erro quadrático médio:
model.compile(loss='mse', optimizer='rmsprop')
#adicionar um modelo de compilação com otimizador e uma função de erro
#rmsprop: uma boa escolhe de otimizador para RNN


# In[17]:


history = model.fit(x_train, y_train, batch_size=64, epochs=100, verbose=2, validation_split=0.2)
#fit: Treinamento do modelo


# In[18]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# ### Step 3 Predict

# In[19]:


train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

train_predict_unnorm = scaler.inverse_transform(train_predict)
test_predict_unnorm = scaler.inverse_transform(test_predict)

# CREATING SIMILAR DATASET TO PLOT TRAINING PREDICTIONS
trainPredictPlot = np.empty_like(currency_close_price)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict_unnorm)+look_back, :] = train_predict_unnorm

# CREATING SIMILAR DATASSET TO PLOT TEST PREDICTIONS
testPredictPlot = np.empty_like(currency_close_price)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict_unnorm)+(look_back*2)+1:len(currency_close_price)-1, :] = test_predict_unnorm

plt.figure(figsize=(30, 20))
plt.plot(currency_close_price, 'g', label = 'original dataset')
plt.plot(trainPredictPlot, 'r', label = 'training set')
plt.plot(testPredictPlot, 'b', label = 'predicted price/test set')
plt.legend(loc = 'upper left')
plt.xlabel('Time in Days')
plt.ylabel('Price')

plt.title("%s price %s - % s" % (currency, 
                                 utilities.get_date_from_current(offset=len(currency_close_price)), 
                                 utilities.get_date_from_current(0)))

plt.show()

