import numpy as np 
import pandas as pd 
import os 
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
from sklearn.preprocessing import MinMaxScaler
import math  
from keras.models import Sequential 
from keras.layers import Dense , BatchNormalization , Dropout , Activation 
from keras.layers import LSTM , GRU 
from sklearn.preprocessing import MinMaxScaler  
from sklearn.metrics import mean_squared_error 
from keras.optimizers import Adam , SGD , RMSprop
from keras.callbacks import ReduceLROnPlateau , ModelCheckpoint
from sklearn.metrics import r2_score

def process_data(data , n_features):
    dataX, dataY = [], []
    for i in range(len(data)-n_features): 
        a = data[i:(i+n_features), 0] 
        dataX.append(a)
        dataY.append(data[i + n_features, 0])
    return np.array(dataX), np.array(dataY)

def plotter(code):
    global closing_stock ,opening_stock 
    f, axs = plt.subplots(2,2,figsize=(15,8))
    plt.subplot(212)
    company = df[df['symbol']==code]
    company = company.open.values.astype('float32')
    company = company.reshape(-1, 1) 
    opening_stock = company
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel(code + " open stock prices") 
    plt.title('prices Vs Time')
    plt.plot(company , 'g')
    plt.subplot(211)
    company_close = df[df['symbol']==code]
    company_close = company_close.close.values.astype('float32')
    company_close = company_close.reshape(-1, 1)
    closing_stock = company_close
    plt.xlabel('Time')
    plt.ylabel(code + " close stock prices")
    plt.title('prices Vs Time')
    plt.grid(True)
    plt.plot(company_close , 'b')
    plt.show()


    
if __name__=="__main__":
    while True:
        df =pd.read_csv("C:\\Users\\tanay\\Downloads\\PROJECTS\\PRJ Stock Price Prediction\\prices.csv", header=0)
        display(df)
        print("The company abbreviations are as follows :")
        names = df["symbol"].unique() 
        print(names)
        comp_info = pd.read_csv("C:\\Users\\tanay\\Downloads\\PROJECTS\\PRJ Stock Price Prediction\\securities.csv")
        comp_plot = comp_info.loc[(comp_info["Security"] == 'Yahoo Inc.') | (comp_info["Security"] == 'Xerox Corp.') | (comp_info["Security"] == 'Adobe Systems Inc')
                      | (comp_info["Security"] == 'Microsoft Corp.') | (comp_info["Security"] == 'Adobe Systems Inc') 
                      | (comp_info["Security"] == 'Facebook') | (comp_info["Security"] == 'Goldman Sachs Group') , ["Ticker symbol"] ]["Ticker symbol"] 

        for i in comp_plot:
            display(plotter(i))

        company = input("Enter the company abbreviation : ").upper()
        
        if company not in names:
            print("A company Abbreviation does not exists")
            choice = input("Do you want to continue (yes or no) :")
            lst = ["y","Y","Yes","YEs","YES","yes","yES","yEs","yeS"]
            if choice not in lst:
                print("Thank You for using the Prediction Algorithm")
                break
            else:
                continue
            
        stocks= np.array (df[df.symbol.isin ([company])].close)
        stocks = stocks.reshape(len(stocks) , 1)

        scaler = MinMaxScaler(feature_range=(0, 1)) 
        stocks = scaler.fit_transform(stocks) 

        train = int(len(stocks) * 0.80)

        test = len(stocks) - train

        train = stocks[0:train]

        test = stocks[len(train) : ]

        n_features = 2
        trainX, trainY = process_data(train, n_features)

        testX, testY = process_data(test, n_features)

        stocksX, stocksY = process_data(stocks, n_features)

        trainX = trainX.reshape(trainX.shape[0] , 1 ,trainX.shape[1])

        testX = testX.reshape(testX.shape[0] , 1 ,testX.shape[1])

        stocksX= stocksX.reshape(stocksX.shape[0] , 1 ,stocksX.shape[1])

        filepath="stock_weights1.hdf5"
        lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, epsilon=0.0001, patience=1, verbose=1)
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')

        model = Sequential()
        model.add(GRU(256 , input_shape = (1 , n_features) , return_sequences=True))
        model.add(Dropout(0.4))
        model.add(LSTM(256))
        model.add(Dropout(0.4)) 
        model.add(Dense(64 ,  activation = 'relu')) 
        model.add(Dense(1))

        model.compile(loss='mean_squared_error', optimizer=Adam(lr = 0.0005) , metrics = ['mean_squared_error'])

        history = model.fit(trainX, trainY, epochs=100 , batch_size = 128 , callbacks = [checkpoint , lr_reduce] , validation_data = (testX,testY))    

        test_pred = model.predict(testX)
        test_pred = scaler.inverse_transform(test_pred)
        testY = testY.reshape(testY.shape[0] , 1)
        testY = scaler.inverse_transform(testY)

        print("Red - Predicted Stock Prices  ,  Blue - Actual Stock Prices")
        plt.rcParams["figure.figsize"] = (15,7)
        plt.plot(testY , 'b')
        plt.plot(test_pred , 'r')
        plt.xlabel('Time')
        plt.ylabel('Stock Prices')
        plt.title('Check the accuracy of the model with time') 
        plt.grid(True)
        plt.show()

        train_pred = model.predict(trainX)
        train_pred = scaler.inverse_transform(train_pred)
        trainY = trainY.reshape(trainY.shape[0] , 1)
        trainY = scaler.inverse_transform(trainY)

        stocks_pred = model.predict(stocksX)
        stocks_pred = scaler.inverse_transform(stocks_pred)
        stocksY = stocksY.reshape(stocksY.shape[0] , 1)
        stocksY = scaler.inverse_transform(stocksY)

        results= df[df.symbol.isin ([company])]
        results= results [2:]
        results = results.reset_index(drop=True)
        df_stocks_pred= pd.DataFrame(stocks_pred, columns = ['Close_Prediction'])
        results= pd.concat([results,df_stocks_pred],axis =1)
        results.to_excel('results.xlsx')
        display(results)

        choice = input("Do you want to continue (yes or no) :")
        lst = ["y","Y","Yes","YEs","YES","yes","yES","yEs","yeS"]
        if choice not in lst:
            print("Thank You for using the Prediction Algorithm")
            break
        
