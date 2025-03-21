df1 = pd.read_csv(r'Desktop/可达性计算结果.csv', encoding='gb2312')


class Preprocessing:
    train_rate = 0.85
    seq_len = 60
    pre_len = 1

    def train_test_split(self, data, train_portion=train_rate):
        time_len = data.shape[1]
        train_size = int(time_len * train_portion)
        train_data = np.array(data.iloc[:, :train_size])
        test_data = np.array(data.iloc[:, train_size:])
        return train_data, test_data

    def scale_data(self, train_data, test_data):
        max_speed = train_data.max()
        min_speed = train_data.min()
        train_scaled = (train_data - min_speed) / (max_speed - min_speed)
        test_scaled = (test_data - min_speed) / (max_speed - min_speed)
        return train_scaled, test_scaled

    def sequence_data_preparation(self, train_data, test_data, seq_len=seq_len, pre_len=pre_len):
        trainX, trainY, testX, testY = [], [], [], []

        for i in range(train_data.shape[1] - int(seq_len + pre_len - 1)):
            a = train_data[:, i: i + seq_len + pre_len]
            trainX.append(a[:, :seq_len])
            trainY.append(a[:, -1])
        for i in range(test_data.shape[1] - int(seq_len + pre_len - 1)):
            b = test_data[:, i: i + seq_len + pre_len]
            testX.append(b[:, :seq_len])
            testY.append(b[:, -1])
        trainX = np.array(trainX)
        trainY = np.array(trainY)
        testX = np.array(testX)
        testY = np.array(testY)
        return trainX, trainY, testX, testY


class GcnLstm:
    gc_layer_sizes = [32, 64]
    gc_activations = ["relu", "relu"]
    lstm_layer_sizes = [64, 64]
    lstm_activations = ["tanh", "tanh"]
    batch_size = 32

    def create_gcn_lstm_object(self, adj):
        return GCN_LSTM(
            seq_len=Preprocessing.seq_len,
            adj=adj,
            # functional_adj=functional_adj,
            gc_layer_sizes=self.gc_layer_sizes,
            gc_activations=self.gc_activations,
            lstm_layer_sizes=self.lstm_layer_sizes,
            lstm_activations=self.lstm_activations,
        )

    def custom_loss(self, y_true, y_pred):
        mae_loss = K.mean(K.square(y_true - y_pred))
        mpe_accessibility_loss = ((y_true - y_pred) / (y_true + 0.1))
        # *df1['结果']# 计算MPE与交通可达性的乘积
        rmse_loss = K.sqrt(K.mean(K.square(y_true - y_pred)))
        # total_loss = pow(mae_loss,2) +pow(abs(mpe_accessibility_loss),2)
        total_loss = mae_loss + abs(mpe_accessibility_loss) * 20
        # +pow(rmse_loss,2) *10
        # +pow(abs(mpe_accessibility_loss),2)
        return total_loss

    def build_model(self, x_input, x_output, trainX, trainY, testX, testY):
        es = EarlyStopping(monitor="val_mse", patience=20)
        model = Model(inputs=x_input, outputs=x_output)
        model.compile(optimizer="adam", loss=self.custom_loss, metrics=["mse"])
        history = model.fit(
            trainX,
            trainY,
            epochs=32,
            batch_size=self.batch_size,
            shuffle=True,
            validation_data=(testX, testY), callbacks=[es])
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        y_predict = model.predict(testX)
        return y_predict


class Evaluation:
    def rescale_values(self, train_data, trainY, test_data, testY, y_predict):
        max_speed = train_data.max()
        min_speed = train_data.min()
        train_rescref = trainY * (max_speed - min_speed) + min_speed
        test_rescref = testY * (max_speed - min_speed) + min_speed
        test_rescpred = y_predict * (max_speed - min_speed) + min_speed
        return test_rescref, test_rescpred

    def performance_measure(self, test_rescref, test_rescpred):
        mae = mean_absolute_error(test_rescref, test_rescpred)
        rmse = np.sqrt(mean_squared_error(test_rescref, test_rescpred))
        mpe = np.mean((test_rescref - test_rescpred) / (test_rescref + 0.1))
        return mae, rmse, mpe


if __name__ == '__main__':
    speed = pd.read_csv(r'Desktop/订单数据横版.csv', header=None)
    # adj= pd.read_csv(r'Desktop/毕业设计/数据部分/dataset/邻接矩阵.csv',header=None)
    adj = pd.read_csv(r'Desktop/融合图.csv', header=None)
    weather_data = pd.read_csv(r'Desktop/天气.csv', header=None)

    combined_data = np.concatenate((speed, weather_data), axis=1)
    df = pd.DataFrame(combined_data)

    train_data, test_data = Preprocessing().train_test_split(df)
    train_scaled, test_scaled = Preprocessing().scale_data(train_data, test_data)
    trainX, trainY, testX, testY = Preprocessing().sequence_data_preparation(train_scaled, test_scaled)

    gcn_lstm = GcnLstm().create_gcn_lstm_object(adj)
    x_input, x_output = gcn_lstm.in_out_tensors()
    y_predict = GcnLstm().build_model(x_input, x_output, trainX, trainY, testX, testY)

    test_rescref, test_rescpred = Evaluation().rescale_values(train_data, trainY, test_data, testY, y_predict)
    print(Evaluation().performance_measure(test_rescref, test_rescpred))
    mpe_sgc_lstm = (test_rescref - test_rescpred) / (test_rescref + 0.1)
    mpe_sgc_lstm = pd.DataFrame(mpe_sgc_lstm)
    mpe_sgc_lstm.to_csv(r'Desktop/mpe_sgc_lstm5.csv')