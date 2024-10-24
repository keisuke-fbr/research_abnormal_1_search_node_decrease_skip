#必要ライブラリのインストール
#必要ライブラリのインストール

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import setuptools
import tensorflow as tf
from tensorflow import keras


from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses, initializers
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping


#モデルの最小化関数
# カスタム損失関数の定義
def custom_loss(delta):
    def loss(y_true, y_pred):
        error = tf.abs(y_true - y_pred)
        small_error_loss = tf.square(error)
        large_error_loss = error
        return tf.where(error < delta, large_error_loss, small_error_loss)
    return loss

def add_to_dict_with_average(dictionary, key, value):
    if key in dictionary:
        # 既存の値がリストでない場合、リストに変換
        if not isinstance(dictionary[key], list):
            dictionary[key] = [dictionary[key]]  # 値をリストに変換
        # リストに値を追加
        dictionary[key].append(value)
        # 平均値を再計算して辞書に格納
        dictionary[key] = np.mean(dictionary[key])
    else:
        # キーが存在しない場合、新しいリストとして値を追加
        dictionary[key] = value


def root_mean_squared_error(y_true, y_pred):
    return tf.math.sqrt(tf.math.reduce_mean(tf.math.square(y_pred - y_true)))

#異常スコア算出関数
def abnomalScores(test_data,predicted_data):
  # 平均二乗誤差 (MSE) を各サンプルに対して計算
    mse_per_sample = np.mean(np.square(test_data - predicted_data), axis=1)

    # 平方根を取って、各サンプルごとのRMSEを計算
    rmse_per_sample = np.sqrt(mse_per_sample)

    return rmse_per_sample

#閾値を求める関数
def biggest_threshold(test_data, result_data):
    # 平均二乗誤差 (MSE) を各サンプルに対して計算
    mse_per_sample = np.mean(np.square(test_data - result_data), axis=1)

    # 平方根を取って、各サンプルごとのRMSEを計算
    rmse_per_sample = np.sqrt(mse_per_sample)

    # 最も大きなRMSEを取得
    max_rmse = np.max(rmse_per_sample)

    return max_rmse


#AutoEncoderのモデル
def model_autoencoder(initializer, units_1_3, units_2):
    # モデルについて
    #　中間層数は３で設定。中間層１と３は同じユニット数

    # モデルを作成
    model = keras.Sequential(name="autoencoder")

    #重みの初期化方法
    initializer = initializer

    #入力層の情報
    input_unit = 11

    #中間層の情報
    middle_unit_1 = units_1_3
    middle_unit_2 = units_2
    middle_unit_3 = units_1_3
   

    #出力層の情報
    output_unit = 11

    #層の追加
    #中間層１の作成
    model.add(layers.Dense(units=middle_unit_1, activation="sigmoid" ,input_shape=(input_unit,), kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev = pow(units_1_3, -0.5))))
    # 中間層２の作成
    model.add(layers.Dense(units=middle_unit_2, activation="sigmoid", kernel_initializer = initializers.TruncatedNormal(mean=0.0, stddev = pow(units_2, -0.5))))
    # 中間層３の作成
    model.add(layers.Dense(units=middle_unit_3, activation="sigmoid", kernel_initializer = initializers.TruncatedNormal(mean=0.0, stddev = pow(units_1_3, -0.5))))

    #出力層の作成
    model.add(layers.Dense(units=output_unit, activation="linear", kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev = pow(175, -0.5))))
    
    #最適化手法
    learning_rate = 1e-4
    #opt=tf.keras.optimizers.legacy.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False) # default

    

    model.compile(optimizer="adam",
                        loss=root_mean_squared_error
                       )  

    return model


#モデル探索の学習関数、モデルと学習データを入力することで学習を完了させ、その損失が得られる
#学習アルゴリズムはmax_epochs回学習させ、損失関数に変化がなくなった時点で学習を終了させる
def train_and_evaluate(model, X_train, max_epochs, early_stopping_params):

    early_stopping = EarlyStopping(**early_stopping_params)

    history = 0
    #ここでの終了条件はあくまでも収束条件
    history = model.fit(X_train, X_train, epochs=max_epochs, batch_size=64, verbose=0
                        , shuffle = True, validation_data = (X_train,X_train) ,callbacks=[early_stopping])

    # EarlyStoppingで打ち切られたかどうかを確認する
    if len(history.epoch) < max_epochs:
        print(f"モデルは収束しました。訓練は {len(history.epoch)} エポックで停止しました。")
    else:
        print("モデルは最大エポック数まで収束条件に達しませんでした。収束条件を満たさず終了しました。")

    #最終モデルの決定値は損失そのもの
    #損失関数はroot_mean_squared_errorであり、今回はバッチサイズが64なので64データの損失の平均を取得する
    final_loss = history.history['loss'][-1]  # 最終エポックの検証損失
    
    
    return final_loss


def search_model(initializer, X_train, error_threshold, max_epochs, early_stopping_params, units, num_initializations=4):
    best_model = None
    best_units = None
    best_loss = float('inf')

    max_units = 30

    # 各ユニット数に対するfinal_lossを保存する辞書
    final_losses_per_units = {}

    #unit_2に対するunit1_3の最適ユニット数
    optimal_units = {}
    #各ユニットに対応するmodel
    optimal_models = {}

    stop_exploration = False  # 閾値を満たさなかった場合の探索終了フラグ

    # 探索開始
    for units_2 in reversed(range(2, units)):
        if stop_exploration:
            break  # 閾値を満たさなかったため探索を終了

        best_model_unit_2 = None
        optimal_unit = None

        # ノード数を3つずつ減らして探索
        for units_1_3 in reversed(range(units + 1, max_units + 1, 3)):
            final_losses_for_units = []  # 各ユニット数に対する複数回の初期化結果を保存するリスト

            # 初期化方法を4回試行
            flag_stop = 0  # 初期化4回行った後に閾値を満たさない場合、詳細探索を行う
            for init_num in range(num_initializations):
                print("----------------------------------------------------------------------------------------")
                print(f"探索中: units_1_3={units_1_3}, units_2={units_2}, 初期化方法={initializer}, 試行回数={init_num+1}")
                
                model = model_autoencoder(initializer, units_1_3, units_2)
                final_loss = train_and_evaluate(model, X_train, max_epochs, early_stopping_params)
                final_losses_for_units.append(final_loss)
                
                if final_loss < error_threshold:
                    # 閾値を下回った場合、次へ進む
                    print(f"閾値を下回るモデルを発見: units_1_3={units_1_3}, units_2={units_2}")
                    best_model_unit_2 = model
                    optimal_unit = units_1_3
                    flag_stop = 1  # 閾値を下回ったので詳細探索は不要
                    break
                
                print("閾値を下回りませんでした" + str(init_num+1) + "回目")

            average_loss = np.mean(final_losses_for_units)
            final_losses_per_units[(units_1_3, units_2)] = average_loss

            # 初期化4回とも閾値を下回らなかった場合に詳細な探索を開始
            if flag_stop==0:
                print(f"初期化4回すべてで閾値を上回りました: units_1_3={units_1_3}, units_2={units_2}")
                print("ノード数を上げて探索を続けます")
                for fine_units_1_3 in range(units_1_3+1, units_1_3 + 3):  # `units_1_3`自体は探索しない
                    print(f"unit1_3={fine_units_1_3}の探索")
                    fine_losses_for_units = []
                    for init_num in range(num_initializations):
                        model = model_autoencoder(initializer, fine_units_1_3, units_2)
                        print(f"unit1_3={fine_units_1_3}の探索：初期化{init_num+1}回目")
                        fine_loss = train_and_evaluate(model, X_train, max_epochs, early_stopping_params)
                        fine_losses_for_units.append(fine_loss)

                        if fine_loss < error_threshold:
                            print(f"詳細探索で閾値を満たすモデルを発見: units_1_3={fine_units_1_3}, units_2={units_2}")
                            best_model_unit_2 = model
                            optimal_unit = fine_units_1_3
                            flag_stop = 2
                            break
                        print("閾値を下回りませんでした")
                    average_fine_loss = np.mean(fine_losses_for_units)
                    final_losses_per_units[(fine_units_1_3, units_2)] = average_fine_loss
                    if flag_stop == 2:
                        break
                optimal_models[(optimal_unit, units_2)] = best_model_unit_2
                optimal_units[units_2] = optimal_unit

                flag_stop == 2

                # units_1_3 が 30 でも閾値を下回らなかった場合、以降の units_2 の探索を停止
                if units_1_3 == max_units:
                    print(f"units_1_3 が {max_units} でも閾値を下回らなかったため、探索を終了します。")
                    stop_exploration = True
                    break  # `units_2` の探索を中止
            
                if flag_stop == 2:
                    print("次のunit2の探索に移ります")
                    break
        if stop_exploration or units_2 == 2:
            break  # `units_2` の探索全体を終了

    # 最適なunit_1_3とunit_2の組み合わせを見つける
    min_total_units = float('inf')
    best_unit_1_3 = None
    best_unit_2 = None

    for unit_2, unit_1_3 in optimal_units.items():
        total_units = unit_1_3 * 2 + unit_2

        if total_units < min_total_units:
            min_total_units = total_units
            best_unit_1_3 = unit_1_3
            best_unit_2 = unit_2

    print(f"最小ユニットの組み合わせ: units_1_3={best_unit_1_3}, units_2={best_unit_2}, 合計ユニット数={min_total_units}")
    best_model = optimal_models[(best_unit_1_3, best_unit_2)]

    del optimal_units, optimal_models

    return best_model, final_losses_per_units

    


#15期間においてモデルを作成し、結果を格納する関数
#各期間においてほしい情報は、日付に対応する異常スコア、閾値
def result(data_ex, colums_list,  initializer,error_threshold, max_epochs, early_stopping_params, units, num = 4  ):

    #異常スコアを格納する配列
    results_df = pd.DataFrame(columns=["measurement_date"] + ["anomaly_score"])

    #トレーニングデータの再構成値を格納する配列
    traindata_model_df = pd.DataFrame(columns=["measurement_date"] + colums_list)
    
    #各期間における閾値を格納する配列
    thresholds = []

    #lossを保持するリスト
    loss_values = {}



    #初期データの抜き出し及び、繰り返し処理の準備
    data_trainstart = data_ex["measurement_date"][0]

    data_trainstart_year = data_trainstart.year
    data_trainstart_month = data_trainstart.month
    data_trainstart = str(data_trainstart_year) + "/" + str(data_trainstart_month) + "/01 00:00:00"

    #手動でデータの設定
    data_trainstart = "2016/6/01 00:00:00"
    data_trainstart_year = 2016
    data_trainstart_month = 6

    data_trainend_year = data_trainstart_year + 2
    data_trainend_month = data_trainstart_month
    data_trainend = str(data_trainend_year) + "/" + str(data_trainend_month) + "/01 00:00:00"
    
    #手動でデータの設定
    data_trainend = "2018/6/01 00:00:00"
    data_trainend_year = 2018
    data_trainend_month = 6
    
    data_teststart = data_trainend
    if data_trainend_month == 12:
        data_testend_year = data_trainend_year + 1
        data_testend_month = 1
    else:
        data_testend_year = data_trainend_year
        data_testend_month = data_trainend_month + 1
    
    data_testend = str(data_testend_year) + "/" + str(data_testend_month) + "/01 00:00:00"

    #モデル実装部分
    for i in range(num):
        print("=====================================================================================")
        print(str(i+1) + "回目の期間のモデル")

        #i回目のデータの抜き出し
        #トレーニングデータの抜き出し
        train_data = data_ex[(data_ex["measurement_date"]<data_trainend) & (data_ex["measurement_date"]>=data_trainstart)]
        train_data = train_data[colums_list]
        train_data = train_data.values
    
    
        #テストデータの抜き出し
        #テストデータはdata_trainendの一か月である
        test_data = data_ex[(data_ex["measurement_date"]<data_testend) & (data_ex["measurement_date"]>=data_teststart)]
        test_data_origin = data_ex[(data_ex["measurement_date"]<data_testend) & (data_ex["measurement_date"]>=data_teststart)]
        test_data = test_data[colums_list]
        test_data = test_data.values


        #該当データ期間
        print("該当期間の確認")
        print("はじまり(train)" + str(data_trainstart))
        print("終わり(train)" + str(data_trainend))
        print("はじまり(test)" + str(data_teststart))
        print("終わり(test)" + str(data_testend))
    

        #初期点を変更しながら、最適モデルを構築する
        print(str(i+1) + "回目の期間のモデル作成開始")
        print(str(i+1) + "回目の期間のトレーニングデータ数:" + str(len(train_data)))

        #モデルの作成
        best_model, final_losses_per_units = search_model(initializer,train_data, error_threshold, max_epochs, early_stopping_params, units)


        for key, value in final_losses_per_units.items():
            add_to_dict_with_average(loss_values, key, value)


        print("====================================================================================================")
        print(str(i+1) + "回目の期間のモデル作成終了")
        print("====================================================================================================")


        #結果
        print(str(i+1) + "回目の期間のテストデータ数:" + str(len(test_data)))

        #テストデータの予測値
        result_data = best_model.predict(test_data)

        #トレーニングデータの再構成値
        traindata_model = best_model.predict(train_data)

        #再構築データ数
        print("再構築データ数 : " + str(len(traindata_model)))
    
        #pandas形式へ変換
        test_data = pd.DataFrame(test_data,columns=colums_list)
        result_data = pd.DataFrame(result_data,columns=colums_list)
        traindata_model = pd.DataFrame(traindata_model, columns=colums_list)
    
        #異常値の算出
        abnormal_value = abnomalScores(test_data,result_data)

        #最も再構成誤差の大きい特徴の抽出
        best_threshold = biggest_threshold(train_data, traindata_model)
        print(f"異常スコアの閾値: {best_threshold}")
    
    
        #今期間の異常スコア、予測の結果の格納
        temp_df = pd.DataFrame({
            "measurement_date": test_data_origin["measurement_date"].values,
            "anomaly_score": abnormal_value
        })


        train_data_date = data_ex[(data_ex["measurement_date"]<data_trainend) & (data_ex["measurement_date"]>=data_trainstart)]
        temp_traindata_df = pd.DataFrame({
            "measurement_date" : train_data_date["measurement_date"].values
        })

        #再構成値の格納
        for col in colums_list:
            temp_traindata_df[col] = traindata_model[col]

        
    
        #これまでの期間の結果の結合
        results_df = pd.concat([results_df, temp_df], ignore_index=True)

        traindata_model_df = pd.concat([traindata_model_df, temp_traindata_df],ignore_index=True)


        print("traindata_model_dfの日付の最大値:" + str(traindata_model_df["measurement_date"].max()))
        print("trainの日付の最大値:" + str(data_trainend)) 
        print("testの日付の最大値:" + str(data_testend))

        #各期間の閾値を格納
        thresholds.append({
            "term" : i+1,
            "threshold" : best_threshold,
            "test_start" : data_teststart,
            "test_end" : data_testend
        })

        #日付の更新
        if data_trainstart_month == 12:
            data_trainstart_year += 1
            data_trainstart_month = 1
        else:
            data_trainstart_month += 1
        data_trainstart = str(data_trainstart_year) + "/" + str(data_trainstart_month) + "/01 00:00:00"        
        
        data_trainend_year = data_trainstart_year + 2
        data_trainend_month = data_trainstart_month
        data_trainend = str(data_trainend_year) + "/" + str(data_trainend_month) + "/01 00:00:00"
    
        data_teststart = data_trainend
        if data_trainend_month == 12:
            data_testend_year = data_trainend_year + 1
            data_testend_month = 1
        else:
            data_testend_year = data_trainend_year
            data_testend_month = data_trainend_month + 1
    
        data_testend = str(data_testend_year) + "/" + str(data_testend_month) + "/01 00:00:00"

    #再構築データに関して、期間が被る部分があるので同じ日付を予測している箇所がある。よって平均をとってデータ数をそろえる。
    traindata_model_df = traindata_model_df.groupby("measurement_date").mean().reset_index()

    #データの確認ログ
    print("data_exの日付の最大値:" + str(data_ex["measurement_date"].max()))
    print("traindata_model_dfの日付の最大値:" + str(traindata_model_df["measurement_date"].max()))
    print("data_exのデータ数:" + str(len(data_ex)))
    print("traindata_model_dfのデータ数:" +str(len(traindata_model_df)))
    # 日付が文字列で保存されている場合、日付型に変換します
    data_ex['measurement_date'] = pd.to_datetime(data_ex['measurement_date'])
    # 2019年2月のデータをフィルタリングして、そのデータ数をカウントします
    data_feb_2019 = data_ex[(data_ex['measurement_date'].dt.year == 2019) & (data_ex['measurement_date'].dt.month == 2)]
    # データ数を表示
    feb_2019_count = data_feb_2019.shape[0]
    print(f"data_exの最終月に含まれるデータ数:{feb_2019_count}")
    return  results_df, traindata_model_df, thresholds, loss_values
        
        
    
    