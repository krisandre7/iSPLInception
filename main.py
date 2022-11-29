# title       :main
# description :The main script.
# author      :Ronald Mutegeki
# date        :20210203
# version     :1.0
# usage       :main.py
# notes       :This script is where all the magic happens. Ensure to adjust all the required settings and preferences.
#              One can decide which model architectures they want to run, specify hyperparameters even for individual
#              models, decide which items to add to the report, etc... We have also configured gpu memory growth.
# outputs     :reports, images, logs (TensorBoard), and trained models.

from datetime import datetime
from time import time

import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

# Util Imports
from utils import *

# ................................................................................................ #
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ................................................................................................ #
# Model Configurations
model_list = ['ispl_inception', 'cnn']
acc_per_model = {}
loss_per_model = {}

img_path = f"images/"

# Models to run
for list_of_models in [
    # ['stacked_lstm', 'bilstm', 'ispl_inception'],
    ['vanilla_lstm', 'ispl_inception','cnn', 'cnn_lstm'],
    # ['vanilla_lstm'],
]:
    try:
        model_list = list_of_models
        # Loss and acc
        METRICS = [
            'accuracy'
        ]

        # Training time
        training_id = f'{time():.0f}'
        epochs = 30
        batch_size = 64
        patience = 250

        if dataset == "daphnet":
            batch_size = 64

        acc_per_model = {}
        loss_per_model = {}
        trained_models = {}
        model_histories = {}
        mod_name = 'iSPLInception'

        # Report Generation
        report_name = f"reports/{dataset}/{training_id}_{'__'.join(model_list)}_report.txt"
        os.makedirs(os.path.dirname(report_name), exist_ok=True)

        # Path to images
        img_path = f"images/{dataset}/{training_id}_{'__'.join(model_list)}"
        if not os.path.exists(img_path):
            os.makedirs(img_path)

        pd.set_option('display.max_rows', 400)
        pd.set_option('display.max_columns', 200)

        # Display the input data signal

        with open(report_name, "w") as report:
            print("################# Information #################")
            print("Dataset: ", dataset)
            print("Training_id:", training_id)
            print("Epochs:", epochs)
            print(f"Train:{X_train.shape} | {y_train.shape}")
            print(f"Validation:{X_val.shape} | {y_val.shape}")
            print(f"Test:{X_test.shape} | {y_test.shape}")

            print("Models being trained : ", model_list)

            report.write(f"This is the report for the {dataset} Dataset\n\n")

            report.write("Data Distribution: \n\n")
            report.write(f"Train:  X -> {X_train.shape} Class count -> {list(np.bincount(y_train.argmax(1)))} \n\n"
                         f"{pd.DataFrame(y_train.mean(axis=0) * 100, index=labels, columns=['frequency'])}\n\n")
            report.write(f"Validation:  X -> {X_val.shape} Class count -> {list(np.bincount(y_val.argmax(1)))} \n\n"
                         f"{pd.DataFrame(y_val.mean(axis=0) * 100, index=labels, columns=['frequency'])}\n\n")
            report.write(f"Test:  X -> {X_test.shape} Class count -> {list(np.bincount(y_test.argmax(1)))} \n\n"
                         f"{pd.DataFrame(y_test.mean(axis=0) * 100, index=labels, columns=['frequency'])}\n\n")

            i = 0
            start = time()
            for model_name in model_list:
                s = time()
                i += 1
                print(f"\n\n ************* {i}  ************* {i}  ************* {i}  *************")
                print(f'Model {i}/{len(model_list)}')
                print('###############################################################################')
                print(f"Training {model_name} : {datetime.now()}")
                print('###############################################################################')
                log_dir = os.path.abspath(f'logs/fit/{dataset}/{training_id}/{model_name}')
                save_name = os.path.abspath(f"models/{dataset}/{training_id}/{model_name}.h5")

                input_shape = X_train.shape

                if model_name == 'vanilla_lstm':
                    # Give model specific configurations
                    hyperparameters = {'n_hidden': 128, 'learning_rate': 0.0005, 'regularization_rate': 0.000093}
                    epochs = 350
                    # batch_size = 64
                    mod_name = 'vLSTM'
                    patience = 300

                elif model_name == 'stacked_lstm':
                    # Give model specific configurations
                    hyperparameters = {'n_hidden': 128, 'learning_rate': 0.0005, 'regularization_rate': 0.000093,
                                       'depth': 4}
                    mod_name = 'sLSTM'
                    epochs = 350
                    patience = 150

                elif model_name == 'bilstm':
                    # Give model specific configurations
                    hyperparameters = {'n_hidden': 128, 'learning_rate': 0.0005, 'regularization_rate': 0.000093,
                                       'depth': 1,
                                       'merge_mode': 'concat'}
                    mod_name = 'BiLSTM'
                    epochs = 350
                    patience = 200

                elif model_name == 'cnn':
                    # Give model specific configurations
                    hyperparameters = {'filters': [32, 64, 32], 'fc_hidden_nodes': 100, 'learning_rate': 0.0005,
                                       'regularization_rate': 0.000093}
                    mod_name = 'CNN'
                    epochs = 350
                    patience = 300

                elif model_name == 'cnn_lstm':
                    # Give model specific configurations
                    hyperparameters = {'n_hidden': 512, 'n_steps': n_steps, 'length': length, 'n_signals': n_signals,
                                       'learning_rate': 0.0005, 'cnn_depth': 3, 'lstm_depth': 2,
                                       'regularization_rate': 0.000093}
                    mod_name = 'CNN_LSTM'
                    epochs = 350
                    patience = 300

                else:
                    # Give default model specific configurations
                    # Our default model is the iSPLInception model
                    use_residual = True
                    use_bottleneck = True
                    hyperparameters = {'learning_rate': 0.0005, 'regularization_rate': 0.00593,
                                       'network_depth': 5, 'filters_number': 64, 'max_kernel_size': 68,
                                       'use_residual': use_residual, 'use_bottleneck': use_bottleneck}
                    mod_name = 'iSPLInception'
                    epochs = 350
                    patience = 300

                # Create and initialize the Model
                model = eval(model_name + f"(input_shape, n_classes, metrics=METRICS, **hyperparameters)")

                model.summary()

                try:
                    # Train and evaluate the current model on the dataset. Save the trained models and histories
                    model, history = evaluate_model(model, X_train, y_train, X_val, y_val, patience=patience,
                                                    _epochs=epochs, _save_name=save_name, _log_dir=log_dir)
                    scores = model.evaluate(X_test, y_test, verbose=1)

                    acc_per_model[mod_name] = scores[1] * 100
                    loss_per_model[mod_name] = scores[0]

                    trained_models[mod_name] = model
                    model_histories[mod_name] = history

                except Exception as er:
                    print(f"################# Oh Man! An error occurred. #################\n{er}")
                    pass
                else:
                    print("Nothing to see here, move on")
                    pass

                # **************** Time to summarize the model's performance ****************
                report.write(f"{mod_name} Model : {datetime.now()}\n")
                # Training History
                report.write(f"Model History \n{pd.DataFrame(history.history)}\n\n")
                model_str = []
                model.summary(print_fn=lambda x: model_str.append(x))
                report.write("\n".join(model_str))
                report.write('\n\n')
                report.write("+++Hyperparameters+++\n")
                report.write(f"Number of Epochs: {epochs}\n")
                report.write(f"Batch Size: {batch_size}\n")
                for key, item in hyperparameters.items():
                    report.write(f"{key.replace('_', ' ').capitalize()}: {item}\n")
                report.write('\n\n')
                report.write(f"Test Accuracy: {scores[1] * 100}\n")
                report.write(f"Test Loss: {scores[0]}\n")
                report.write('\n\n\n')

                # let's plot our training history
                plot_metrics(history, mod_name, dataset, f"{img_path}/{mod_name}_history.png")

                # Results for each model
                predictions = model.predict(X_test).argmax(1)

                classificationReport = classification_report(y_test.argmax(1), predictions,
                                                             labels=np.unique(y_test.argmax(1)), target_names=labels)
                print(classificationReport)
                report.write(f"Classification Report\n{classificationReport}\n\n\n")

                print("Confusion Matrix:")
                cm = confusion_matrix(y_test.argmax(axis=1), predictions)
                print(cm)
                report.write(f"Confusion Matrix\n{cm}\n\n\n")

                print("Normalised Confusion Matrix: True")
                normalised_confusion_matrix = np.around(confusion_matrix(y_test.argmax(axis=1),
                                                                         predictions,
                                                                         normalize='true') * 100, decimals=2)
                print(normalised_confusion_matrix)
                normalised_confusion_matrix_df = pd.DataFrame(normalised_confusion_matrix, index=labels, columns=labels)
                report.write(f"Normalised Confusion Matrix: True\n{normalised_confusion_matrix_df}\n\n\n")
                sns.heatmap(normalised_confusion_matrix_df, cmap='rainbow')
                plt.title("Confusion matrix\n(normalised to % of total test data)")
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
                plt.savefig(f"{img_path}/{model_name}_confusion_matrix.png", bbox_inches='tight')
                plt.show()

                print("Finished working on: ", mod_name, " at: ", datetime.now(), " -> ", time() - s)
                report.write(f"Finished working on: {mod_name} at: {datetime.now()} -> {time() - s}\n\n")
    except Exception as ex:
        print("Caught a tensorflow exception: ", ex)
        continue

    print("Finished working on: ", list_of_models, " at: ", datetime.now(), " -> ", time() - start)

print('**Finished**')

with open(report_name, "a") as rep:
    # Comparing the trained models
    accuracy_dataframe = pd.Series(acc_per_model)
    loss_dataframe = pd.Series(loss_per_model)

    rep.write(f"Accuracy comparison \n{pd.DataFrame(accuracy_dataframe)}\n\n")
    rep.write(f"Loss comparison \n{pd.DataFrame(loss_dataframe)}\n\n")
