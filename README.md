# multi-task-speech-classification
Multi-Task Speech classification of accent and gender of an english speaker on [Mozilla's common voice dataset](https://www.kaggle.com/mozillaorg/common-voice).
Paper can be found [here](200420608-Multi_task_speech_classification.pdf)

# Run instructions
1. To `preprocess` the audio data, run 
   ```shell
   python main.py preprocess -r <audio_data_path>
   ```

2. To `train` the model using the preprocessed audio data, run
    ```shell
    python main.py train -r <audio_data_path> -m <model_name> 
    ```
   **Models Implemented:** simple_cnn, resnet18, resnet34, resnet50, simple_lstm, bi_lstm, lstm_attention, bi_lstm_attention

3. To `test` the model on the test data, run
    ```shell
    python main.py test -r <audio_data_path> -m <model_name> -c <saved_model_path> 
    ```

4. To perform `inference` on the audio files directly, run
    ```shell
    python main.py inference -r <audio_files_path> -m <saved_model_path>
    ```
