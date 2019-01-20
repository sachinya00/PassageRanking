# Architecture
## PassageRanking
* BERT Model Layer( Bert-Base-Uncased ) - See bert_config.json for bert configuration.
    * N - Max Sequence Length = 128
    * C - Batch Size = 32
    * Hidden_Size - Bert Hidden Size=768
    * IN:
        * Input Ids 
            * (Shape: (C*10) x N) 
            * (Single Input ex: [CLS] query Ids [SEP] Passage Ids [SEP])
        * Segment Ids [Separate query Ids from passage Ids]
 
            * (ex: [query Id, query Id, passage Id] -> [1,1,0])
            * (Shape: (C*10) x N)

        * Query Mask [For Handling Variable length sequences]

            * (ex: [Id, Id, Id, 0, 0] -> [1, 1 1, 0, 0])
            * (Shape: (C*10) x N)

        * Labels [Passage Instance Number 0-9]

            * (Shape: (C*10) x 1)

   * OUT: Pooled Output
        
        * (Shape : (C*10) x Hidden_Size)

* Dropout Layer - p=0.5

    * IN: (Shape : (C*10) x Hidden_Size)
    * OUT: (Shape : (C*10) x Hidden_Size)

* Classifier - Fully Connected Layer (Linear Layer)

    * IN: (C*10) x Hidden_Size 
    * OUT: (C*10) x 1

* Output reshaped to C x 10 x 1


# Instructions to run evaluation
* **pip install -r requirements.txt** (Preferably in a virtual env)
* Put the Test File (.tsv) inside data folder
* Download the models from [Link] (https://drive.google.com/open?id=1XsS_1zjRtR06XADG7xFNk_a6JDUtDrEv)
        and place them inside models folder.
* In **test_model.py** change the **data_name** to name of the test file in options dictionary.
* Run **test_model.py**
* The resulting file **answer.tsv** will be created in the **output** folder.

**NOTE**: Increasing the batch size greater than **32** on a **NVIDIA Tesla K80 GPU** results in inaccurate predictions. Keep the batch size equal to or below 32 on K80. If you get a very low score reduce the batch size.