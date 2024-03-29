# KGLink
Code of paper: KGLink: A column type annotation method that combines Knowledge Graph and pre-trained language model

##### Main Requirements:
Python==3.6  
pytorch=1.10.2  
spacy==3.2.4  
transformers==4.18.0  
scikit-learn==0.24.2  
scipy==1.5.4  
networkx==2.5.1  
jsonlines

##### Code Usage:
Link the table with KB

```
python find_near.py --dataset_name <iswc or viznet in our case>
```

The linked KG knowledge for these 2 datasets is provided with size of 10 per cell:
[Semtab2019](https://hkustconnect-my.sharepoint.com/:u:/g/personal/ywangnx_connect_ust_hk/EccsTGgIbO9Mpz-EgKuInbcBypnZEQEc7EVGLVF13MIxRw?e=Bf3pS7)
[Viznet](https://hkustconnect-my.sharepoint.com/:u:/g/personal/ywangnx_connect_ust_hk/EVR1F6SqxJ1EteoBfPb23n4BR_ZJAO-Vs4lAGflxLMcSjA?e=WRF7Am)

We used the Elasticsearch to index the Wikidata KG. You may refer [here](https://github.com/Zinc-30/wikidata_es_index) to set up Wikidata on Elasticsearch.

Link tables to KG:

```
python find_near.py 
```

Filter the knowledge extracted:

```
python source_col_filter.py --dataset_name <iswc or viznet in our case> --filter_size <25 in our size>
```



The dataset with generated candidate types for these 2 dataset are provided with 25 rows per table, you may download and put into the data folder:


[Semtab2019](https://hkustconnect-my.sharepoint.com/:u:/g/personal/ywangnx_connect_ust_hk/EZsDhOj-_WVHqnoC6z4pjLUB_IPISAphcCSsVQwE9_UxGQ?e=fIFagD)
[Viznet](https://hkustconnect-my.sharepoint.com/:u:/g/personal/ywangnx_connect_ust_hk/EUkie95dLndFoSgXsodC1CsB7X0Z0XlYuTv2ZGvJvixrVw?e=kfQdC6)


[Semtab2019_Label](https://hkustconnect-my.sharepoint.com/:u:/g/personal/ywangnx_connect_ust_hk/Edk6OjMqTmJJs_vJUGz4Yh4BMy7Iaw2VMJ28JkKxIA7ezw?e=CGPcNv)
[Viznet_Label](https://hkustconnect-my.sharepoint.com/:u:/g/personal/ywangnx_connect_ust_hk/EeEYQaK8RkFNtMLqYE1DDBcBbubN9pSmzjHuQxvUTKRmUw?e=4ThT5X)

You may use the above-provided files to skip the previous steps.

Generate the feature vectors:

```
python feature_vec.py --dataset <iswc or viznet in our case>
```

Construct dataset:
```
python dataset_multicol.py --dataset_name <iswc or viznet in our case>
```

Start training and validating:
```
python main.py \
    --gpu_count <How many GPU used, automatically selected, if --manual_GPU, is the number of GPU used> \
    --lr <learning rate> \
    --batch_size <batch size> \
    --dataset_name <iswc or viznet in our case> \
    --learn_weight <adaptly learn the weight between loss> \
    --LM <Language model used, default bert, deberta and roberta are also supported> \
    --seed <Random seed> \
    --manual_GPU <Whether to manually select GPU to use> \
    --exp_name <experiment name used iswc if semtab dataset, viznet if viznet dataset>
    --feature_vec <Whether to use feature vector for the model>
    --drop_out <Set to 0.1 if on SemTab dataset, 0.2 for default>
```
For Semtab:
```
python main.py --gpu_count 1 --end_fix iswc --learn_weight --feature_vec
```
For Viznet:
```
python main.py --gpu_count 1 --end_fix viznet --learn_weight --feature_vec
```
### Experiment with GPT3
We also provided the GPT3 version of our model, which substitutes the BERT encoder with GPT3. We placed the codes in the GPT3 folder. You may substitute the OpenAI API in gpt3_emb.py to generate embedding with GPT3 and use gpt3.py to run the code.

### Acknowledgment:
Some of our code is from the [Tabbie code base](https://github.com/SFIG611/tabbie). We sincerely thank them for their contributions.


