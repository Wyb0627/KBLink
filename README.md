# KBLink
Code of paper: KBLink: A column type annotation method that combines Knowledge Graph and pre-trained language model

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



Filter the knowledge extracted:

```
python source_col_filter.py --dataset_name <iswc or viznet in our case> --filter_size <25 in our size>
```

The dataset with generated candidate types for these 2 dataset are provided with 25 rows per table:
[Semtab2019](https://hkustconnect-my.sharepoint.com/:u:/g/personal/ywangnx_connect_ust_hk/EZsDhOj-_WVHqnoC6z4pjLUB_IPISAphcCSsVQwE9_UxGQ?e=fIFagD)
[Viznet](https://hkustconnect-my.sharepoint.com/:u:/g/personal/ywangnx_connect_ust_hk/EUkie95dLndFoSgXsodC1CsB7X0Z0XlYuTv2ZGvJvixrVw?e=kfQdC6)
[Semtab2019 Label](https://hkustconnect-my.sharepoint.com/:u:/g/personal/ywangnx_connect_ust_hk/Edk6OjMqTmJJs_vJUGz4Yh4BMy7Iaw2VMJ28JkKxIA7ezw?e=CGPcNv)
[Viznet Label](https://hkustconnect-my.sharepoint.com/:u:/g/personal/ywangnx_connect_ust_hk/EeEYQaK8RkFNtMLqYE1DDBcBbubN9pSmzjHuQxvUTKRmUw?e=4ThT5X)

You may use the above-provided files to directly run the dataset_multicol.py to avoid linking to KG by yourself.

Construct dataset:
```
python dataset_multicol.py --dataset_name <iswc or viznet in our case>
```

Start training and validating:
```
python main.py \
    --gpu_count <number of GPU used> \
    --label_count <number of labels, 275 for semtab, 77 for Viznet> \
    --lr <learning rate> \
    --epochs <num of epochs> \
    --batch_size <batch size> \
    --dataset_name <iswc or viznet in our case> \
    --learn_weight <adaptly learn the weight between loss>
```

### Acknowledgment:
Some of our code is from the [Tabbie code base](https://github.com/SFIG611/tabbie). We sincerely thank them for their contributions.


