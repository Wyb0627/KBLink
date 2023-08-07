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

The linked knowledge for theses 2 dataset are provided with size 10 per cell:
[Semtab2019](https://hkustconnect-my.sharepoint.com/:u:/g/personal/ywangnx_connect_ust_hk/EccsTGgIbO9Mpz-EgKuInbcBypnZEQEc7EVGLVF13MIxRw?e=Bf3pS7)
[Viznet](https://hkustconnect-my.sharepoint.com/:u:/g/personal/ywangnx_connect_ust_hk/EVR1F6SqxJ1EteoBfPb23n4BR_ZJAO-Vs4lAGflxLMcSjA?e=WRF7Am)

Filter the knowledge extracted:

```
python source_col_filter.py --dataset_name <iswc or viznet in our case> --filter_size <25 in our size>
```

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

#### Acknowledgment:
Some of our code is from the [Tabbie code base](https://github.com/SFIG611/tabbie). We sincerely thank them for their contributions.


