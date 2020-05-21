### Sentiment Analysis
Dataset format: 
```
qid	text	label
110686	不好意思3个月没玩游戏有点那个	1
200003	肯定会从我下手 来损我	0
...
```
Entries separated by `\t`. Label 1 for negative, 0 for non-negative. The dataset is not included in this repo.  
<br>
To evaluate different models' performance, run
```
> python train.py --method {logistic_regression,svm,random_forest,multinomial_nb}
```

Train the final model and save it, run
```
> python train_final.py
```

To predict on test set and write the result to output file, run
```
> python main_test.py -m model_file -d data_file -o output_file
```