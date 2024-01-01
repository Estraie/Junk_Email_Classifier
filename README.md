# Junk_Email_Classifier

Create a new conda environment and install the requirements

```bash
conda create -n jec
```
 
```bash
conda activate jec
```

```bash
pip install -r requirements.txt
```
## File Description

The `train_cn/en.py` includes the training and testing for 4 model on trec06c/p, and `test_cn/en.py` includes tests on customed data. 
<br>
`rff_dims.sh` conduct exploration on the influence caused by dims of [random fourier features](https://github.com/tiskw/random-fourier-features)

## TODO

- [ ] Get to know how to train a model
