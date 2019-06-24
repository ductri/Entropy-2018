Requirement: install all required packages:
```
RUN pip install ruamel.yaml
RUN pip install nltk
RUN pip install pyvi
RUN pip install joblib
```
- Create `input.csv` file, put it in `input/`
`input.csv` file should have only 1 columns containing sentences/posts

- Run `./start_predict.sh` to predict

- Result will be stored as `output/output.csv`


