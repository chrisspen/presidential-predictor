Attempts to predict the next US President by running a corpus of historical metrics for each candidate through all available Weka classifiers.

The prediction of each classifier, as well as the classifier's historical accuracy is then recorded to a spreadsheet.

As new data becomes available, update:

    src/fixtures/presidential_candidates_{year}.ods

and then to regenerate the predictions, run:

    ./init_virtualenv.sh
    ./run.sh

The results will be written to `data/all-results-{year}.csv`.
