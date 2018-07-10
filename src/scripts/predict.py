#!/usr/bin/env python
"""
Given a corpus of features between two US presidential candidates, predicts which one is most likely to win.
"""
from __future__ import print_function
import os
import sys
import csv
import copy
from collections import defaultdict
import tempfile
import commands
# from pprint import pprint

from weka.arff import ArffFile, Nom, Num, Int, MISSING
from weka.classifiers import Classifier, WEKA_CLASSIFIERS

import xlsxwriter

DEMOCRAT = 'Democrat'
REPUBLICAN = 'Republican'

PROJECT_DIR = os.path.abspath(os.path.split(__file__)[0])
DATA_DIR = os.path.join(PROJECT_DIR, '../../data')
FIXTURES_DIR = os.path.join(PROJECT_DIR, '../fixtures')

# The data we used for the 2016 election.
#PRESIDENTIAL_CANDIDATES_SOURCE = 'presidential_candidates_2016.ods'

# The data we use for all future elections.
# PRESIDENTIAL_CANDIDATES_SOURCE = 'presidential_candidates_current.ods'

# Experimental data.
PRESIDENTIAL_CANDIDATES_SOURCE = 'presidential_candidates_2016_extended.ods'

OUTPUT_FN = 'all-results.csv'

def spreadsheet_to_csv(fn):
    """
    Converts an arbitrary spreadsheet file to a comma-delimited-values file.
    """

    assert os.path.isfile(fn)

    _, outfn = tempfile.mkstemp()
    os.remove(outfn)
    outfn = outfn + '.csv'

    cmd = 'ssconvert "%s" "%s"' % (fn, outfn)
    print('Converting %s to %s...' % (fn, outfn))
    print(cmd)
    status, output = commands.getstatusoutput(cmd)
    assert os.path.isfile(outfn), 'Output file "%s" was not generated:\n\n%s' % (outfn, output)

    return outfn


def validate_line(line, prefix=''):
    new_line = {}

    ignore_fields = set([
        'First Name',
        'Middle Name',
        'Last Name',
        'Election',
        'Party',
        'Wikipedia',
        'Born',
        'Additional Source 1',
        'Additional Source 3',
        'Additional Source 2',
    ])

    states = ['IA', 'NH', 'SC', 'NV', 'OH']
    for state in states:
        if line['Won %s' % state] != MISSING:
            int(line['Won %s' % state])
        new_line['Won %s' % state] = Nom(line['Won %s' % state])
        new_line['Vote %s' % state] = Num(line['Vote %s' % state])
        new_line['Win Margin %s' % state] = Num(line['Win Margin %s' % state])

        # pprint(line, indent=4)

        # Sanity check the boolean column for having won the state vs the column
        # for the win margin.
        # If win=1 then margin should be positive. Otherwise, margin should be negative.
        won_state = line['Won %s' % state]
        win_margin = line['Win Margin %s' % state]
        if won_state == MISSING or win_margin == MISSING:
            # Don't validate won_state if we don't know the margin.
            continue
        won_state = int(won_state)
        win_margin = float(win_margin)

        err_msg = None
        if (won_state and win_margin < 0) or (not won_state and win_margin > 0):
            err_msg = 'Invalid column: Won %s = %s but win margin = %s' % (state, won_state, win_margin)
        if err_msg:
            print(err_msg, file=sys.stderr)
            sys.exit(1)

    new_line['Military'] = Nom(int(line['Military']))
    new_line['More Military'] = Nom(int(line['More Military']))

    new_line['Was Governor'] = Nom(int(line['Was Governor']))
    new_line['Was Businessman'] = Nom(int(line['Was Businessman']))
    new_line['Was Senator'] = Nom(int(line['Was Senator']))
    new_line['Was Representative'] = Nom(int(line['Was Representative']))
    new_line['Was VP'] = Nom(int(line['Was VP']))
    new_line['Was AG'] = Nom(int(line['Was AG']))
    new_line['Was Secretary of State'] = Nom(int(line['Was Secretary of State']))

    new_line['Incumbent'] = Nom(int(line['Incumbent']))

    new_line['Younger Than Opponent'] = Nom(int({'TRUE':1, 'FALSE':0}[line['Younger Than Opponent']]))
    new_line['Of Party Last In Office'] = Nom(int(line['Of Party Last In Office']))
    new_line['Years Since Party In Office'] = Nom(int(line['Years Since Party In Office']))

    new_line['Age At Election'] = Int(int(line['Age At Election']))
    new_line['Age Difference'] = Int(int(line['Age Difference']))
    new_line['Incumbent Job Approval Rating'] = Int(int(line['Incumbent Job Approval Rating']))

    new_line['ID'] = Nom(line['ID'])

    #new_line['Election'] = Nom(int(line['Election']))

    #new_line['Party'] = Nom(line['Party'])

    new_line['TV Debates'] = Int(int(line['TV Debates']))
    new_line['Most Charismatic'] = Int(int(line['Most Charismatic']))

    new_line['Won'] = Nom(int(line['Won']) if line['Won'] else MISSING, cls=True)

    all_fields = set(line)
    all_fields = all_fields.difference(new_line)
    all_fields = all_fields.difference(ignore_fields)
    if all_fields:
        raise Exception, 'Unprocessed fields: %s' % all_fields

    # print('A1')
    # pprint(new_line, indent=4)
    if prefix:
        for key in list(new_line):
            new_line[prefix+' '+key] = new_line[key]
            del new_line[key]
    # print('A2')
    # pprint(new_line, indent=4)

    return new_line


def read_raw_csv(filename):
    """
    Combines separate rows of Dem/Rep into a single line with the attributes of each prefixed by party name.
    """

    cache = defaultdict(list) # {year: [line]}

    i = 0
    for line in sorted(csv.DictReader(open(filename)), key=lambda o: int(o['Election']), reverse=True):
        i += 1
        year = int(line['Election'])
        party = line['Party'].title()
        print('Validating line %i...' % i)
        # Note, the line should be returned with all Excel field prefixed with the party name.
        # For example the "Won" column should be returned as "Democrat Won" if the "Party" column contains "Democrat".
        line = validate_line(line, prefix=party)
        # pprint(line, indent=4)
        cache[year].append(line)

    for year, data in sorted(cache.items(), reverse=True):
        if len(data) == 2:
            final_data = {}
            for line in data:
                final_data.update(line)
            dem_won = final_data['Democrat Won']
            rep_won = final_data['Republican Won']
            del final_data['Democrat Won']
            del final_data['Republican Won']

            # Remove other useful fields.
            del final_data['Republican ID']
            del final_data['Democrat ID']

            final_data['Won'] = Nom(MISSING, cls=True)
            if dem_won.value != MISSING and rep_won.value != MISSING:
                if dem_won.value:
                    final_data['Won'] = Nom('Democrat', cls=True)
                else:
                    final_data['Won'] = Nom('Republican', cls=True)

            final_data['Election'] = Nom(year)

            yield final_data


def walk_classifier(name, ckargs=None):

    evaluation_sets = {} # {year: [training_list, test_list, expected_list]}

    final_query_lines = []

    # Build training data.
    #training_data = ArffFile(relation='presidential-candidates')
    fn = os.path.join(FIXTURES_DIR, PRESIDENTIAL_CANDIDATES_SOURCE)
    print('Reading %s...' % fn)
    fn2 = spreadsheet_to_csv(fn)
    i = 0
    for line in read_raw_csv(fn2):
        i += 1
        print('line:', i, line)
        #line = validate_line(line)

        year = int(line['Election'].value)
        #print(year

        evaluation_sets.setdefault(year, [[], [], []])

        if line['Won'].value != MISSING:
            #training_data.append(line)

            # Add line to test set.
            test_line = copy.deepcopy(line)
            evaluation_sets[year][2].append(test_line['Won'].value)
            test_line['Won'].value = MISSING
            evaluation_sets[year][1].append(test_line)

            # Add line to all future sets.
            for other_year in evaluation_sets:
                if year < other_year:
                    evaluation_sets[other_year][0].append(line)

        else:
            final_query_lines.append(line)

    accuracy = []

    final_training_data = None
    final_year = None

    # Evaluate each evaluation set.
    #pprint(evaluation_sets, indent=4)
    print('%i evaluation_sets.' % len(evaluation_sets))
    for year, data in sorted(evaluation_sets.iteritems()):
        raw_training_data, raw_testing_data, prediction_values = data
        print('Evaluation set:', year, len(raw_training_data), len(raw_testing_data), len(prediction_values))

        if not raw_training_data:
            print('No training data. Skipping.')
            continue

        # Create training set.
        training_data = ArffFile(relation='presidential-candidates')
        for _line in raw_training_data:
            training_data.append(_line)
        training_data.attribute_data['Won'].update([DEMOCRAT, REPUBLICAN])
        training_data.write(open('training_data_%i.arff' % year, 'w'))

        if not raw_testing_data:
            final_training_data = training_data
            final_year = year
            print('No testing data. Skipping.')
            continue

        # Create query set.
        query_data = training_data.copy(schema_only=True)
        for _line in raw_testing_data:
            query_data.append(_line)
        query_data.write(open('query_data_%i.arff' % year, 'w'))

        # Train
        print('='*80)
        c = Classifier(name=name, ckargs=ckargs)
        print('Training...')
        c.train(training_data, verbose=True)

        # Test
        print('Predicting...')
        predictions = c.predict(query_data, verbose=True, distribution=True)
        print('predictions:')
        for predicted_value, actual_value in zip(predictions, prediction_values):
            print('predicted_value =', predicted_value, 'actual_value =', actual_value)
            accuracy.append(predicted_value.predicted == actual_value)

    print('-'*80)
    accuracy_history = accuracy
    if accuracy:
        accuracy = sum(accuracy)/float(len(accuracy))
    else:
        accuracy = None
    print('accuracy_history:', accuracy_history)
    print('accuracy:', accuracy)

    # Make final prediction.
    predicted_cls = None
    certainty = None
    if final_training_data:

        # Create final query set.
        query_data = final_training_data.copy(schema_only=True)
        for _line in final_query_lines:
            query_data.append(_line)
        query_data.write(open('query_data_%i.arff' % year, 'w'))

        # Train
        print('!'*80)
        c = Classifier(name=name, ckargs=ckargs)
        print('Final Training...')
        c.train(final_training_data, verbose=True)

        # Test
        print('~'*80)
        print('Final Predicting...')
        predictions = c.predict(query_data, verbose=True, distribution=True)
        print('final predictions:')
        for predicted_value in predictions:
            print('predicted_value:', predicted_value)
            with open('prediction_%i_%s.txt' % (year, name), 'w') as fout:
                print('stdout:', file=fout)
                print(c.last_training_stdout, file=fout)
                print(file=fout)
                print('stderr.begin:', file=fout)
                print(c.last_training_stderr, file=fout)
                print('stderr.end:', file=fout)
                print(file=fout)
                print('predicted_value.probability:', predicted_value.probability, file=fout)
                predicted_cls = predicted_value.predicted
                certainty = predicted_value.certainty

    else:
        raise Exception('No final training data! Are there no empty "won" columns?')

    return accuracy, predicted_cls, certainty


def main(stop_on_error=False, **kwargs):
    # names = [
    #     dict(name='weka.classifiers.lazy.IBk', ckargs={'-K':1}),
    # ]
    names = [dict(name=_name) for _name in WEKA_CLASSIFIERS]
    # names = [dict(name='weka.classifiers.trees.ADTree')]#TODO:remove
    fieldnames = ['Name', 'Accuracy', 'Predicted', 'Certainty', 'Error', 'Dem Score', 'Rep Score']
    os.chdir(DATA_DIR)
    error_count = 0
    xl_fn = 'all-results.xlsx'
    workbook = xlsxwriter.Workbook(xl_fn)
    worksheet = workbook.add_worksheet()
    row = 0
    for col, fieldname in enumerate(fieldnames):
        worksheet.write(row, col, fieldname)
    row += 1
    with open(OUTPUT_FN, 'w') as fout:
        results = csv.DictWriter(fout, fieldnames=fieldnames)
        results.writerow(dict(zip(fieldnames, fieldnames)))
        for kwargs in names:
            acc, pred_cls, cert, error = '?', '?', '?', ''
            try:
                acc, pred_cls, cert = walk_classifier(**kwargs)
            except Exception as e:
                if stop_on_error:
                    raise
                error_count += 1
                print('Error!', e, file=sys.stderr)
                error = str(e).strip().split('\n')[0]
            print('acc, pred_cls, cert=', acc, pred_cls, cert)
            if acc != MISSING:
                results.writerow(dict(
                    Name=kwargs['name'],
                    Accuracy=acc,
                    Predicted=pred_cls,
                    Certainty=cert,
                    Error=error,
                ))
                print('row:', row)
                worksheet.write(row, 0, kwargs['name'])
                worksheet.write(row, 1, acc)
                worksheet.write(row, 2, pred_cls)
                worksheet.write(row, 3, cert)
                worksheet.write(row, 4, error)
                print('col:', xlsxwriter.utility.xl_col_to_name(5))
                worksheet.write_formula('%s%i' % (xlsxwriter.utility.xl_col_to_name(5), row+1), '=IF(C{i}="Democrat", B{i}, (1-B{i}))'.format(i=row+1))
                worksheet.write_formula('%s%i' % (xlsxwriter.utility.xl_col_to_name(6), row+1), '=IF(C{i}="Republican", B{i}, (1-B{i}))'.format(i=row+1))
                row += 1
            fout.flush()
    worksheet.write_formula('%s%i' % (xlsxwriter.utility.xl_col_to_name(5), row+1), '=AVERAGE(F2:F{i})'.format(i=row))
    worksheet.write_formula('%s%i' % (xlsxwriter.utility.xl_col_to_name(6), row+1), '=AVERAGE(G2:G{i})'.format(i=row))
    workbook.close()
    print('Results written to %s/%s.' % (DATA_DIR, OUTPUT_FN))
    if error_count:
        print('Some errors were encountered. If these are coming from a handful of classifiers that could not handle the data, you can ignore these.')
        sys.exit(1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stop-on-error', action='store_true', default=False)
    args = parser.parse_args()
    main(**args.__dict__)
