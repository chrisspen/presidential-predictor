#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import csv
import copy
from pprint import pprint
from collections import defaultdict

from weka.arff import ArffFile, Nom, Num, Int, MISSING
from weka.classifiers import Classifier, WEKA_CLASSIFIERS

DEMOCRAT = 'Democrat'
REPUBLICAN = 'Republican'

PROJECT_DIR = os.path.abspath(os.path.split(__file__)[0])
DATA_DIR = os.path.join(PROJECT_DIR, '../../data')
FIXTURES_DIR = os.path.join(PROJECT_DIR, '../fixtures')

#PRESIDENTIAL_CANDIDATES_SOURCE = 'presidential_candidates_2016.ods'
PRESIDENTIAL_CANDIDATES_SOURCE = 'presidential_candidates_current.ods'

def spreadsheet_to_csv(fn):
    """
    Converts an arbitrary spreadsheet file to a comma-delimited-values file.
    """
    import tempfile, commands
    
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
        if line['Won %s' % state] != MISSING: int(line['Won %s' % state])
        new_line['Won %s' % state] = Nom(line['Won %s' % state])
        new_line['Vote %s' % state] = Num(line['Vote %s' % state])
        new_line['Win Margin %s' % state] = Num(line['Win Margin %s' % state])
        
        # Sanity check the boolean column for having won the state vs the column
        # for the win margin.
        # If win=1 then margin should be positive. Otherwise, margin should be negative.
        won_state = int(line['Won %s' % state])
        win_margin = float(line['Win Margin %s' % state])
#         print('state:', state)
#         print('won_state:', won_state)
#         print('win_margin:', win_margin)
        if won_state:
            if win_margin < 0:
                print('Invalid column:', 'Won %s' % state, file=sys.stderr)
                sys.exit(1)
        else:
            if win_margin > 0:
                print('Invalid column:', 'Won %s' % state, file=sys.stderr)
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
    
    new_line['Won'] = Nom(int(line['Won']) if line['Won'] else MISSING, cls=True)
    
    all_fields = set(line)
    all_fields = all_fields.difference(new_line)
    all_fields = all_fields.difference(ignore_fields)
    if all_fields:
        raise Exception, 'Unprocessed fields: %s' % all_fields
    
    if prefix:
        for key in new_line.keys():
            new_line[prefix+' '+key] = new_line[key]
            del new_line[key]
    
    return new_line

def read_raw_csv(filename):
    """
    Combines separate rows of Dem/Rep into a single line with the attributes of each prefixed by party name.
    """
    
    cache = defaultdict(list) # {year: [line]}
    
    for line in sorted(csv.DictReader(open(filename)), key=lambda o: int(o['Election']), reverse=True):
        year = int(line['Election'])
        party = line['Party'].title()
        line = validate_line(line, prefix=party)
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
    for year, data in sorted(evaluation_sets.iteritems()):
        raw_training_data, raw_testing_data, prediction_values = data
        print(year, len(raw_training_data), len(raw_testing_data), len(prediction_values))
        
        if not raw_training_data:
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
            print(predicted_value, actual_value)
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
        print('='*80)
        c = Classifier(name=name, ckargs=ckargs)
        print('Final Training...')
        c.train(final_training_data, verbose=True)
         
        # Test
        print('Final Predicting...')
        predictions = c.predict(query_data, verbose=True, distribution=True)
        print('final predictions:')
        for predicted_value in predictions:
            print(predicted_value)
            with open('prediction_%i_%s.txt' % (year, name), 'w') as fout:
                print('stdout:', file=fout)
                print(c.last_training_stdout, file=fout)
                print(file=fout)
                print('stderr:', file=fout)
                print(c.last_training_stderr, file=fout)
                print(file=fout)
                print(predicted_value.probability, file=fout)
                predicted_cls = predicted_value.predicted
                certainty = predicted_value.certainty
    
    return accuracy, predicted_cls, certainty

# names = [
#     dict(name='weka.classifiers.lazy.IBk', ckargs={'-K':1}),
# ]
names = [dict(name=_name) for _name in WEKA_CLASSIFIERS]
fieldnames = ['name', 'accuracy', 'predicted', 'certainty', 'error']
os.chdir(DATA_DIR)
with open('all-results.csv', 'w') as fout:
    results = csv.DictWriter(fout, fieldnames=fieldnames)
    results.writerow(dict(zip(fieldnames, fieldnames)))
    for kwargs in names:
        acc, pred_cls, cert, error = '?', '?', '?', ''
        try:
            acc, pred_cls, cert = walk_classifier(**kwargs)
        except Exception as e:
            print(e)
#             raise
            error = str(e).strip().split('\n')[0]
        results.writerow(dict(
            name=kwargs['name'],
            accuracy=acc,
            predicted=pred_cls,
            certainty=cert,
            error=error,
        ))
        fout.flush()
