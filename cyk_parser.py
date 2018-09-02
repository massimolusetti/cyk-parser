#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# UZH - Parsing Technology - Spring Semester 2017
# Author: Massimo Lusetti
# Student ID: 15-706-435

"""
This module performs syntactic parsing of sentences based on dependencies
by implementing a CYK algorithm. The user can choose whether to use a
default grammar or learn a new grammar from a training set extracted from
the Penn Treebank. If a gold standard is available (i.e. if the test
sentence is extracted from the Penn Treebank and not entered directly by the
user) the evaluation is carried out by computing the UAS of each sentence.

Run from the command line:
    python3 mlusetti_parser.py

It is important to make sure that the directory 'postagger' is located
in the same directory where the file mlusetti_parser.py is saved.
"""

import os
import re
import time
import json
import random
from collections import OrderedDict, defaultdict
from nltk.corpus import treebank
from nltk.tokenize import word_tokenize
from nltk.tag.stanford import StanfordPOSTagger
import StanfordDependencies


def output_files(dict_rules):
    """
    Creates output file to store the grammar learned in the training phase.
    This function is called from the function learn_grammar.

    Parameters:
        - dict_rules: a dictionary that stores the rules learned in the training phase.

    Output:
        - the grammar in .json format.
    """
    output_file = open('grammar_new.json', 'w')
    json.dump(dict_rules, output_file, ensure_ascii=False, indent=2)
    output_file.close()


def learn_grammar(training_set, sd):
    """
    Learns a grammar based on the training set extracted from the whole data set.
    This function is called from the main function.

    Parameters:
        - training_set: the set with the sentences used to learn the grammar.
        - sd: the module used to convert the parsing from constituency into dependencies.

    Output:
        - during the training phase, prints on the console for each sentence the
          constituency parsing, the dependency parsing in conll format and a
          list of the rules learned.

    Returns:
        - dict_rules: a dictionary with the learned grammar.
    """
    dict_rules = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # Loop over all the fileids of the Penn Treebank
    for i, fileid in enumerate(training_set):
        print('\n------------------- Fileid {}: {} ----------------------'.format(i+1, fileid))
        print()
        # Extract the parsed sentences from the Penn Treebank
        penn_sent = treebank.parsed_sents(fileid)[0]
        print(penn_sent)
        # Convert the syntax annotation from constituency into dependencies
        dep_sent = sd.convert_tree(str(penn_sent))
        list_tokens = []
        for token in dep_sent:
            print(token)
            pos = token[4] # Part of speech of the token
            head = token[6] # Index of the head of the token
            list_tokens.append((pos, head))
        print()
        print(list_tokens)
        print()
        # Loop over all the tuples that contain the pos and the head of each token
        for k, pos_head in enumerate(list_tokens):
            pos = pos_head[0]
            head = pos_head[1]
            # If the token is the head of the whole sentence
            if head == 0:
                relation = pos, '-'
            else:
                relation = pos, list_tokens[(head-1)][0]
                if k < head:
                    direction = 'left'
                    print('From index {} to index {}, direction: {}'
                          .format(k+1, pos_head[1], direction))
                    print(relation, ' -> ', relation[1])
                    key = ' '.join(relation)
                    dict_rules[key][relation[1]]['Right'] += 1
                else:
                    direction = 'right'
                    print('From index {} to index {}, direction: {}'
                          .format(k+1, pos_head[1], direction))
                    relation_right = relation[1], relation[0]
                    print(relation_right, ' -> ', relation[1])
                    key = ' '.join(relation_right)
                    dict_rules[key][relation[1]]['Left'] += 1
        # Use the following 'if' statement to break the learning process after
        # the specified number of sentences (replace 'pass' with 'break'):
        if i == 10:
            pass

    print()
    print('-'*50)
    print()
    #Eliminate from the grammar all the rules which occur only once
    dict_rules_new = {}
    for key, value in dict_rules.items():
        one_rule = 0
        for value1 in value.values():
            if (value1['Left'] in [0, 1] and value1['Right'] in [0, 1]):
                one_rule += 1
        if one_rule == len(value):
            continue
        dict_rules_new[key] = value

    output_files(dict_rules_new)

    return dict_rules_new


def conll_format(parse, dict_parse=None):
    """
    Converts a flattened nested list, which represents the structure of the parsed sentence,
    into conll format, in order to enable a comparison with the gold standard.
    This function is recursive and reduces each pair of head and dependent to the
    head. The recursion stops when only one token remains, which is the head
    of the whole sentence.
    This function is called from the function parse_sentence.

    Parameters:
        - parse: the flattened nested list representing the parsing.
        - dict_parse: a dictionary that stores the conll format of the parsed sentence.
          The dictionary is initialized as default parameter and then populated through
          recursive calls of the function.

    Output:
        - prints on the console the conll format of the parsed sentence.

    Returns:
        - dict_parse_final: an ordered dictionary that stores the conll
          format of the parsed sentence, sorted by index of the tokens.
    """
    if dict_parse is None:
        dict_parse = defaultdict(lambda: defaultdict(str))
    two_items = []
    two_pos = []
    two_index = []
    for item in parse:
        split_item = item.split('_')
        if split_item[0] in two_pos:
            found = 'no'
            for element in two_pos:
                if element == split_item[0]:
                    ind = two_pos.index(split_item[0])
                    if found == 'yes':
                        ind2 = 1
                    found = 'yes'
                else:
                    ind2 = two_pos.index(element)

            if two_pos[0] == two_pos[1]:
                if split_item[1] == 'Right':
                    dict_parse[two_index[ind]]['form'] = two_items[ind2-1].split('_')[0]
                    dict_parse[two_index[ind]]['pos'] = two_pos[ind2-1]
                    dict_parse[two_index[ind]]['head'] = two_index[ind+1]
                    two_items.remove(two_items[ind+1])
                else:
                    dict_parse[two_index[ind+1]]['form'] = two_items[ind2].split('_')[0]
                    dict_parse[two_index[ind+1]]['pos'] = two_pos[ind2]
                    dict_parse[two_index[ind+1]]['head'] = two_index[ind]
                    two_items.remove(two_items[ind])
            else:
                dict_parse[two_index[ind-1]]['form'] = two_items[ind2].split('_')[0]
                dict_parse[two_index[ind-1]]['pos'] = two_pos[ind2]
                dict_parse[two_index[ind-1]]['head'] = two_index[ind]
                two_items.remove(two_items[ind])
            new_parse = parse[:]
            new_parse.remove(item)
            new_parse.remove(two_items[0])
            break

        if len(two_items) == 2:
            del two_items[0]
            del two_pos[0]
            del two_index[0]
        two_items.append(item)
        match = re.search(r'_(.+)_', item)
        match2 = re.search(r'_(\d+)', item)
        if match:
            pos = match.group(1)
            index = match2.group(1)
            two_pos.append(pos)
            two_index.append(index)
    print()

    try:
        print(new_parse)
    except UnboundLocalError:
        while True:
            print('\nThe sentence could not be parsed.\n')
            answer_9 = input("Press 'c' to  continue.\n")
            if answer_9 == 'c':
                return 0
    # If only one token remains, assign the head 0 to it,
    # that signals the head of the whole sentence.
    if len(new_parse) == 1:
        token, pos, index = new_parse[0].split('_')
        dict_parse[index]['form'] = token
        dict_parse[index]['pos'] = pos
        dict_parse[index]['head'] = 0
        # Convert the dictionary into an ordered dictionary sorted ba index of the tokens.
        dict_parse_final = OrderedDict(sorted(dict_parse.items(), key=lambda x: int(x[0])))
        print('\n\n')
        for key, value in dict_parse_final.items():
            value = dict(value)
            print(key)
            print(value)
            print()

        return dict_parse_final

    return conll_format(new_parse, dict_parse)


def flatten(nested_list):
    """
    Converts the nested list that represents the parsed sentence into a flattened
    sequence of tokens with corresponding pos and index, as well as the pos of the head
    after each bigram that should be reduced according to a rule.
    Given the nested nature of the input parameter, this function is recursive.
    This function is called from the function parse_sentence.

    Parameters:
        - nested_list: a nested list representing the structure of the parsed sentence.
          Example of a nested list, representing the toy sentence:
          [[[[[['The_DT_1', 'cat_NN_2'], 'NN_Right'], 'sleeps_VBZ_3'], 'VBZ_Right'],
          [['in_IN_4', [['the_DT_5', 'garden_NN_6'], 'NN_Right']], 'NN_Right']], 'VBZ_Left']

    Output:
        - a flattened version of the list.
          Example of the resulting flattend version:
          ['The_DT_1', cat_NN_2', 'NN_Right', 'sleeps_VBZ_3', 'VBZ_Right', 'in_IN_4',
           'the_DT_5','garden_NN_6', 'NN_Right', 'NN_Right', 'VBZ_Left']
    """
    for i in nested_list:
        if isinstance(i, list):
            for j in flatten(i):
                yield j
        else:
            yield i


def display_chart(chart, i, j):
    """
    Displays the chart on the console. This function is called from
    the function parse_sentence.

    Parameters:
        - chart: a dictionary with information about rules and their probabilities
          for each cell of the chart.
        - i, j: the coordinates of the top-right cell, which stores the final parse.

    Returns:
        - parse: the nested list representing the parsing.
    """
    print('\n\n')
    print('*'*25 + ' The chart ' + '*'*25 + '\n')
    rule_discarded = 0
    rule_accepted = 0
    for key, value in chart.items():
        print('Cell:', key)
        list_prob = []
        for key1, value1 in value.items():
            print(key1, '\n', dict(value1))
            # Independently of the probabilities of the rules in the top-right cell,
            # keep the rule that has a verb as head, since it is assumed that the
            # root of the sentence must be a verb.
            one_rule = 'no'
            if key == (i, j):
                if len(value) == 1:
                    one_rule = 'yes'
                list_prob.append(value1['Probability'])
                if not value1['Head'].startswith('V'):
                    discard_rule = key1
                    rule_discarded += 1
                else:
                    parse = value1
                    rule_accepted += 1
                if rule_discarded == 2:
                    parse = [value1 for value1 in value.values() \
                             if value1['Probability'] == max(list_prob)][0]
                if len(value) == 1 and rule_discarded == 1:
                    parse = [value1 for value1 in value.values()][0]
        print()
        print('-'*50)
    print()

    if rule_discarded != 0 and one_rule == 'no':
        del chart[(i, j)][discard_rule]
    print()

    try:
        parse = dict(parse)
    except:
        while True:
            print('\nThe sentence could not be parsed.\n')
            answer_8 = input("Press 'c' to  continue.\n")
            if answer_8 == 'c':
                return 0
    print('This is a representation of the nested structure of the parsing:\n')
    print(parse['Components'])
    print()

    return parse


def parse_sentence(grammar, input_sent):
    """
    Based on the rules that have been learned in the training phase,
    performs dependency parsing of the test sentence.
    Calls function flatten to flatten the nested structure of the parsing.
    Calls function conll_format to convert the parsing into conll format.
    This function is called from the main function.

    Parameters:
        - grammar: the set of rules learned in the training phase.
        - input_sent: the sentence to be parsed.

    Returns:
        - dict_parse_final: a dictionary with the conll format of the test sentence.
    """

    sentence_tok = word_tokenize(input_sent)
    postagger = StanfordPOSTagger('postagger/models/english-bidirectional-distsim.tagger', \
                                  'postagger/stanford-postagger.jar')
    sent = postagger.tag(sentence_tok)

    print('\n\nPOS tagging of the input sentence:\n')
    print(sent)
    while True:
        answer_10 = input("\nPress 'c' to  continue.\n")
        if answer_10 == 'c':
            break
    chart = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
    print()

    # Loop over the rows of the chart
    for j in range(1, len(sent)+1):
        # Fill cells 0,1 - 1,2 - 2,3 - 3,4 etc. (the cells with the pos of each terminal)
        pos = sent[j-1][1]
        terminal = sent[j-1][0]
        rule = 'Rule1'
        chart[(j-1, j)][rule]['Head'] = pos
        chart[(j-1, j)][rule]['Components'] = terminal + '_' + pos + '_' + str(j)
        chart[(j-1, j)][rule]['Probability'] = 1.0

        # Loop over the columns of the chart
        for i in range(j-2, -1, -1):
            print('\n\n')
            print('*'*25 + ' CELL ' + str(i) + ',' + str(j) + ' ' + '*'*25 + '\n')
            found = 'no'
            rule_n = ''
            replaced = 'no'
            # Move the separator to try different splits
            for k in range(i+1, j):
                # Take one pos from the left group of cells
                # and one pos from the bottom group of cells
                print('\n----- Processing cell {},{} from spans {},{} and {},{} -----\n'
                      .format(i, j, i, k, k, j))
                for rule_a in chart[(i, k)].keys():
                    for rule_b in chart[(k, j)].keys():
                        left_bottom_pos = chart[(i, k)][rule_a]['Head'], \
                                                chart[(k, j)][rule_b]['Head']
                        left_bottom_words = [chart[(i, k)][rule_a]['Components'], \
                                             chart[(k, j)][rule_b]['Components']]
                        left_bottom_prob = [chart[(i, k)][rule_a]['Probability'], \
                                            chart[(k, j)][rule_b]['Probability']]

                        # Use the two pos as left part of a rule
                        # (Example of rule: DT NN -> NN)
                        to_reduce = ' '.join(list(left_bottom_pos))

                        # If a rule is found for the two given pos,
                        # write the resulting right part of the rule in the cell
                        if to_reduce in grammar:
                            heads = []
                            for head, count in grammar[to_reduce].items():
                                position_head = max(count, key=lambda k: count[k])
                                heads.append((head, count[position_head], position_head))
                                all_counts = sum([head[1] for head in heads])
                            for head in heads:
                                prob_ = head[1] / all_counts
                                prob_cell = prob_ * left_bottom_prob[0] * left_bottom_prob[1]
                                print('\nFound rule for {} -> {}'.format(to_reduce, head[0]))
                                print('Position of the head: ', head[2])
                                print('Probability of rule:', prob_)
                                print('Probability of left component:', left_bottom_prob[0])
                                print('Probability of bottom component:', left_bottom_prob[1])
                                print('Total probability:', prob_cell)
                                print()

                                discard_rule = 'no'
                                for rule, value in chart[(i, j)].items():
                                    if head[0] == value['Head']:
                                        if prob_cell < value['Probability']:
                                            print('Rule discarded: found rule with higher probability for the same head\n')
                                            discard_rule = 'yes'
                                        else:
                                            chart[(i, j)][rule]['Components'] = [left_bottom_words,
                                                                                 head[0] + '_' + head[2]]
                                            chart[(i, j)][rule]['Probability'] = prob_cell
                                            replaced = 'yes'
                                if discard_rule == 'yes':
                                    break
                                else:
                                    if replaced != 'yes':
                                        if 'Rule1' in chart[(i, j)]:
                                            rule_n = 'Rule2'
                                        else:
                                            rule_n = 'Rule1'
                                    else:
                                        break
                                found = 'yes'
                                chart[(i, j)][rule_n]['Head'] = head[0]
                                chart[(i, j)][rule_n]['Components'] = [left_bottom_words, head[0] + '_' + head[2]]
                                chart[(i, j)][rule_n]['Probability'] = prob_cell
                        else:
                            print('No rule found for {}'. format(to_reduce))
                            print()
            # If no rule is found to fill the cell, write the '-' symbol in it
            if found == 'no':
                chart[(i, j)][rule]['Head'] = '-'
                chart[(i, j)][rule]['Components'] = left_bottom_words
                chart[(i, j)][rule]['Probability'] = 0
                print('### No rule found for cell {},{}'.format(i, j))
                print()
    print()
    while True:
        answer_5 = input("\nPlease press 'h' to display the whole chart for this sentence.\n")
        if answer_5 == 'h':
            break

    # Call function to display the chart
    parse = display_chart(chart, i, j)
    nested_parse = (parse['Components'])
    # Call function to flatten the nested parse
    flattened_parse = list(flatten(nested_parse))
    print()

    while True:
        answer_6 = input("Please press 'c' to convert the parse into conll format.\n")
        if answer_6 == 'c':
            dict_parse_final_ = conll_format(flattened_parse)
            break

    return dict_parse_final_


def split_data_set(fileids):
    """
    Randomly splits the data set into training and test set.
    This function is called from the main function.

    Parameters:
        - fileids: a list of the file ids of the sentences in the data set.

    Returns:
        - Two lists of file ids for the training and data set respectively.
    """
    test_set = random.sample(fileids, int((len(fileids)/10)))
    training_set = [fileid for fileid in fileids if fileid not in test_set]

    return test_set, training_set


def get_sentence(fileid, sd):
    """
    Converts the conll format of a sentence into a string.
    This function is called from the main function.

    Parameters:
        - fileid: the file id of the sentence to be converted.
        - sd: the module used to convert the parsing from constituency into dependencies.

    Returns:
        - a string representation of a sentence.
    """
    penn_sent = treebank.parsed_sents(fileid)[0]
    # Convert the syntax annotation into dependencies
    dep_sent = sd.convert_tree(str(penn_sent))
    list_tokens = []
    for token in dep_sent:
        word = token[1]
        list_tokens.append(word)
    sentence = ' '.join(list_tokens)

    return sentence


def gold_standard(fileid, sd):
    """
    Converts the constituency parsing of a test sentence into a dependency parsing,
    in order to obtain a reference for the same sentence annotated by the parser.
    This function is called from the main function.

    Parameters:
        - fileid: the file id of the sentence to be converted.
        - sd: the module used to convert the parsing from constituency into dependencies.

    Returns:
        - a reference sentence, parsed based on dependencies.
    """
    penn_sent = treebank.parsed_sents(fileid)[0]
    # Convert the syntax annotation into dependencies
    dep_sent = sd.convert_tree(str(penn_sent))
    return dep_sent


def print_test_training_set(test_set, training_set):
    """
    Prints on the console the ids of the test and training sets.
    This function is called from the main function.
    """
    print()
    print('-'*10, 'The test set consists of the following sentences', '-'*10)
    print()
    print(test_set)
    print('\n\n')
    print('-'*10, 'The training set consists of the following sentences', '-'*10)
    print()
    print(training_set)


def evaluation(ref_sentence, dict_parse_final):
    """
    Performs evaluation of the parsed sentence based on comparison with a reference.
    This function is called from the main function.

    Parameters:
        - ref_sentence: the reference sentence in conll format.
        - dict_parse_final: a dictionary whose value is the parsed sentence in conll format.

    Output:
        - UAS (Unlabeled Accuracy Score) of the parsed sentence is printed on the console.
    """
    correct = 0
    print()
    for token, value in zip(ref_sentence, dict_parse_final.values()):
        print(token[0])
        print('Reference =', 'form:', token[1], '\tpos:', token[4], '\thead:', token[6])
        print('Parser output =', dict(value))
        print()
        real_head = token[6]
        pred_head = int(value['head'])
        if real_head == pred_head:
            correct += 1
    accuracy = correct / len(ref_sentence)
    print('\n' + '*'*25 + ' Results of the evaluation ' + '*'*25 + '\n')
    print('Correct: ', correct)
    print('Accuracy (UAS): ', round(accuracy, 3))
    print()
    print('*'*50)
    while True:
        answer_7 = input("\nPlease press 's' to parse another sentence.\n")
        if answer_7 == 's':
            return


def main():
    """
    Imports a data set from the Penn Treebank.
    Splits the data set into training and test set.
    Calls the functions to learn a grammar and to perform parsing and evaluation.
    """
    fileids = treebank.fileids()
    # This file is removed for being no complete sentence:
    fileids.remove('wsj_0056.mrg')
    sd = StanfordDependencies.get_instance(backend='subprocess')

    # Toy sentence, not extracted from the Penn Treebank. No gold standard is
    # available for this sentence, therefore evaluation must be done manually.
    # In order to allow a better visual representation of the chart in the attached
    # report, the period at the end of the sentence is omitted.
    simple_sent = 'The cat sleeps in the garden'

    os.system('clear')

    while True:
        answer_1 = input('\nPlease select one of the following actions (1, 2):\n\n' \
                         '1 - Use the same splitting in training and test set that was used ' \
                         'for the evaluation illustrated in the report. If you choose this ' \
                         'option, the parser will use an already existing grammar.\n\n'
                         '2 - Randomly split the data set into training and test set and ' \
                         'perform a new evaluation. If you choose this option, the parser will ' \
                         'have to learn a new grammar based on the training set. ' \
                         'This might take a few minutes.\n')
        if answer_1 in ['1', '2']:
            break

    if answer_1 == '1':
        test_set = ['wsj_0004.mrg', 'wsj_0008.mrg', 'wsj_0014.mrg', 'wsj_0050.mrg', \
                    'wsj_0063.mrg', 'wsj_0065.mrg', 'wsj_0070.mrg', 'wsj_0073.mrg', \
                    'wsj_0089.mrg', 'wsj_0096.mrg', 'wsj_0099.mrg', 'wsj_0118.mrg', \
                    'wsj_0120.mrg', 'wsj_0137.mrg', 'wsj_0144.mrg', 'wsj_0165.mrg', \
                    'wsj_0171.mrg', 'wsj_0181.mrg', 'wsj_0182.mrg', 'wsj_0199.mrg']

        training_set = [fileid for fileid in fileids if fileid not in test_set]
        print_test_training_set(test_set, training_set)

        with open('grammar_reduced.json', 'rb') as data_file:
            grammar = json.load(data_file)

    elif answer_1 == '2':
        test_set, training_set = split_data_set(fileids)
        print_test_training_set(test_set, training_set)
        while True:
            answer_2 = input("\nPlease press 't' to proceed with the training.\n")
            if answer_2 == 't':
                break

        tic = time.time()
        # Call function to learn a grammar from the Penn Treebank
        grammar = learn_grammar(training_set, sd)
        toc = time.time() - tic
        print('\nIt took {0:.4f} seconds to learn a grammar'.format(toc))

    while True:
        answer_3 = input("\nPlease press 'p' to proceed with the parsing.\n")
        if answer_3 == 'p':
            break
    valid = []
    test_sentences = []
    for i, fileid in enumerate(test_set):
        # Call function to convert the conll format into a string
        input_sent_test = get_sentence(fileid, sd)
        test_sentences.append(input_sent_test)
        print()
        print('Converting sentence n. {} into string form...'.format(i+1))
        valid.append(str(i+1))

    quit_program = 'no'
    while True:
        print('\n\nThis is a list of the sentences in the test set:\n')
        for i, (sentence, fileid) in enumerate(zip(test_sentences, test_set)):
            print()
            print(str(i+1) + ') id: ' + fileid + '\tSentence length: ' + str(len(sentence.split())))
            print(sentence)
            print()
        print('-'*40)
        print()
        while True:
            to_parse = input('\nPlease select the number of the sentence to be parsed.\n' \
                             "Press 't' to parse the toy sentence that is described in the report.\n" \
                             "Press 's' to type your own sentence.\n" \
                             "Press 'q' to quit.\n")
            if to_parse in valid or to_parse in ['t', 's']:
                break
            elif to_parse == 'q':
                quit_program = 'yes'
                break

        if quit_program == 'yes':
            break
        if to_parse == 't':
            test_sent = simple_sent
        elif to_parse == 's':
            test_sent = input('Please type the sentence you would like to parse.\n')
        else:
            test_sent = test_sentences[int(to_parse)-1]
        # Call function to perform the parsing of the selected sentence
        dict_parse_final = parse_sentence(grammar, test_sent)
        if dict_parse_final == 0:
            continue
        if to_parse in ['t', 's']:
            print('\nNo gold standard available for the present sentence\n')
            while True:
                answer_9 = input("\nPlease press 's' to parse another sentence.\n")
                if answer_9 == 's':
                    break
        else:
            print()
            print('*'*50)
            while True:
                answer_4 = input("\nPlease press 'e' to proceed with the evaluation.\n")
                if answer_4 == 'e':
                    ref_sentence = gold_standard(test_set[int(to_parse)-1], sd)
                    # Call function to perform the evaluation
                    evaluation(ref_sentence, dict_parse_final)
                    break

if __name__ == '__main__':
    main()
