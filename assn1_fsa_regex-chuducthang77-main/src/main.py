import argparse
import csv
import os
import re
import pandas as pd  # used to sort csv data


def load_file(input_path, file):
    """
    This function loads the file and returns the content of the file
    :param input_path: str
    :param file: str
    :return: contents
    """
    with open(input_path + '/' + file, 'r') as f:
        contents = f.read()
    return contents


def write_file(output_path, results):
    """
    This function saves the result to csv file
    :param output_path: str
    :param results: str
    :return: None
    """
    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(results)


def main(input_path, output_path):
    """
    This is a main function
    :param input_path: str
    :param output_path: str
    :return: None
    """
    # list files from input dir
    files = os.listdir(input_path)
    files.sort()

    results = [['article_id', 'expr_type', 'value', 'offset']]

    # loop over all input files
    for file in files:
        # Load the current file
        contents = load_file(input_path, file)

        # part-of-decades or relative decades (for example, early 1990s, mid-1980s, late 2000s)
        for m in re.finditer(
                r'\b(early|mid|late)[\s-]\d{3}0s\b', contents):
            results.append([file, 'part-of-decades', m.group(), m.start()])

        # decade (ex. the 1990s and short form ex. the 70s)
        for m in re.finditer(
                r'\bthe\s(\d{3}|\d)0s\b', contents):
            results.append([file, 'decade', m.group(), m.start()])

        # day-month-year (for example 5 May, 1994)
        for m in re.finditer(
                r'\b\d{1,2}\s(January|February|March|April|May|June|July|August|September|October|November|December),?\s\d{4}\b',
                contents):
            results.append([file, 'day-month-year', m.group(), m.start()])

        # month-day-year (ex September 1 2022)
        for m in re.finditer(
                r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s\d{1,2},?\s\d{4}\b', contents):
            results.append([file, 'month-day-year', m.group(), m.start()])

        # part-of-year (for example, early 2000, mid-1999, late 1999)
        for m in re.finditer(
                r'\b(early|mid|late)(\s|-)\d{4}\b', contents):
            results.append([file, 'part-of-year', m.group(), m.start()])

        # month-year (for example: May, 1994). Excluding overlapping case for day-month-year.
        for m in re.finditer(
                r'\b(?<!\d\s)(?<!\d{2}\s)(January|February|March|April|May|June|July|August|September|October|November|December),?\s\d{4}\b',
                contents):
            results.append([file, 'month-year', m.group(), m.start()])

        # day-month (for example: 5 May) Excluding overlapping case for day-month-year.
        for m in re.finditer(
                r'\b\d{1,2}\s(January|February|March|April|May|June|July|August|September|October|November|December)(?!(\s\d{4}|,\s\d{4}))\b',
                contents):
            results.append([file, 'day-month', m.group(), m.start()])

        # month-day (for example: May 5th). Excluding overlapping case for month-day-year.
        for m in re.finditer(
                r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s\d{1,2}(st|rd|nd|th)?(?!(\s\d{4}|,\s\d{4}))\b',
                contents):
            results.append([file, 'month-day', m.group(), m.start()])

        # season-year (for example: Fall 2020)
        for m in re.finditer(
                r'\b([sS]pring|[sS]ummer|[fF]all|[aA]utumn|[wW]inter)\s\d{4}\b', contents):
            results.append([file, 'season-year', m.group(), m.start()])

        # quarter-year (for example: first quarter of 2000)
        for m in re.finditer(
                r'\b(first|second|third|fourth)\squarter\sof\s\d{4}\b', contents):
            results.append([file, 'quarter-year', m.group(), m.start()])

        # relative-week/month/year (for example: next/this/that/last/coming/following/previous year).
        # Excluding overlapping cases for part-of-relative-week/month/year and month, relative-year
        for m in re.finditer(
                r'\b(?<!(early)\s)(?<!(earlier)\s)(?<!(late)\s)(?<!(later)\s)(?<!(start\sof)\s)(?<!(end\sof)\s)'
                r'(?<!January\s)(?<!February\s)(?<!March\s)(?<!April\s)(?<!(May\s))(?<!June\s)(?<!July\s)(?<!August\s)' # Overlapping case for month, relative-year
                r'(?<!September\s)(?<!October\s)(?<!November\s)(?<!December\s)'
                r'(next|this|that|last|coming|following|previous)\s(week|month|year|weekend|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b', contents):
            results.append([file, 'relative-week/month/year/weekday/+', m.group(), m.start()])

        # part-of-relative-week/month/year (ex. late last year)
        for m in re.finditer(
                r'\b(earl(y|ier)|late|later|start\sof|end\sof)\s(the\s)?(next|this|that|last|coming|following|previous)\s(week|month|year)\b', contents):
            results.append([file, 'part-of-relative-week/month/year', m.group(), m.start()])

        # relative-years/months/weeks/days (ex. the following six weeks)
        for m in re.finditer(
                r'\b[tT]he\s(last|next|previous|following|coming)\s(\d+|two|three|four|five|six|seven|eight|nine|few)\s(years|months|weeks|days|decades)', contents):
            results.append([file, 'relative-years/months/weeks/days', m.group(), m.start()])

        # dayofweek (for example: Monday, Tuesday, Wednesday).
        # Excluding overlaping case for relative weekday.
        for m in re.finditer(
                r'\b(?<!next\s)(?<!this\s)(?<!last\s)(?<!coming\s)(?<!previous\s)(?<!following\s)(?<!that\s)'
                r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|weekend)\b', contents):
            results.append([file, 'dayofweek', m.group(), m.start()])

        # year (for example: 1990)
        # Excluding overlapping cases
        for m in re.finditer(
                r'\b(?<!early\s)(?<!mid-)(?<!late\s)(?<!mid\s)(?<!quarter of\s)' # overlapping case for relative-year, quarter-year
                r'(?<!spring\s)(?<!summer\s)(?<!fall\s)(?<!autumn\s)(?<!winter\s)' # overlapping case for season-year
                r'(?<!Spring\s)(?<!Summer\s)(?<!Fall\s)(?<!Autumn\s)(?<!Winter\s)'
                r'(?<!January\s)(?<!February\s)(?<!March\s)(?<!April\s)(?<!May\s)(?<!June\s)(?<!July\s)(?<!August\s)' # Overlapping case for month year
                r'(?<!September\s)(?<!October\s)(?<!November\s)(?<!December\s)' 
                r'(?<!January,\s)(?<!February,\s)(?<!March,\s)(?<!April,\s)(?<!May,\s)(?<!June,\s)(?<!July,\s)(?<!August,\s)'  # Overlapping case for month, year
                r'(?<!September,\s)(?<!October,\s)(?<!November,\s)(?<!December,\s)'
                r'(?<!\d\sJanuary\s)(?<!\d\sFebruary\s)(?<!\d\sMarch\s)(?<!\d\sApril\s)(?<!\d\sMay\s)(?<!\d\sJune\s)(?<!\d\sJuly\s)'# Overlapping case for day month year (single-digit day)
                r'(?<!\d\sAugust\s)(?<!\d\sSeptember\s)(?<!\d\sOctober\s)(?<!\d\sNovember\s)(?<!\d\sDecember\s)' 
                r'(?<!\d\sJanuary,\s)(?<!\d\sFebruary,\s)(?<!\d\sMarch,\s)(?<!\d\sApril,\s)(?<!\d\sMay,\s)(?<!\d\sJune,\s)' # Overlapping case for day month, year (single-digit day)
                r'(?<!\d\sJuly,\s)(?<!\d\sAugust,\s)(?<!\d\sSeptember,\s)(?<!\d\sOctober,\s)(?<!\d\sNovember,\s)(?<!\d\sDecember,\s)' 
                r'(?<!\d{2}\sJanuary\s)(?<!\d{2}\sFebruary\s)(?<!\d{2}\sMarch\s)(?<!\d{2}\sApril\s)(?<!\d{2}\sMay\s)(?<!\d{2}\sJune\s)' # Overlapping case for day month year (double-digit day)
                r'(?<!\d{2}\sJuly\s)(?<!\d{2}\sAugust\s)(?<!\d{2}\sSeptember\s)(?<!\d{2}\sOctober\s)(?<!\d{2}\sNovember\s)(?<!\d{2}\sDecember\s)' 
                r'(?<!\d{2}\sJanuary,\s)(?<!\d{2}\sFebruary,\s)(?<!\d{2}\sMarch,\s)(?<!\d{2}\sApril,\s)(?<!\d{2}\sMay,\s)'  # Overlapping case for day month, year (double-digit day)
                r'(?<!\d{2}\sJune,\s)(?<!\d{2}\sJuly,\s)(?<!\d{2}\sAugust,\s)(?<!\d{2}\sSeptember,\s)(?<!\d{2}\sOctober,\s)(?<!\d{2}\sNovember,\s)(?<!\d{2}\sDecember,\s)'
                r'(\d{4})\b', contents):
            results.append([file, 'year', m.group(), m.start()])

        # month, relative year. (ex. May last year)
        for m in re.finditer(
                r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s(of\s)?(the\s)?(next|this|that|last|coming|following|previous)\syear\b',
                contents):
            results.append([file, 'month, relative-year', m.group(), m.start()])

        # century (ex. 16th century)
        for m in re.finditer(
                r'\b\d+(th|st)\scentury\b', contents):
            results.append([file, 'century', m.group(), m.start()])

        # relative date (yesterday, tomorrow, today)
        for m in re.finditer(
                r'\b([yY]esterday|[tT]omorrow|[tT]oday)\b', contents):
            results.append([file, 'relative', m.group(), m.start()])

        # future (ex. in two months)
        for m in re.finditer(
                r'\b[iI]n\s(\d+|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|fourty|fifty)\s(days|weeks|months|years|decades)\b', contents):
            results.append([file, 'future-years/months/weeks/days', m.group(), m.start()])

        # past (ex. 50 years ago)
        for m in re.finditer(
                r'\b(\d+|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|fourty|fifty)\s(days|weeks|months|years|decades)\sago\b',
                contents):
            results.append([file, 'past-years/months/weeks/days', m.group(), m.start()])

    # Write the file
    write_file(output_path, results)

    # making data frame from output file
    data = pd.read_csv(output_path)

    # sorting data by article_id and offset
    data.sort_values(["article_id", "offset"], axis=0,
                     ascending=[True, True], inplace=True)

    # write sorted data back to output_path
    data.to_csv(output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract dates from news")
    parser.add_argument("input", type=str, default="data/dev/", help="Provide path to directory of data input")
    parser.add_argument("output", type=str, default="output/dev.csv", help="Provide path to directory of output")
    args = parser.parse_args()
    main(args.input, args.output)
