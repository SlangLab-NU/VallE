import collections

with open("CF02_mlf.txt", 'r') as file:
    lines = file.readlines()
    codes = {}
    words = {}
    countUW = 0
    for line in lines:
        line = line.strip()

        # Skip comments or empty lines
        if line.startswith("#") or line.startswith(".") or not line:
            continue
        if line.startswith('"'):
            line = line.split('_')
            if line[2] in codes:
                codes[line[2]] += 1
            else:
                codes[line[2]] = 0
                if "UW" in line[2]:
                    countUW += 1
        else:
            if line in words:
                words[line] += 1
            else:
                words[line] = 0

    codes = collections.OrderedDict(sorted(codes.items()))
    totalUtterances = 0
    totalUniqueCodes = 0
    for key, val in codes.items():
        totalUniqueCodes += 1
        totalUtterances += val
        print(key, val)
    print(f"Total unique codes: {totalUniqueCodes}")
    print(f"Total utterances: {totalUtterances}")

    totalWords = 0
    totalUniqueWords = 0
    for key, val in words.items():
        totalUniqueWords += 1
        totalWords += val
        print(key, val)
    print(f"Total unique words: {totalUniqueWords}")
    print(f"Total words: {totalWords}")
    print(f"unique words: {countUW}")

# TODO take 50 random codes to make test set, then div that to 25/25 test and dev
