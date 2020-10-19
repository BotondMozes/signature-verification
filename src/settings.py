# variables
MCYT_DATA = "/home/mozesbotond/WorkSpace/Signature Verification/data/MCYT/"
MOBISIG_DATA = "/home/mozesbotond/WorkSpace/Signature Verification/data/MOBISIG/"

MCYT_STAT_DATA = "/home/mozesbotond/WorkSpace/Signature Verification/output/stat/mcyt_stat.csv"
MOBISIG_STAT_DATA = "/home/mozesbotond/WorkSpace/Signature Verification/output/stat/mobisig_stat.csv"

MCYT_OUTPUT = "/home/mozesbotond/WorkSpace/Signature Verification/output/MCYT/DTW"
MOBISIG_OUTPUT = "/home/mozesbotond/WorkSpace/Signature Verification/output/MOBISIG/DTW"

MCYT_INTERP = "/home/mozesbotond/WorkSpace/Signature Verification/output/MCYT/interpolated/"
MOBISIG_INTERP = "/home/mozesbotond/WorkSpace/Signature Verification/output/MOBISIG/interpolated/"

FINAL_OUTPUT = "/home/mozesbotond/WorkSpace/Signature Verification/output/FINAL/"

MCYT_GENUINE = FINAL_OUTPUT + "mcyt_genuine.csv"
MCYT_FORGERY = FINAL_OUTPUT + "mcyt_forgery.csv"

MCYT_FIELDS = [0, 1, 2]
MOBISIG_FIELDS = [0, 1, 3]

# settings

INPUT = MOBISIG_DATA
OUTPUT = MOBISIG_OUTPUT
FIELDS = MOBISIG_FIELDS
LENGTH = 512
