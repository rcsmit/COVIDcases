# 27/28 feb 2021
# Calculate the relation between gliding R and mobility (Apple and Google)
# Calculate the corelation with hospital admissions and factors mentioned above
# Plotting a heatmap with correlations
# Plotting a scattermap
# Plotting a graph in time, with an adjusted x-

# 1 maart 2021
# Merging files on date in different date formats
# Remove outliers (doesnt work)
# Calculating moving avarages
# Make different statistics for weekdays and weekend
# Scraping statistics from RIVM

# 2 maart
# R van ziekenhuisopnames
# weekgrafiek
# corrigeren vd merge functie

# 3 maart
# added restrictions (file van @HK_nien, MIT-licence)
# downloaden en mergen hospital admission
# downloaden en mergen r-getal RIVM
# alles omgeFzet in functies

# 4 maart
# meer onderverdeling in functies. Alles aan te roepen vanuit main() met parameters

# 5 maart
# custom colors
# weekend different color in barplot
# annoying problem met een join (van outer naar inner naar outer en toen werkte het weer)
# R value (14 days back due to smoothing)

#6 maart
# last row in bar-graph was omitted due to ["date of statistics"] instead of ["date"] in addwalkingR
# Bug wit an reset.index() somewhere. Took a long time to find out
# Tried to first calculate SMA and R, and then cut of FROM/UNTIL. Doesnt
# work. Took also a huge amount of time. Reversed everything afterwards

# 7 maart
# weekgraph function with parameters

# 8 maart
# find columns with max correlation
# find the timelag between twee columns
# added a second way to calculate and display R

# 9-11 maart: Grafieken van Dissel : bezetting bedden vs R

# 12 maart
# Genormeerde grafiek (max = 1 of begin = 100)
# Various Tg vd de R-number-curves

#14 maart
# Streamlit :)


#15 maart
# weekgrafieken
#  Series.dt.isocalendar().week
# first value genormeerde grafiek
# bredere grafiek
# leegmaken vd cache DONE

#16 maart
# knmi gegevens
# alle databronnen in een lijst ipv aparte inlaad-routines

#17 maart
# rioolwaardes
# scenarios
# Q en q toevoegd
# q > 6


# kiezen welke R je wilt in de bargraph
# waarde dropdown anders dan zichtbaar

