import pandas as pd

# original location  https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv
url ="C:\\Users\\rcxsm\\Documents\phyton_scripts\\covid19_seir_models\\input\\google_mob_world.csv"

df = pd.read_csv(url, delimiter=",", low_memory=False)
print (df)
#df = df.loc[df['sub_region_1'] == None]
df = df[df.sub_region_1.isnull()]
df = df[df.sub_region_2.isnull()]
df = df[df.metro_area.isnull()]
df = df[df.iso_3166_2_code.isnull()]
df = df[df.census_fips_code.isnull()]

print (df)
name_ = "C:\\Users\\rcxsm\\Documents\phyton_scripts\\covid19_seir_models\\input\\google_mob_world_new.csv"
compression_opts = dict(method=None, archive_name=name_)
df.to_csv(name_, index=False, compression=compression_opts)
print("--- Saving " + name_ + " ---")

