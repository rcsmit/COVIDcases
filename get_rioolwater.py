import json
import bs4
import requests
from datetime import datetime
import pandas as pd


def get_normal_date(unixdate):
    """ Convert the unixdate in a normal readable date
        https://stackoverflow.com/questions/3682748/converting-unix-timestamp-string-to-readable-date
    Args:
        unixdate (str): the unix date

    Returns:
        str : the readable date in yyyy-mm-dd
    """    
    ts = int(unixdate)

    # if you encounter a "year is out of range" error the timestamp
    # may be in milliseconds, try `ts /= 1000` in that case
    return (datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d'))



def load_data_from_csv():
    if platform.processor() != "":
        file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\rioolwaardes_official_rivm.csv"
    else: 
        file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/rioolwaardes_official_rivm.csv"
        df_ = pd.read_csv(
            file,
            delimiter=";",
            low_memory=False,
        )
    
    return df_



def save_df(df, name):
    """  _ _ _ """
    name_ =  name + ".csv"
    compression_opts = dict(method=None, archive_name=name_)
    df.to_csv(name_, index=False, compression=compression_opts)

    print("--- Saving " + name_ + " ---")



def scrape_data_from_site():
    res = requests.get("https://coronadashboard.rijksoverheid.nl/landelijk/rioolwater") # your link here
    soup = bs4.BeautifulSoup(res.content, features="lxml")
    item=soup.select_one('script[id="__NEXT_DATA__"]').text

    jsondata=json.loads(item)

    #print and save the JSON in a nice way
    output = (json.dumps(jsondata, indent=4))
    with open('output_rioolwater.txt', 'w') as f:
        f.write(output)

    # TODO: see if using named tuples is better https://www.youtube.com/watch?v=BlVciXgsBYI

    l=[]
    columns = ["date_unix","value_rivm_official", "date_rivm"]
    for i in jsondata["props"]["pageProps"]["selectedNlData"]["sewer"]["values"]:
        date_unix =  (i["date_unix"])
        value_rivm_official =  (i["average"])
        date_rivm = get_normal_date(date_unix)
        
        l.append([date_unix,value_rivm_official, date_rivm])    

    total_df = pd.DataFrame(l, columns=columns)
    
    return total_df

def make_grouped_df(total_df):
    total_df["date_rivm"] =  pd.to_datetime(total_df["date_rivm"] , format="%Y-%m-%d")
    total_df['year_number'] = total_df['date_rivm'].dt.isocalendar().year
    total_df['week_number'] = total_df['date_rivm'].dt.isocalendar().week
    total_df["weeknr"] = total_df["year_number"].astype(str) +"_" + total_df["week_number"].astype(str).str.zfill(2)
    total_df["value_rivm_official_sma"] =  total_df["value_rivm_official"].rolling(window = 5, center = False).mean().round(1)
    df_grouped = total_df.groupby([total_df["weeknr"]], sort=True).mean().reset_index() #.round(1)
    return df_grouped

def scrape_rioolwater():
    """Scrape rioolwaterdata van de RIVM site. Dit is verpakt in een stuk javascript met JSON
    """    
    try:
        total_df = scrape_data_from_site()
    except:
        total_df = load_data_from_csv()

    df_grouped = make_grouped_df(total_df)
    return total_df

def main():
    total_df,df_grouped = scrape_rioolwater()
    print (total_df)
    print (df_grouped)
    save_df(df_grouped,"rioolwater_per_week")
    
if __name__ == "__main__":
    main()



    """"Toen Delta kwam, zagen we zowel in het rioolwater als bij de positieve testen een duidelijke stijging.
     Maar opvallend genoeg zagen we bij Omikron (BA.1) geen stijging in het rioolwater terwijl
      het aantal positieve testen flink steeg". Het verband tussen het rioolwater en het aantal
       positieve testen was bij BA.1 "helemaal weggevallen". "Met de opkomst van Omikron BA.2 
       was het verband tussen rioolwater en positieve testen weer terug."

       https://www.corona-lokaal.nl/locatie/Nederland/waterzuivering/Nederland
    """