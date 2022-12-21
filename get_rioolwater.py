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

def scrape_rioolwater():
    """Scrape rioolwaterdata van de RIVM site. Dit is verpakt in een stuk javascript met JSON
    """    
    res = requests.get("https://coronadashboard.rijksoverheid.nl/landelijk/rioolwater") # your link here
    soup = bs4.BeautifulSoup(res.content, features="lxml")
    item=soup.select_one('script[id="__NEXT_DATA__"]').text

    jsondata=json.loads(item)

    # print and save the JSON in a nice way
    # output = (json.dumps(jsondata, indent=4))
    # with open('output_rioolwater.txt', 'w') as f:
    #     f.write(output)

    # TODO: see if using named tuples is better https://www.youtube.com/watch?v=BlVciXgsBYI

    l=[]
    columns = ["date_unix","value_rivm_official", "date_rivm"]
    for i in jsondata["props"]["pageProps"]["selectedNlData"]["sewer"]["values"]:
        date_unix =  (i["date_unix"])
        value_rivm_official =  (i["average"])
        date_rivm = get_normal_date(date_unix)
        
        l.append([date_unix,value_rivm_official, date_rivm])    

    total_df = pd.DataFrame(l, columns=columns)
    total_df["date_rivm"] =  pd.to_datetime(total_df["date_rivm"] , format="%Y-%m-%d")
    total_df['year_number'] = total_df['date_rivm'].dt.isocalendar().year
    total_df['week_number'] = total_df['date_rivm'].dt.isocalendar().week
    total_df["weeknr"] = total_df["year_number"].astype(str) +"_" + total_df["week_number"].astype(str).str.zfill(2)
    total_df["value_rivm_official_sma"] =  total_df["value_rivm_official"].rolling(window = 5, center = False).mean()
    df_grouped = total_df.groupby([total_df["weeknr"]], sort=True).mean().reset_index()
    return total_df,df_grouped

def main():
    df,df_grouped = scrape_rioolwater()
   
    print(df_grouped)
    
if __name__ == "__main__":
    main()

