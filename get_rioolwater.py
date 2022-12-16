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


    l=[]
    columns = ["date_unix","value_rivm_official", "date_rivm"]
    for i in jsondata["props"]["pageProps"]["selectedNlData"]["sewer"]["values"]:
        date_unix =  (i["date_unix"])
        value_rivm_official =  (i["average"])
        date_rivm = get_normal_date(date_unix)

        l.append([date_unix,value_rivm_official, date_rivm])    

    total_df = pd.DataFrame(l, columns=columns)
    print (total_df)
  
    return total_df
def main():
    scrape_rioolwater()

if __name__ == "__main__":
    main()

