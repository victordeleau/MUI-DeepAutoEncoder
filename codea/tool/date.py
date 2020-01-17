import datetime

def get_day_month_year_hour_minute_second():

    d = datetime.date.today()

    t = str( datetime.datetime.now().time() ).split('.')[0].replace(':', '')

    return '{:02d}'.format(d.day) + '{:02d}'.format(d.month) + '{:02d}'.format(d.year) + "_" + t